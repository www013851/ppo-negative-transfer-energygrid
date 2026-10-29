"""Training stages of the experiment.

    pretrain        new model, phase_steps in world A                       -> runs/pretrain/{arch}_seed{seed}
    train_branch    from the final A model, phase_steps in world B under one
                    condition (plain continuation, training intervention or reset)
                                                                           -> runs/branches/{condition}_{arch}_seed{seed}
    train_control   scratch_B: new model in world B; no_shift: continue in A -> runs/controls/{condition}_{arch}_seed{seed}
    extend          continue a finished branch for another phase_steps        -> runs/extended/{condition}_{arch}_seed{seed}

All branches of one (architecture, seed) pair start from the same A checkpoint, use the same world-B environment
seed and the same restored global RNG state, so the paired difference between two conditions is caused by the
condition itself. Every eval_interval steps the deterministic policy is evaluated on fixed starts in world B
(world A for no_shift) and the hash of the policy weights is stored next to the episodes, which makes any rerun
checkable bit by bit.
"""
from pathlib import Path

import torch

from . import agents
from .env import EnergyGridEnv
from .evaluation import evaluate_agent
from .interventions import RESET_COMPONENTS, TRAINING_CONDITIONS, GradualShift, apply_reset, gradual_deltas
from .utils import (append_jsonl, capture_rng, evaluation_seeds, read_json, restore_rng, sha256, state_hash,
                    write_json)


def _start(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "status.json", {"status": "running"})
    return directory


def _checkpoint_and_evaluate(model, directory, prefix, step, world, identity, cfg):
    checkpoint = directory / f"{prefix}_{step:06d}.zip"
    model.save(checkpoint)
    rng = state_hash(capture_rng())
    episodes = evaluate_agent(model, world, evaluation_seeds(cfg), True, cfg["environment"])
    assert state_hash(capture_rng()) == rng, "evaluation must not disturb training randomness"
    policy = state_hash(model.policy.state_dict())
    append_jsonl(directory / "eval_records.jsonl",
                 [{**identity, "world": world, "step": step, "total_timesteps": int(model.num_timesteps),
                   "n_updates": int(model._n_updates), "policy_sha256": policy, **e} for e in episodes])
    return checkpoint


def _finish(directory, model, checkpoint, extra):
    write_json(directory / "metadata.json", {"status": "complete", **extra, "final_checkpoint": checkpoint.name,
                                             "final_policy_sha256": state_hash(model.policy.state_dict()),
                                             "final_optimizer_sha256": state_hash(model.policy.optimizer.state_dict())})
    write_json(directory / "status.json", {"status": "complete"})
    model.get_env().close()


def pretrain(runs, arch, seed, cfg):
    torch.set_num_threads(cfg["torch_threads"])
    directory = _start(Path(runs) / "pretrain" / f"{arch}_seed{seed}")
    identity = {"stage": "pretrain", "condition": "pretrain", "architecture": arch, "train_seed": seed}
    model = agents.new_model(arch, seed, "A", cfg)
    interval = cfg["eval_interval"]
    for step in range(interval, cfg["phase_steps"] + 1, interval):
        model.learn(interval, reset_num_timesteps=(step == interval))
        checkpoint = _checkpoint_and_evaluate(model, directory, "A", step, "A", identity, cfg)
    # Branches restore this snapshot, so they all continue the same random stream.
    write_json(directory / "rng_after_A.json", capture_rng())
    _finish(directory, model, checkpoint, {**identity, "parameters": agents.parameter_counts(model)})
    return directory


def _parent(runs, arch, seed):
    pre = Path(runs) / "pretrain" / f"{arch}_seed{seed}"
    meta = read_json(pre / "metadata.json")
    return pre, pre / meta["final_checkpoint"], meta


def train_branch(runs, arch, seed, condition, cfg):
    """phase_steps in world B from the final A model under one condition."""
    assert condition in TRAINING_CONDITIONS or condition in RESET_COMPONENTS, condition
    torch.set_num_threads(cfg["torch_threads"])
    directory = _start(Path(runs) / "branches" / f"{condition}_{arch}_seed{seed}")
    pre, checkpoint_a, parent = _parent(runs, arch, seed)

    env_overrides = None
    if condition == "high_energy":
        energy = cfg["high_energy_training"]
        env_overrides = {"initial_energy": energy, "max_energy": energy}
    env = EnergyGridEnv(phase="B", **{**cfg["environment"], **(env_overrides or {})})
    if condition == "gradual_shift":
        # Start the branch with world-A utilities; the callback moves them to world B.
        env.set_resource_deltas(*gradual_deltas(0, cfg["intervention_steps"], cfg["environment"]["resource_magnitude"]))
    model = agents.load_model(checkpoint_a, "B", cfg, env=env)
    assert state_hash(model.policy.state_dict()) == parent["final_policy_sha256"]
    assert state_hash(model.policy.optimizer.state_dict()) == parent["final_optimizer_sha256"]
    model.get_env().seed(seed + cfg["seed_offsets"]["world_b_environment"])
    model._last_obs = None  # force a reset into the newly seeded world-B environment
    restore_rng(read_json(pre / "rng_after_A.json"))
    audit = apply_reset(model, condition, seed, cfg["seed_offsets"]) if condition in RESET_COMPONENTS else None

    identity = {"stage": "branch", "condition": condition, "architecture": arch, "train_seed": seed}
    start = int(model.num_timesteps)
    interval, boundary = cfg["eval_interval"], cfg["intervention_steps"]
    schedule = (GradualShift(start, boundary, cfg["environment"]["resource_magnitude"])
                if condition == "gradual_shift" else None)
    checkpoint = _checkpoint_and_evaluate(model, directory, "B", 0, "B", identity, cfg)
    for step in range(interval, cfg["phase_steps"] + 1, interval):
        # Split a learning chunk only where the entropy coefficient changes inside it.
        cuts = [step - interval, step]
        if cuts[0] < boundary < cuts[1]:
            cuts.insert(1, boundary)
        for left, right in zip(cuts, cuts[1:]):
            model.ent_coef = (cfg["entropy_spike_coefficient"] if condition == "entropy_spike" and left < boundary
                              else cfg["ppo"]["ent_coef"])
            model.learn(right - left, callback=schedule, reset_num_timesteps=False)
        assert model.num_timesteps == start + step
        checkpoint = _checkpoint_and_evaluate(model, directory, "B", step, "B", identity, cfg)
    _finish(directory, model, checkpoint, {**identity, "parent_checkpoint_sha256": sha256(checkpoint_a),
                                           "reset": audit, "training_environment_overrides": env_overrides})
    return directory


def train_control(runs, arch, seed, condition, cfg):
    """scratch_B: a new model trained in world B; no_shift: the A model trained further in world A."""
    torch.set_num_threads(cfg["torch_threads"])
    directory = _start(Path(runs) / "controls" / f"{condition}_{arch}_seed{seed}")
    if condition == "scratch_B":
        world = "B"
        model = agents.new_model(arch, seed, "B", cfg)
    elif condition == "no_shift":
        world = "A"
        pre, checkpoint_a, parent = _parent(runs, arch, seed)
        model = agents.load_model(checkpoint_a, "A", cfg)
        assert state_hash(model.policy.state_dict()) == parent["final_policy_sha256"]
        restore_rng(read_json(pre / "rng_after_A.json"))
    else:
        raise ValueError(condition)
    model.get_env().seed(seed + cfg["seed_offsets"]["world_b_environment"])
    model._last_obs = None
    identity = {"stage": "control", "condition": condition, "architecture": arch, "train_seed": seed}
    interval = cfg["eval_interval"]
    checkpoint = _checkpoint_and_evaluate(model, directory, "C", 0, world, identity, cfg)
    for step in range(interval, cfg["phase_steps"] + 1, interval):
        model.learn(interval, reset_num_timesteps=False)
        checkpoint = _checkpoint_and_evaluate(model, directory, "C", step, world, identity, cfg)
    _finish(directory, model, checkpoint, {**identity, "training_world": world})
    return directory


def extend(runs, arch, seed, condition, cfg, label="head_reset_extended"):
    """Continue a finished world-B branch for another phase_steps (Г+: the head-reset branch, labelled `label`)."""
    torch.set_num_threads(cfg["torch_threads"])
    source = Path(runs) / "branches" / f"{condition}_{arch}_seed{seed}"
    meta = read_json(source / "metadata.json")
    directory = _start(Path(runs) / "extended" / f"{label}_{arch}_seed{seed}")
    model = agents.load_model(source / meta["final_checkpoint"], "B", cfg)
    offset = cfg["seed_offsets"]["extension_environment_and_torch"]
    model.get_env().seed(seed + offset)
    model._last_obs = None
    torch.manual_seed(seed + offset)
    identity = {"stage": "extension", "condition": label, "architecture": arch, "train_seed": seed}
    interval, phase = cfg["eval_interval"], cfg["phase_steps"]
    for step in range(phase + interval, 2 * phase + 1, interval):
        model.learn(interval, reset_num_timesteps=False)
        checkpoint = _checkpoint_and_evaluate(model, directory, "B", step, "B", identity, cfg)
    _finish(directory, model, checkpoint, {**identity, "source_checkpoint_sha256": sha256(source / meta["final_checkpoint"])})
    return directory
