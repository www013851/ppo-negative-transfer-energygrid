"""Behaviour of the final models in world B (Fig. 2 and the mechanism of the deterministic/stochastic gap).

For every (architecture, seed) and every final model:
  * 100 held-out episodes with the greedy action and 100 with the sampled action;
  * direction shares of the chosen actions on the states the agent actually visited;
  * hidden-layer activation statistics on a common pool of visited states (models A, К, Н, Г of the same seed);
  * entropy and direction on 2048 uniformly random probe states;
  * the same probe for every reset applied to the world-A model, i.e. the policy right at the start of the branch.

Usage:  python scripts/evaluate_behaviour.py --checkpoints CHECKPOINT_DIR [--workers 8]
Output: data/rollout_episodes.csv.gz, data/behaviour.csv, data/probe_states.npy
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import gzip
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energygrid_transfer import agents, behaviour  # noqa: E402
from energygrid_transfer.interventions import RESET_COMPONENTS, applies_to  # noqa: E402
from energygrid_transfer.utils import REPO, evaluation_seeds, load_config  # noqa: E402

MODELS = ["pretrain", "baseline", "scratch_B", "optimizer_reset", "head_optimizer_reset", "hidden_optimizer_reset",
          "critic_optimizer_reset", "critic_head_optimizer_reset", "head_reset_extended"]
ACTIVATION_POOL = ["pretrain", "baseline", "scratch_B", "head_optimizer_reset"]
EPISODE_FIELDS = ["architecture", "train_seed", "model", "mode", "eval_seed", "length", "beneficial", "harmful",
                  "died", "stuck", "blocked_fraction", "swing_fraction"]
BEHAVIOUR_FIELDS = ["architecture", "train_seed", "model", "measurement", "new_only", "old_only", "both", "neither",
                    "entropy_fraction_of_max", "dormant_max_layer", "saturated_max_layer", "srank_last", "width"]


def available(arch, model):
    if model == "head_reset_extended":
        return arch != "linear"
    return applies_to(model, arch)


def task(checkpoints, arch, seed):
    import torch
    torch.set_num_threads(1)
    cfg = load_config()
    ro = cfg["rollouts"]
    seeds = evaluation_seeds(cfg, "rollouts")
    probe = behaviour.make_probe(cfg["probe"]["seed"], cfg["probe"]["states"])
    episodes, rows, visited = [], [], {}
    policies = {}
    for model in [m for m in MODELS if available(arch, m)]:
        policy = agents.load_model(Path(checkpoints) / model / f"{arch}_seed{seed}.zip", "B", cfg).policy
        policy.set_training_mode(False)
        policies[model] = policy
        for mode, deterministic in [("deterministic", True), ("stochastic", False)]:
            eps, obs, act = behaviour.rollout(policy, deterministic, seeds, cfg["environment"], ro["action_seed"],
                                              ro["stuck_window"], ro["stuck_max_cells"])
            episodes += [{"architecture": arch, "train_seed": seed, "model": model, "mode": mode, **e} for e in eps]
            rows.append({"architecture": arch, "train_seed": seed, "model": model, "measurement": f"visited_{mode}",
                         **behaviour.direction_shares(obs, act)})
            if deterministic:
                visited[model] = obs[:: max(1, len(obs) // 1500)]
        rows.append({"architecture": arch, "train_seed": seed, "model": model, "measurement": "probe_final",
                     **behaviour.probe_metrics(agents.action_probabilities(policy, probe), probe)})
    pool = np.concatenate([visited[m] for m in ACTIVATION_POOL])
    for model, policy in policies.items():
        act = behaviour.activation_stats(policy, pool)
        if act:
            rows.append({"architecture": arch, "train_seed": seed, "model": model, "measurement": "activations",
                         "dormant_max_layer": max(layer["dormant"] for layer in act["layers"]),
                         "saturated_max_layer": max(layer["saturated"] for layer in act["layers"]),
                         "srank_last": act["srank_last"], "width": act["width"]})
    # The policy right after each reset: apply the reset to the world-A model in memory.
    from energygrid_transfer.interventions import apply_reset
    for condition in RESET_COMPONENTS:
        if not applies_to(condition, arch):
            continue
        model = agents.load_model(Path(checkpoints) / "pretrain" / f"{arch}_seed{seed}.zip", "B", cfg)
        apply_reset(model, condition, seed, cfg["seed_offsets"])
        model.policy.set_training_mode(False)
        rows.append({"architecture": arch, "train_seed": seed, "model": condition, "measurement": "probe_after_reset",
                     **behaviour.probe_metrics(agents.action_probabilities(model.policy, probe), probe)})
    return episodes, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    tasks = [(a, s) for a in cfg["actor_architectures"] for s in sum(cfg["train_seeds"].values(), [])]
    episodes, rows = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(task, args.checkpoints, a, s) for a, s in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            e, r = f.result()
            episodes += e
            rows += r
            print(f"done {i}/{len(tasks)}", flush=True)
    key = lambda r: (r["architecture"], r["train_seed"], r["model"], r.get("mode", r.get("measurement")), r.get("eval_seed", 0))
    with gzip.open(REPO / "data" / "rollout_episodes.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, EPISODE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(({**e, "died": int(e["died"]), "stuck": int(e["stuck"])} for e in episodes), key=key))
    with (REPO / "data" / "behaviour.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, BEHAVIOUR_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=key))
    np.save(REPO / "data" / "probe_states.npy", behaviour.make_probe(cfg["probe"]["seed"], cfg["probe"]["states"]))


if __name__ == "__main__":
    main()
