"""Behavioural diagnostics of trained policies.

* rollouts: episodes in world B on held-out starts, with the greedy (argmax) or the sampled action;
* direction: whether an action brings the agent closer only to the new useful resource (B), only to the former
  one (A), to both or to neither -- four mutually exclusive categories that sum to 100 %;
* stuck episodes: the agent died after spending its last `window` steps in at most `max_cells` cells without eating;
* hidden-layer activations: share of dormant units (Sokar et al., 2023: E|h_i| / layer mean <= 0.1), share of
  saturated Tanh units (E|h_i| > 0.95) and the effective rank of the last hidden layer (99 % of singular values);
* a probe of uniformly random valid states, identical for every model, for entropy and direction on common inputs.
"""
import numpy as np
import torch

from .agents import action_probabilities
from .env import EnergyGridEnv

CATEGORIES = ["new_only", "old_only", "both", "neither"]


def toward(dx, dy, action):
    """True where `action` reduces the distance to a target at offset (dx, dy)."""
    return np.select([action == 0, action == 1, action == 2, action == 3], [dy > 0, dy < 0, dx < 0, dx > 0]).astype(bool)


def direction_shares(obs, actions):
    new = toward(obs[..., 4], obs[..., 5], actions)  # resource B: useful in world B
    old = toward(obs[..., 2], obs[..., 3], actions)  # resource A: useful in world A
    return {"new_only": float((new & ~old).mean()), "old_only": float((old & ~new).mean()),
            "both": float((new & old).mean()), "neither": float((~new & ~old).mean())}


def rollout(policy, deterministic, seeds, env_cfg, action_seed, stuck_window=30, stuck_max_cells=2):
    """Episodes in world B. Returns per-episode summaries and the visited observations with chosen actions."""
    env = EnergyGridEnv(phase="B", **env_cfg)
    episodes, observations, actions = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        rng = np.random.default_rng(np.random.SeedSequence([action_seed, s]))
        positions, eaten = [tuple(env.agent_pos)], []
        ben = harm = blocked = 0
        done = terminated = False
        while not done:
            p = action_probabilities(policy, obs[None])[0]
            a = int(p.argmax()) if deterministic else int(rng.choice(4, p=p / p.sum()))
            observations.append(obs)
            actions.append(a)
            before = tuple(env.agent_pos)
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            blocked += tuple(env.agent_pos) == before
            positions.append(tuple(env.agent_pos))
            eaten.append(info["beneficial_consumed"] + info["harmful_consumed"])
            ben += info["beneficial_consumed"]
            harm += info["harmful_consumed"]
        n = len(eaten)
        stuck = bool(terminated and len(set(positions[-(stuck_window + 1):])) <= stuck_max_cells
                     and sum(eaten[max(0, n - stuck_window):]) == 0)
        swings = sum(positions[i] == positions[i - 2] and positions[i] != positions[i - 1] for i in range(2, len(positions)))
        episodes.append({"eval_seed": s, "length": n, "beneficial": ben, "harmful": harm, "died": bool(terminated),
                         "stuck": stuck, "blocked_fraction": blocked / n, "swing_fraction": swings / n})
    return episodes, np.asarray(observations), np.asarray(actions)


def hidden_activations(policy, obs):
    x = torch.as_tensor(obs, dtype=torch.float32)
    outs = []
    with torch.no_grad():
        for layer in policy.mlp_extractor.policy_net:
            x = layer(x)
            if isinstance(layer, torch.nn.Tanh):
                outs.append(x.numpy())
    return outs


def activation_stats(policy, obs):
    layers = hidden_activations(policy, obs)
    if not layers:
        return None
    per_layer = []
    for h in layers:
        mean_abs = np.abs(h).mean(axis=0)
        score = mean_abs / max(mean_abs.mean(), 1e-12)
        per_layer.append({"dormant": float((score <= 0.1).mean()), "saturated": float((mean_abs > 0.95).mean())})
    sv = np.linalg.svd(layers[-1], compute_uv=False)
    srank = int(np.searchsorted(np.cumsum(sv) / sv.sum(), 0.99) + 1)
    return {"layers": per_layer, "srank_last": srank, "width": layers[-1].shape[1]}


def make_probe(seed, n):
    """Uniformly random valid states (three distinct cells, energy 1..100); the same array for every model."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        cells = rng.choice(100, 3, replace=False)
        x, y = cells[0] % 10, cells[0] // 10
        ax, ay = cells[1] % 10, cells[1] // 10
        bx, by = cells[2] % 10, cells[2] // 10
        rows.append([x / 9, y / 9, (ax - x) / 9, (ay - y) / 9, (bx - x) / 9, (by - y) / 9, rng.integers(1, 101) / 100])
    return np.asarray(rows, dtype=np.float32)


def probe_metrics(probabilities, obs):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(axis=1)
    return {"entropy_fraction_of_max": float(entropy.mean() / np.log(4)),
            **direction_shares(obs, p.argmax(axis=1))}
