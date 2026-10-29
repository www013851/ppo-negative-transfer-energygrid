"""Evaluation on fixed start positions. It uses its own environment and never touches the training RNG."""
import numpy as np

from .env import EnergyGridEnv


def evaluate_agent(model, world, eval_seeds, deterministic=True, env_params=None, random_action_seed=123456):
    """Play one episode per evaluation seed.

    `model` is a PPO model, None (uniformly random actions) or the string "fixed_left".
    Deterministic evaluation takes the most probable action of the policy.
    """
    env = EnergyGridEnv(phase=world, **(env_params or {}))
    records = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        action_rng = np.random.default_rng(np.random.SeedSequence([random_action_seed, int(s)]))
        done = False
        length, ret, ben, harm = 0, 0.0, 0, 0
        terminated = truncated = False
        while not done:
            if model is None:
                action = int(action_rng.integers(4))
            elif isinstance(model, str) and model == "fixed_left":
                action = 2
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                if isinstance(action, np.ndarray):
                    action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            length += 1
            ret += reward
            ben += info["beneficial_consumed"]
            harm += info["harmful_consumed"]
            done = terminated or truncated
        records.append({"eval_seed": s, "episode_length": length, "return": float(ret),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "beneficial_consumed": ben, "harmful_consumed": harm})
    env.close()
    return records


def simple_baselines(cfg, eval_seeds):
    """Random and always-left strategies in both worlds: reference points for 'surviving without food'."""
    rows = []
    for world in ["A", "B"]:
        for strategy in cfg["baselines"]["strategies"]:
            episodes = evaluate_agent(None if strategy == "random" else strategy, world, eval_seeds,
                                      env_params=cfg["environment"], random_action_seed=cfg["baselines"]["action_seed"])
            rows += [{"strategy": strategy, "world": world, **r} for r in episodes]
    return rows
