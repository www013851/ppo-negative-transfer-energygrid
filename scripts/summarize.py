"""All numbers reported in the paper, computed from data/ only.

Final length L of a run = mean episode length over the last `final_last_checkpoints` evaluations
(30 fixed starts each); food = mean number of useful resources eaten per episode at the same checkpoints.

Usage:  python scripts/summarize.py
Output: results/summary.json, results/table1.md
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energygrid_transfer.stats import PairedTest, holm  # noqa: E402
from energygrid_transfer.utils import REPO, load_config, write_json  # noqa: E402

LABELS = {"baseline": "К", "entropy_spike": "Э", "gradual_shift": "П", "high_energy": "Б250", "optimizer_reset": "О",
          "critic_optimizer_reset": "Кр", "head_optimizer_reset": "Г", "critic_head_optimizer_reset": "Кр+Г",
          "hidden_optimizer_reset": "С", "scratch_B": "Н", "no_shift": "А→А", "head_reset_extended": "Г+"}
ARCH_NAMES = {"linear": "Linear", "mlp16": "MLP16×16", "mlp64": "MLP64×64", "mlp256": "MLP256×256"}


def final_runs(cfg):
    ev = pd.read_csv(REPO / "data" / "evaluations.csv.gz")
    ev = ev[ev.stage != "pretrain"]
    phase, interval, k = cfg["phase_steps"], cfg["eval_interval"], cfg["evaluation"]["final_last_checkpoints"]
    last = lambda end: {end - i * interval for i in range(k)}
    ev = ev[np.where(ev.stage == "extension", ev.step.isin(last(2 * phase)), ev.step.isin(last(phase)))]
    runs = ev.groupby(["architecture", "condition", "train_seed"]).agg(
        L=("episode_length", "mean"), food=("beneficial", "mean"), n=("episode_length", "size")).reset_index()
    assert (runs.n == k * cfg["evaluation"]["episodes"]).all()
    return runs


def main():
    cfg = load_config()
    seeds = sorted(sum(cfg["train_seeds"].values(), []))
    st = cfg["statistics"]
    test = PairedTest(len(seeds), st["bootstrap_samples"], st["bootstrap_seed"])
    runs = final_runs(cfg)
    per = {(a, c): g.set_index("train_seed").loc[seeds] for (a, c), g in runs.groupby(["architecture", "condition"])}
    archs = list(cfg["actor_architectures"])
    out = {"conditions": {}, "vs_K": {}, "vs_G": {}, "vs_N": {}, "holm_family_size": {}}
    for (a, c), g in sorted(per.items()):
        out["conditions"][f"{a}/{c}"] = {"L_mean": g.L.mean(), "L_sd": g.L.std(ddof=1),
                                         "food_mean": g.food.mean(), "food_sd": g.food.std(ddof=1)}
        if c not in ("baseline", "no_shift"):
            out["vs_K"][f"{a}/{c}"] = test(g.L.values - per[a, "baseline"].L.values)
        if c not in ("scratch_B", "no_shift"):
            out["vs_N"][f"{a}/{c}"] = test(g.L.values - per[a, "scratch_B"].L.values)
        if c in ("optimizer_reset", "hidden_optimizer_reset", "critic_optimizer_reset", "critic_head_optimizer_reset",
                 "head_reset_extended"):
            out["vs_G"][f"{a}/{c}"] = test(g.L.values - per[a, "head_optimizer_reset"].L.values)
    # One Holm family per architecture: every network is a separate experiment with its own n = 10 seeds.
    for a in archs:
        family = {k: v for k, v in out["vs_K"].items() if k.split("/")[0] == a and k.split("/")[1] in st["holm_family"]}
        holm(family)
        out["holm_family_size"][a] = len(family)

    base = pd.read_csv(REPO / "data" / "baselines.csv")
    out["simple_baselines"] = {f"{w}/{s}": {"length": g.episode_length.mean(), "food": g.beneficial.mean()}
                               for (w, s), g in base.groupby(["world", "strategy"])}

    path = REPO / "data" / "rollout_episodes.csv.gz"
    if path.exists():
        ep = pd.read_csv(path)
        beh = pd.read_csv(REPO / "data" / "behaviour.csv")
        out["rollouts"] = {}
        for (a, m), g in ep.groupby(["architecture", "model"]):
            row = {}
            for mode, short in [("deterministic", "det"), ("stochastic", "sto")]:
                h = g[g["mode"] == mode]
                by_seed = h.groupby("train_seed")
                row[f"{short}_L"] = by_seed.length.mean().mean()
                row[f"{short}_runs"] = by_seed.length.mean().loc[seeds].tolist()
                row[f"{short}_food"] = by_seed.beneficial.mean().mean()
                row[f"{short}_deaths"] = h.died.mean()
                row[f"{short}_stuck_of_all"] = h.stuck.mean()
                row[f"{short}_blocked"] = h.blocked_fraction.mean()
                row[f"{short}_swing"] = h.swing_fraction.mean()
                d = beh[(beh.architecture == a) & (beh.model == m) & (beh.measurement == f"visited_{mode}")]
                row[f"dir_{short}"] = d[["new_only", "old_only", "both", "neither"]].mean().to_dict()
            act = beh[(beh.architecture == a) & (beh.model == m) & (beh.measurement == "activations")]
            if len(act):
                row["act"] = act[["dormant_max_layer", "saturated_max_layer", "srank_last"]].mean().to_dict()
            for meas in ["probe_final", "probe_after_reset"]:
                pr = beh[(beh.architecture == a) & (beh.model == m) & (beh.measurement == meas)]
                if len(pr):
                    row[meas] = pr[["entropy_fraction_of_max", "new_only", "old_only"]].mean().to_dict()
            out["rollouts"][f"{a}/{m}"] = row
        out["rollout_pairs"] = {}
        for a in archs:
            for x, y in [("head_optimizer_reset", "scratch_B"), ("head_optimizer_reset", "baseline"),
                         ("head_reset_extended", "scratch_B")]:
                if f"{a}/{x}" in out["rollouts"]:
                    for short in ["det", "sto"]:
                        out["rollout_pairs"][f"{a}/{x}-{y}/{short}"] = test(
                            np.array(out["rollouts"][f"{a}/{x}"][f"{short}_runs"]) - np.array(out["rollouts"][f"{a}/{y}"][f"{short}_runs"]))
    (REPO / "results").mkdir(exist_ok=True)
    write_json(REPO / "results" / "summary.json", json.loads(json.dumps(out, default=float)))

    # Table 1 of the paper: L ± SD (food) per condition and architecture; * = significant vs К after Holm.
    order = ["baseline", "entropy_spike", "gradual_shift", "high_energy", "optimizer_reset", "critic_optimizer_reset",
             "head_optimizer_reset", "critic_head_optimizer_reset", "hidden_optimizer_reset", "scratch_B"]
    lines = ["| Условие | " + " | ".join(ARCH_NAMES[a] for a in archs) + " |", "|---" * (len(archs) + 1) + "|"]
    for c in order:
        cells = []
        for a in archs:
            s = out["conditions"].get(f"{a}/{c}")
            if s is None:
                cells.append("—")
                continue
            star = "*" if out["vs_K"].get(f"{a}/{c}", {}).get("p_holm", 1) < st["alpha"] else ""
            cells.append(f"{s['L_mean']:.0f} ± {s['L_sd']:.0f}{star} ({s['food_mean']:.1f})")
        lines.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    (REPO / "results" / "table1.md").write_text(
        "Итоговая длина эпизода L в мире B, шаги: среднее ± SD по 10 запускам, в скобках — съеденный полезный ресурс "
        "за эпизод; * — отличие от К значимо (p < 0,05, точный перестановочный тест с поправкой Холма).\n\n" + table,
        encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
