"""Figures 1 and 2 of the paper from data/ (run scripts/summarize.py first).

Fig. 1: final length L of every run by condition, one panel per architecture.
Fig. 2: (a) direction of the chosen action on visited states (mean over the three MLPs);
        (b) episode length with the greedy (filled) and the sampled (hollow) action, mean ± SD over seeds.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from energygrid_transfer.utils import REPO, load_config  # noqa: E402
from summarize import ARCH_NAMES, LABELS, final_runs  # noqa: E402

ORDER = ["baseline", "entropy_spike", "gradual_shift", "high_energy", "optimizer_reset", "critic_optimizer_reset",
         "head_optimizer_reset", "critic_head_optimizer_reset", "hidden_optimizer_reset", "scratch_B"]
COLORS = ["#303030", "#BD352B", "#2166AC", "#278046", "#A77714", "#C77C9B", "#744397", "#4E8C9A", "#D06B1F", "#777777"]


def figure1(cfg):
    runs = final_runs(cfg)
    first = set(cfg["train_seeds"]["first_series"])
    seeds = sorted(sum(cfg["train_seeds"].values(), []))
    jitter = dict(zip(seeds, np.linspace(-0.18, 0.18, len(seeds))))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharey=True, layout="constrained")
    for ax, arch in zip(axes.flat, cfg["actor_architectures"]):
        for x, (c, color) in enumerate(zip(ORDER, COLORS)):
            g = runs[(runs.architecture == arch) & (runs.condition == c)]
            for _, r in g.iterrows():
                hollow = r.train_seed in first
                ax.scatter(x + jitter[r.train_seed], r.L, marker="o" if hollow else "^", s=14, linewidths=0.7,
                           facecolors="none" if hollow else color, edgecolors=color, zorder=3)
            if len(g):
                ax.plot([x - 0.25, x + 0.25], [g.L.mean()] * 2, color="black", lw=1.3, zorder=4)
                ax.scatter([x], [g.L.median()], marker="D", s=9, color="black", zorder=5)
        ax.set_title(ARCH_NAMES[arch], fontsize=10)
        ax.set_xticks(range(len(ORDER)), [LABELS[c] for c in ORDER], fontsize=8)
        ax.set_ylim(0, 515)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("L, шаги")
    fig.savefig(REPO / "figures" / "fig1_final_length.png", dpi=200)
    plt.close(fig)


def figure2(cfg):
    beh = pd.read_csv(REPO / "data" / "behaviour.csv")
    summary = json.loads((REPO / "results" / "summary.json").read_text(encoding="utf-8"))["rollouts"]
    mlp = [a for a in cfg["actor_architectures"] if a != "linear"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    models = ["pretrain", "baseline", "optimizer_reset", "head_optimizer_reset", "scratch_B"]
    names = ["А", "К", "О", "Г", "Н"]
    cats = [("new_only", "только к новому", "#744397"), ("both", "к обоим", "#B9A6CC"),
            ("old_only", "только к прежнему", "#BD352B"), ("neither", "ни к одному", "#D9D9D9")]
    bottom = np.zeros(len(models))
    vis = beh[(beh.measurement == "visited_deterministic") & beh.architecture.isin(mlp)]
    for key, name, color in cats:
        v = np.array([vis[vis.model == m][key].mean() * 100 for m in models])
        a1.bar(range(len(models)), v, 0.62, bottom=bottom, color=color, edgecolor="white", lw=0.4, label=name)
        bottom += v
    a1.set_xticks(range(len(models)), names)
    a1.set_ylabel("Доля посещённых состояний, %")
    a1.set_ylim(0, 100)
    a1.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    a1.set_title("а", loc="left")
    groups = [("baseline", "К", "#222222"), ("head_optimizer_reset", "Г", "#744397"),
              ("head_reset_extended", "Г+", "#D06B1F"), ("scratch_B", "Н", "#999999")]
    w = 0.8 / len(groups)
    for i, (m, name, color) in enumerate(groups):
        for short, face in [("det", color), ("sto", "none")]:
            xs, mean, sd = [], [], []
            for j, arch in enumerate(cfg["actor_architectures"]):
                r = summary.get(f"{arch}/{m}")
                if r is None:
                    continue
                xs.append(j + (i - (len(groups) - 1) / 2) * w + (0.05 if short == "sto" else -0.05))
                mean.append(np.mean(r[f"{short}_runs"]))
                sd.append(np.std(r[f"{short}_runs"], ddof=1))
            a2.errorbar(xs, mean, yerr=sd, fmt="o", ms=4, capsize=2, lw=0.8, color=color, mfc=face,
                        label=name if short == "det" else None)
    a2.set_xticks(range(4), [ARCH_NAMES[a] for a in cfg["actor_architectures"]], fontsize=8)
    a2.set_ylabel("L, шаги")
    a2.set_ylim(0, 520)
    a2.grid(axis="y", color="#dddddd", lw=0.5)
    a2.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    a2.set_title("б", loc="left")
    fig.savefig(REPO / "figures" / "fig2_behaviour.png", dpi=200)
    plt.close(fig)


def main():
    cfg = load_config()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    (REPO / "figures").mkdir(exist_ok=True)
    figure1(cfg)
    figure2(cfg)
    print("figures written to", REPO / "figures")


if __name__ == "__main__":
    main()
