"""Provenance: how data/ and the checkpoint archive were exported from the original run directories.

The experiment was run in four batches (see docs/protocol.md): the first series (seeds 11-55, conditions
К, Э, П, Б250), the replication batch (seeds 101-505 for the same conditions plus the controls Н and А→А for all
seeds), the reset ablation (О, Г) and the review resets (С, Кр, Кр+Г, Г+). They were produced by earlier versions
of the code in training.py, which is bit-identical in behaviour (tests/test_reproducibility.py). This script maps
their directory layouts and record schemas onto the single layout used by this repository:

    data/evaluations.csv.gz    one row per evaluation episode during training
    data/baselines.csv         random and always-left strategies
    <out>/checkpoints/...      final model of every run (+ RNG snapshot after world A), published as a release asset

Usage:
    python tools/import_original_runs.py --first-series DIR --replication DIR --reset-ablation DIR \
        --review-resets DIR --review-extension DIR --checkpoints-out DIR
"""
from pathlib import Path
import argparse
import csv
import gzip
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energygrid_transfer.utils import REPO, load_config, sha256, write_json  # noqa: E402

FIELDS = ["stage", "condition", "architecture", "train_seed", "world", "step", "eval_seed", "episode_length",
          "terminated", "truncated", "beneficial", "harmful", "policy_sha256"]
TRAINING_CONDITIONS = ["baseline", "entropy_spike", "gradual_shift", "high_energy"]
ABLATION = ["optimizer_reset", "head_optimizer_reset"]
REVIEW = ["hidden_optimizer_reset", "critic_optimizer_reset", "critic_head_optimizer_reset"]


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def row(stage, condition, arch, seed, world, step, r):
    return {"stage": stage, "condition": condition, "architecture": arch, "train_seed": seed, "world": world,
            "step": step, "eval_seed": r["eval_seed"], "episode_length": r["episode_length"],
            "terminated": int(r["terminated"]), "truncated": int(r["truncated"]),
            "beneficial": r["beneficial_consumed"], "harmful": r["harmful_consumed"],
            "policy_sha256": r.get("policy_sha256", "")}


def main():
    ap = argparse.ArgumentParser()
    for name in ["first-series", "replication", "reset-ablation", "review-resets", "review-extension", "checkpoints-out"]:
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config()
    phase = cfg["phase_steps"]
    first, replication = cfg["train_seeds"]["first_series"], cfg["train_seeds"]["replication"]
    archs = list(cfg["actor_architectures"])
    ck = args.checkpoints_out / "checkpoints"
    rows, manifest = [], {}

    def keep(source, condition, arch, seed, suffix=".zip"):
        target = ck / condition / f"{arch}_seed{seed}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(target.relative_to(args.checkpoints_out)).replace("\\", "/")] = sha256(target)

    for arch in archs:
        for seed in first + replication:
            root = args.first_series if seed in first else args.replication
            pre = root / "pretraining" / f"{arch}_seed{seed}"
            meta = json.loads((pre / "metadata.json").read_text(encoding="utf-8"))
            keep(root / meta["final_checkpoint"], "pretrain", arch, seed)
            keep(pre / "rng_after_A.json", "pretrain", arch, seed, "_rng_after_A.json")
            rows += [row("pretrain", "pretrain", arch, seed, "A", r["checkpoint_step"], r)
                     for r in jsonl(pre / "eval_records.jsonl")]
            for condition in TRAINING_CONDITIONS:
                d = root / "branches" / f"{condition}_{arch}_seed{seed}"
                for r in jsonl(d / "eval_records.jsonl"):
                    # The first rows repeat the pretraining curve; B_zero is the A model evaluated in world B.
                    if r["phase"] == "B_zero":
                        rows.append(row("branch", condition, arch, seed, "B", 0, r))
                    elif r["phase"] == "B" and r.get("evaluation_world", "standard_100") == "standard_100":
                        rows.append(row("branch", condition, arch, seed, "B", r["checkpoint_step"] - phase, r))
                keep(d / f"B_{2 * phase:06d}.zip", condition, arch, seed)
            for condition in ["scratch_B", "no_shift"]:
                d = args.replication / "controls" / f"{condition}_{arch}_seed{seed}"
                world = "B" if condition == "scratch_B" else "A"
                rows += [row("control", condition, arch, seed, world, r["relative_step"], r) for r in jsonl(d / "eval_records.jsonl")]
                keep(d / f"C_{phase:06d}.zip", condition, arch, seed)
            for condition, base in [(c, args.reset_ablation) for c in ABLATION] + [(c, args.review_resets) for c in REVIEW]:
                if arch == "linear" and condition == "hidden_optimizer_reset":
                    continue
                d = base / "branches" / f"{condition}_{arch}_seed{seed}"
                rows += [row("branch", condition, arch, seed, "B", r["relative_step"], r) for r in jsonl(d / "eval_records.jsonl")]
                keep(d / f"B_{phase:06d}.zip", condition, arch, seed)
            if arch != "linear":
                d = args.review_extension / f"head_long_{arch}_seed{seed}"
                rows += [row("extension", "head_reset_extended", arch, seed, "B", r["relative_step"], r)
                         for r in jsonl(d / "eval_records.jsonl")]
                keep(d / f"B_{2 * phase:06d}.zip", "head_reset_extended", arch, seed)

    rows.sort(key=lambda r: (r["stage"], r["condition"], r["architecture"], r["train_seed"], r["step"], r["eval_seed"]))
    with gzip.open(REPO / "data" / "evaluations.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    base_rows = [json.loads(line) for line in (args.replication / "baselines.jsonl").read_text(encoding="utf-8").splitlines()]
    with (REPO / "data" / "baselines.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["strategy", "world", "eval_seed", "episode_length", "beneficial", "harmful"], lineterminator="\n")
        w.writeheader()
        for r in base_rows:
            w.writerow({"strategy": r["strategy"], "world": r["phase"], "eval_seed": r["eval_seed"],
                        "episode_length": r["episode_length"], "beneficial": r["beneficial_consumed"],
                        "harmful": r["harmful_consumed"]})
    write_json(args.checkpoints_out / "checkpoints" / "SHA256SUMS.json", manifest)
    print(len(rows), "evaluation episodes;", len(manifest), "checkpoint files")


if __name__ == "__main__":
    main()
