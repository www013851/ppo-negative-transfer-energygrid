"""Run the training part of the experiment (or any subset of it).

Stages, in dependency order:
    pretrain   world A, every (architecture, seed)
    branches   world B from the A model: К, Э, П, Б250 and the resets О, Г, С, Кр, Кр+Г
    controls   Н (new model in world B) and А→А (continuation in world A)
    extend     Г+ (head-reset branch trained for another phase_steps)

Finished runs (status.json = complete) are skipped, so an interrupted run can simply be restarted.
The full matrix is 40 pretrainings + 350 branches + 80 controls + 30 extensions; on a 12-core desktop CPU it takes
roughly a day. `--pilot` runs a few hundred steps per stage for a quick technical check.

Examples:
    python scripts/run_experiment.py --stages all --workers 8
    python scripts/run_experiment.py --stages pretrain branches --archs mlp16 --seeds 11 --conditions head_optimizer_reset
    python scripts/run_experiment.py --pilot --runs runs_pilot
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energygrid_transfer import training  # noqa: E402
from energygrid_transfer.interventions import BRANCH_CONDITIONS, applies_to  # noqa: E402
from energygrid_transfer.utils import REPO, load_config  # noqa: E402

STAGES = ["pretrain", "branches", "controls", "extend"]
SUBDIR = {"pretrain": "pretrain", "branches": "branches", "controls": "controls", "extend": "extended"}


def done(runs, stage, name):
    status = Path(runs) / SUBDIR[stage] / name / "status.json"
    if not status.exists():
        return False
    if json.loads(status.read_text(encoding="utf-8"))["status"] != "complete":
        raise RuntimeError(f"{status.parent} is incomplete; delete it to rerun")
    return True


def jobs(stage, archs, seeds, conditions):
    if stage == "pretrain":
        return [(training.pretrain, a, s, None, f"{a}_seed{s}") for a in archs for s in seeds]
    if stage == "branches":
        return [(training.train_branch, a, s, c, f"{c}_{a}_seed{s}") for a in archs for s in seeds
                for c in conditions if c in BRANCH_CONDITIONS and applies_to(c, a)]
    if stage == "controls":
        return [(training.train_control, a, s, c, f"{c}_{a}_seed{s}") for a in archs for s in seeds
                for c in ["scratch_B", "no_shift"] if c in conditions]
    return [(training.extend, a, s, "head_optimizer_reset", f"head_reset_extended_{a}_seed{s}")
            for a in archs for s in seeds if a != "linear" and "head_reset_extended" in conditions]


def call(fn, runs, arch, seed, condition, cfg):
    return str(fn(runs, arch, seed, cfg) if condition is None else fn(runs, arch, seed, condition, cfg))


def main():
    cfg = load_config()
    all_seeds = sum(cfg["train_seeds"].values(), [])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["all"], choices=STAGES + ["all"])
    ap.add_argument("--archs", nargs="+", default=list(cfg["actor_architectures"]), choices=list(cfg["actor_architectures"]))
    ap.add_argument("--seeds", nargs="+", type=int, default=all_seeds)
    ap.add_argument("--conditions", nargs="+", default=list(cfg["conditions"]), choices=list(cfg["conditions"]))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--pilot", action="store_true", help="tiny budget: technical check of the whole pipeline")
    args = ap.parse_args()
    if args.pilot:
        cfg.update(phase_steps=2048, eval_interval=1024, intervention_steps=1024)
        cfg["evaluation"].update(seed_start=90000, episodes=3)
    stages = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    # Larger networks first, so the slowest jobs do not end up alone at the tail.
    archs = sorted(args.archs, key=lambda a: -sum(cfg["actor_architectures"][a]))
    for stage in stages:
        todo = [j for j in jobs(stage, archs, args.seeds, args.conditions) if not done(args.runs, stage, j[4])]
        print(f"[{stage}] {len(todo)} runs to do", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {pool.submit(call, fn, args.runs, a, s, c, cfg): name for fn, a, s, c, name in todo}
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                print(f"[{stage}] {i}/{len(todo)} {futures[f]}", flush=True)


if __name__ == "__main__":
    main()
