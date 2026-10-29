"""Collect the evaluation records of a (new) run directory into the tidy table format of data/.

    python scripts/collect_runs.py --runs runs --out data/evaluations_rerun.csv.gz

The output has exactly the columns of data/evaluations.csv.gz, so scripts/summarize.py works on a rerun after
replacing the file (or compare the two tables row by row: the policy hashes must coincide).
"""
from pathlib import Path
import argparse
import csv
import gzip
import json

FIELDS = ["stage", "condition", "architecture", "train_seed", "world", "step", "eval_seed", "episode_length",
          "terminated", "truncated", "beneficial", "harmful", "policy_sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for records in sorted(args.runs.glob("*/*/eval_records.jsonl")):
        for line in records.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.append({"stage": r["stage"], "condition": r["condition"], "architecture": r["architecture"],
                         "train_seed": r["train_seed"], "world": r["world"], "step": r["step"],
                         "eval_seed": r["eval_seed"], "episode_length": r["episode_length"],
                         "terminated": int(r["terminated"]), "truncated": int(r["truncated"]),
                         "beneficial": r["beneficial_consumed"], "harmful": r["harmful_consumed"],
                         "policy_sha256": r["policy_sha256"]})
    rows.sort(key=lambda r: (r["stage"], r["condition"], r["architecture"], r["train_seed"], r["step"], r["eval_seed"]))
    with gzip.open(args.out, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", args.out)


if __name__ == "__main__":
    main()
