"""Bit-exact rerun of the first evaluation interval of every stage type against the published data.

Needs the checkpoint archive from the release (see README); skipped otherwise. Each case trains 10 240 steps of a
small network (about a minute on one core) and compares the SHA-256 of the policy weights with the hash recorded
in data/evaluations.csv.gz for the original run. Requires the pinned library versions of requirements.txt.
"""
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from energygrid_transfer import agents, training
from energygrid_transfer.utils import REPO, load_config, state_hash

CHECKPOINTS = Path(os.environ.get("ENERGYGRID_CHECKPOINTS", REPO / "checkpoints"))
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (CHECKPOINTS / "pretrain").is_dir(), reason="checkpoint archive not available")]

CASES = [("pretrain", "pretrain", "mlp16", 11), ("branch", "baseline", "mlp16", 11),
         ("branch", "entropy_spike", "mlp16", 11), ("branch", "gradual_shift", "mlp16", 101),
         ("branch", "high_energy", "mlp16", 11), ("branch", "optimizer_reset", "mlp16", 11),
         ("branch", "head_optimizer_reset", "mlp16", 22), ("branch", "hidden_optimizer_reset", "mlp16", 11),
         ("branch", "critic_optimizer_reset", "mlp16", 11), ("branch", "critic_head_optimizer_reset", "linear", 101),
         ("control", "scratch_B", "mlp16", 11), ("control", "no_shift", "mlp16", 101)]


@pytest.fixture(scope="module")
def recorded():
    columns = ["stage", "condition", "architecture", "train_seed", "step", "policy_sha256"]
    return pd.read_csv(REPO / "data" / "evaluations.csv.gz", usecols=columns).drop_duplicates()


def _install_parent(runs, cfg, arch, seed):
    d = runs / "pretrain" / f"{arch}_seed{seed}"
    d.mkdir(parents=True)
    shutil.copy2(CHECKPOINTS / "pretrain" / f"{arch}_seed{seed}.zip", d / "A_final.zip")
    shutil.copy2(CHECKPOINTS / "pretrain" / f"{arch}_seed{seed}_rng_after_A.json", d / "rng_after_A.json")
    model = agents.load_model(d / "A_final.zip", "A", cfg)
    (d / "metadata.json").write_text(json.dumps({
        "final_checkpoint": "A_final.zip", "final_policy_sha256": state_hash(model.policy.state_dict()),
        "final_optimizer_sha256": state_hash(model.policy.optimizer.state_dict())}), encoding="utf-8")


@pytest.mark.parametrize("stage,condition,arch,seed", CASES)
def test_first_interval_is_bit_identical(tmp_path, recorded, stage, condition, arch, seed):
    cfg = load_config()
    cfg["phase_steps"] = cfg["eval_interval"]
    if stage == "pretrain":
        out = training.pretrain(tmp_path, arch, seed, cfg)
    else:
        _install_parent(tmp_path, cfg, arch, seed)
        out = (training.train_branch if stage == "branch" else training.train_control)(tmp_path, arch, seed, condition, cfg)
    rows = [json.loads(line) for line in (out / "eval_records.jsonl").read_text(encoding="utf-8").splitlines()]
    new = {r["policy_sha256"] for r in rows if r["step"] == cfg["eval_interval"]}
    old = recorded[(recorded.stage == stage) & (recorded.condition == condition) & (recorded.architecture == arch)
                   & (recorded.train_seed == seed) & (recorded.step == cfg["eval_interval"])].policy_sha256
    assert len(new) == 1 and new == set(old)
