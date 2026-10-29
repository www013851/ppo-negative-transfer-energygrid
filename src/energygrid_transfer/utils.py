"""Small helpers: configuration, JSON files, hashes of model states and global RNG snapshots."""
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "experiment.json"


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Atomic write, so an interrupted run never leaves a half-written status file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    """Deterministic SHA-256 of a (nested) state dict; used to prove that two models are bit-identical."""
    h = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            h.update(str((str(value.dtype), tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(b"list")
            for item in value:
                update(item)
        else:
            h.update(repr(value).encode())

    update(state)
    return h.hexdigest()


def capture_rng():
    """Snapshot of the torch, Python and NumPy global generators (PPO draws minibatches from them)."""
    ns = np.random.get_state()
    return {"torch": torch.get_rng_state().tolist(), "python": random.getstate(),
            "numpy": [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])]}


def restore_rng(saved):
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value

    torch.set_rng_state(torch.tensor(saved["torch"], dtype=torch.uint8))
    random.setstate(tuples(saved["python"]))
    ns = saved["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))


def evaluation_seeds(cfg, block="evaluation"):
    ev = cfg[block]
    return list(range(ev["seed_start"], ev["seed_start"] + ev["episodes"]))
