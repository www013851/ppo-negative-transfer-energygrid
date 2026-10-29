"""Paired statistics over training seeds (n = 10 per architecture).

* 95 % percentile bootstrap interval of the mean paired difference (one fixed set of resampled seed indices
  for all comparisons, so intervals are comparable);
* exact two-sided sign-flip permutation test over all 2**n sign assignments; with n = 10 the smallest attainable
  p-value is 2/1024 = 0.00195;
* Holm step-down adjustment within one family of comparisons.
"""
import itertools

import numpy as np


class PairedTest:
    def __init__(self, n, bootstrap_samples, bootstrap_seed):
        self.n = n
        self.signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        self.boot = np.random.default_rng(bootstrap_seed).integers(0, n, (bootstrap_samples, n))

    def __call__(self, diff):
        d = np.asarray(diff, float)
        assert len(d) == self.n
        means = d[self.boot].mean(1)
        observed = abs(d.mean())
        p = float((np.abs((self.signs * d).mean(1)) >= observed - 1e-12).mean())
        return {"mean": float(d.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "p_perm": p,
                "positive": int((d > 0).sum()), "over10": int((d > 10).sum()), "worse10": int((d < -10).sum())}


def holm(tests):
    """Adds 'p_holm' to every test in the dict (in place)."""
    order = sorted(tests, key=lambda k: tests[k]["p_perm"])
    m = len(order)
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * tests[key]["p_perm"]))
        tests[key]["p_holm"] = running
    return tests
