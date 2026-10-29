import numpy as np

from energygrid_transfer.stats import PairedTest, holm


def test_smallest_exact_p_value_with_ten_pairs():
    assert PairedTest(10, 2000, 0)(np.arange(1, 11))["p_perm"] == 2 / 1024


def test_null_difference_is_not_significant():
    assert PairedTest(10, 2000, 0)(np.zeros(10))["p_perm"] == 1.0


def test_holm_is_monotone_and_never_below_raw_p():
    tests = {k: {"p_perm": p} for k, p in zip("abcd", [0.002, 0.01, 0.04, 0.5])}
    holm(tests)
    adjusted = [tests[k]["p_holm"] for k in "abcd"]
    assert adjusted == sorted(adjusted)
    assert all(tests[k]["p_holm"] >= tests[k]["p_perm"] for k in tests)
    assert adjusted[0] == 4 * 0.002
