import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

from wearabledata.evaluation.statistics import (
    grouped_bootstrap_ci,
    holm,
    mcnemar,
    paired_permutation,
)


def test_grouped_bootstrap_resamples_participants_not_windows():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 10)
    y = np.repeat(rng.integers(0, 2, 20), 10)
    p = y * 0.6 + rng.normal(0, 0.25, len(y))
    grouped = grouped_bootstrap_ci(y, p, groups, roc_auc_score, n_resamples=300)
    window_level = grouped_bootstrap_ci(y, p, np.arange(len(y)), roc_auc_score,
                                        n_resamples=300)
    gw = grouped["ci_high"] - grouped["ci_low"]
    ww = window_level["ci_high"] - window_level["ci_low"]
    # The whole point: ignoring grouping produces a falsely narrow interval.
    assert gw > ww


def test_mcnemar_uses_exact_test_for_few_discordant_pairs():
    y = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 0, 0, 0, 1])
    b = np.array([1, 1, 0, 1, 1])
    r = mcnemar(y, a, b)
    assert r["test"] == "exact_binomial"
    assert 0.0 <= r["p_value"] <= 1.0


def test_holm_is_monotone_and_conservative():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04})["results"]
    assert out["a"]["holm_p"] <= out["b"]["holm_p"] <= out["c"]["holm_p"]
    assert out["c"]["holm_p"] >= 0.04


def test_paired_permutation_detects_a_real_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0.7, 0.05, 30)
    b = a - 0.1
    assert paired_permutation(a, b, n_perm=2000)["p_value"] < 0.01
