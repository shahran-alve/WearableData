"""Grouped resampling and paired tests.

Every interval in this project resamples PARTICIPANTS, not windows. Windows from
one person are not independent observations; bootstrapping them directly shrinks
intervals by roughly sqrt(windows-per-participant) and is the single most common
way digital-phenotyping papers overstate their evidence.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from ..utils.seeds import rng


def grouped_bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                         metric_fn: Callable[[np.ndarray, np.ndarray], float],
                         n_resamples: int = 2000, ci: float = 0.95,
                         seed_name: str = "bootstrap") -> dict[str, Any]:
    uniq = np.unique(groups)
    g = rng(seed_name)
    index_by_group = {u: np.flatnonzero(groups == u) for u in uniq}
    point = float(metric_fn(y, p))
    draws = []
    for _ in range(n_resamples):
        picked = g.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[u] for u in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            draws.append(float(metric_fn(y[idx], p[idx])))
        except ValueError:
            continue
    if len(draws) < 50:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_valid_resamples": len(draws), "n_groups": int(len(uniq)),
                "status": "UNSTABLE_TOO_FEW_VALID_RESAMPLES"}
    a = (1 - ci) / 2
    lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "ci_level": ci, "n_valid_resamples": len(draws),
            "n_groups": int(len(uniq)), "status": "OK"}


def mcnemar(y: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray,
            exact_threshold: int = 25) -> dict[str, Any]:
    b = int(np.sum((yhat_a == y) & (yhat_b != y)))
    c = int(np.sum((yhat_a != y) & (yhat_b == y)))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "test": "degenerate_no_discordant_pairs"}
    if n < exact_threshold:
        p = float(stats.binomtest(b, n, 0.5).pvalue)
        test = "exact_binomial"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = float(1 - stats.chi2.cdf(chi2, df=1))
        test = "chi2_continuity_corrected"
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "test": test,
            "caveat": "windows from one participant are correlated; treat as descriptive"}


def paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                       seed_name: str = "perm") -> dict[str, Any]:
    """Paired test on per-participant or per-fold metric values."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return {"status": "INSUFFICIENT_PAIRS", "n": int(len(d))}
    obs = float(d.mean())
    g = rng(seed_name)
    signs = g.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"status": "OK", "mean_difference": obs, "p_value": p, "n_pairs": int(len(d)),
            "cohens_dz": float(obs / (d.std(ddof=1) + 1e-12))}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"raw_p": p, "holm_p": adj, "reject_at_alpha": adj <= alpha}
    return {"alpha": alpha, "n_comparisons": m, "results": out}


def bootstrap_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
                         metric_fn: Callable[[np.ndarray, np.ndarray], float],
                         n_resamples: int = 2000, ci: float = 0.95,
                         seed_name: str = "boot_diff") -> dict[str, Any]:
    uniq = np.unique(groups)
    idx_by = {u: np.flatnonzero(groups == u) for u in uniq}
    g = rng(seed_name)
    draws = []
    for _ in range(n_resamples):
        picked = g.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            draws.append(float(metric_fn(y[idx], p_a[idx])) - float(metric_fn(y[idx], p_b[idx])))
        except ValueError:
            continue
    if len(draws) < 50:
        return {"status": "UNSTABLE", "n_valid_resamples": len(draws)}
    a = (1 - ci) / 2
    lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)])
    return {"status": "OK",
            "difference": float(metric_fn(y, p_a) - metric_fn(y, p_b)),
            "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_valid_resamples": len(draws)}
