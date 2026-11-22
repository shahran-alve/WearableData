"""Participant-level bootstrap bias-corrected CV, BBC-CV.

Tsamardinos, Greasidou and Borboudakis (Machine Learning 107, 2018). The pooled
out-of-fold participant scores of all K candidates form a matrix S (participants x K).
Each bootstrap resample draws participants with replacement, stratified by label. The
candidate with the highest AUROC on the in-bag draw (duplicates kept) is the winner, and
its AUROC on the out-of-bag participants is recorded. The mean of those out-of-bag AUROCs
is the selection-corrected estimate, and their 2.5 and 97.5 percentiles are the interval.
No model is refitted: S comes from one set of outer fits on the observed labels.

S is built exactly as the outer half of `analysis()` in
scripts/selection_aware_permutation.py: the same fold skip rule, one fit per (candidate,
outer fold) on train + validation with the same seed, the same aggregation and the same
10-window minimum, so its per-candidate pooled AUROC is the `per_candidate_outer` value
of that function. Each fit is seeded on its own, so the fits run in any order and in
parallel without changing a score.

The engine does not import `scripts/`. The candidate builder and aggregation rule are
passed in, as in fastperm.py.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from ..utils.seeds import derive_seed
from .fastperm import _thread_limit

COST_ORDER = {"rf": 0, "hgb": 1, "logreg": 2}   # submit slow fits first; results unaffected


def outer_fold_scores(spec: dict, X: np.ndarray, y: np.ndarray, fit: np.ndarray,
                      te: np.ndarray, build: Callable, seed: int,
                      threads: int | None) -> tuple[np.ndarray, float, int]:
    """One outer fit: train on `fit`, return the test window probabilities.

    The same three lines as the outer refit in `analysis()`.
    """
    t0 = time.perf_counter()
    with _thread_limit(threads):
        m = build(spec, seed)
        m.fit(X[fit], y[fit])
        p = m.predict_proba(X[te])[:, 1]
    return p, time.perf_counter() - t0, os.getpid()


def oof_matrix(y: np.ndarray, pre: dict, candidates: Sequence[dict], build: Callable,
               aggregate: Callable, seed: int = 0, n_jobs: int = 1,
               threads: int | None = 1, max_nbytes: str | int | None = "1M") -> dict:
    """Out-of-fold scores of every candidate on the supplied labels and folds.

    Returns the window-level OOF vectors, the pooled participant AUROC per candidate
    (`per_candidate_outer`, the rule of `analysis()`), and the participant matrix S
    with one column per candidate that has a pooled AUROC. All columns must cover the
    same participants in the same order, otherwise a ValueError is raised.
    `max_nbytes` is joblib's memmapping threshold for the parallel path.
    """
    groups = pre["groups"]
    branches = pre["branches"]
    valid = [fm for fm in pre["masks"]
             if len(np.unique(y[fm["train"]])) >= 2
             and len(np.unique(y[fm["train"] | fm["validation"]])) >= 2]
    tasks = [(spec, k) for spec in candidates for k in range(len(valid))]
    tasks.sort(key=lambda t: COST_ORDER.get(t[0]["family"], 3))

    def args(spec: dict, k: int) -> tuple:
        fm = valid[k]
        return (spec, branches[spec["branch"]], y, fm["train"] | fm["validation"],
                fm["test"], build, seed, threads)

    t0 = time.perf_counter()
    if n_jobs == 1:
        results = [outer_fold_scores(*args(s, k)) for s, k in tasks]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=max_nbytes)(
            delayed(outer_fold_scores)(*args(s, k)) for s, k in tasks)
    wall = time.perf_counter() - t0

    oof = {c["name"]: np.full(len(y), np.nan) for c in candidates}
    fit_seconds = {c["name"]: 0.0 for c in candidates}
    pids = set()
    for (spec, k), (p, sec, pid) in zip(tasks, results, strict=True):
        oof[spec["name"]][valid[k]["test"]] = p
        fit_seconds[spec["name"]] += sec
        pids.add(pid)

    per_candidate: dict[str, float] = {}
    cols: list[np.ndarray] = []
    names: list[str] = []
    ids_ref = y_ref = None
    for spec in candidates:
        name = spec["name"]
        p = oof[name]
        ok = ~np.isnan(p)
        if ok.sum() < 10:
            continue
        pa, ya, ids = aggregate(p[ok], groups[ok], y[ok])
        if len(np.unique(ya)) < 2:
            continue
        per_candidate[name] = float(roc_auc_score(ya, pa))
        if ids_ref is None:
            ids_ref, y_ref = ids, ya
        elif not (np.array_equal(ids, ids_ref) and np.array_equal(ya, cast(np.ndarray, y_ref))):
            raise ValueError(f"{name} scores a different participant set; S is undefined")
        cols.append(np.asarray(pa, dtype=float))
        names.append(name)
    S = np.column_stack(cols) if cols else np.empty((0, 0))
    return {"S": S, "y": y_ref, "ids": ids_ref, "names": names,
            "per_candidate_outer": per_candidate, "oof": oof,
            "n_fits": len(tasks), "n_valid_folds": len(valid),
            "fit_seconds": fit_seconds, "wall_seconds": wall, "n_workers_seen": len(pids)}


def auroc_columns(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of every column of S, ties counted one half (the Mann-Whitney identity).

    Average ranks are multiples of one half, so the rank sums are exact in float64 and
    the only rounding is the final division.
    """
    y = np.asarray(y)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("both classes are needed for an AUROC")
    r = rankdata(np.asarray(S, dtype=float).reshape(len(y), -1), axis=0)
    return (r[pos].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positives and negatives drawn separately with replacement, class sizes kept."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]


def _winner(auc: np.ndarray, names: Sequence[str]) -> int:
    """Highest AUROC, ties broken by name: the rule of `analysis()`."""
    return max(range(len(names)), key=lambda j: (auc[j], names[j]))


def _summary(x: np.ndarray, level: float) -> dict:
    tail = round(50 * (1 - level), 10)          # 2.5 for level 0.95
    lo, hi = np.percentile(x, [tail, 100 - tail])
    return {"mean": float(x.mean()), "ci": [float(lo), float(hi)],
            "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan")}


def bbc_cv(S: np.ndarray, y: np.ndarray, names: Sequence[str], B: int = 2000,
           seed_name: str = "bbccv", single: str | None = None,
           level: float = 0.95) -> dict:
    """BBC-CV on a participant score matrix S (n x K) with labels y.

    One generator seeded from `seed_name` draws the B stratified resamples in order,
    so the resamples do not depend on K: a run on a subset of the columns sees the same
    resamples. A resample whose out-of-bag set lacks a class is skipped and counted.
    `single` names a candidate whose own out-of-bag AUROC is summarised on the same
    resamples (the single-candidate bootstrap).
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y).astype(int)
    names = list(names)
    if S.shape != (len(y), len(names)):
        raise ValueError(f"S has shape {S.shape}, expected {(len(y), len(names))}")
    if np.isnan(S).any():
        raise ValueError("S contains NaN")
    naive = {n: float(roc_auc_score(y, S[:, j])) for j, n in enumerate(names)}
    naive_winner = _winner(np.array([naive[n] for n in names]), names)
    j_single = names.index(single) if single is not None else None

    rng = np.random.default_rng(derive_seed(seed_name))
    oob_win, inbag_win, win_idx, oob_all = [], [], [], []
    skipped = 0
    for _ in range(B):
        idx = stratified_resample(y, rng)
        inbag = np.zeros(len(y), dtype=bool)
        inbag[idx] = True
        oob = ~inbag
        if len(np.unique(y[oob])) < 2:
            skipped += 1
            continue
        a_in = auroc_columns(y[idx], S[idx])
        a_oob = auroc_columns(y[oob], S[oob])
        w = _winner(a_in, names)
        win_idx.append(w)
        inbag_win.append(a_in[w])
        oob_win.append(a_oob[w])
        oob_all.append(a_oob)
    if not oob_win:
        raise ValueError("no resample had both classes out of bag")
    oob_win_a = np.array(oob_win)
    oob_all_a = np.vstack(oob_all)
    counts = np.bincount(np.array(win_idx), minlength=len(names))
    corrected = _summary(oob_win_a, level)
    naive_max = naive[names[naive_winner]]
    out = {
        "B": B, "n_valid_resamples": len(oob_win), "n_skipped_one_class_oob": skipped,
        "seed_name": seed_name, "level": level,
        "n_participants": int(len(y)), "n_positive": int(y.sum()), "K": len(names),
        "naive_per_candidate": naive,
        "naive_max": naive_max, "naive_winner": names[naive_winner],
        "bbc_corrected": corrected["mean"], "ci": corrected["ci"],
        "sd_oob_winner": corrected["sd"],
        "optimism": naive_max - corrected["mean"],
        "inbag_winner_mean": float(np.mean(inbag_win)),
        "winner_share": {n: float(c / len(oob_win)) for n, c in zip(names, counts, strict=False)},
        "oob_mean_per_candidate": {n: float(oob_all_a[:, j].mean())    # 1-D sums, as for
                                   for j, n in enumerate(names)},    # the winner mean
        "construction": ("participants resampled with replacement within each label; "
                         "winner = highest in-bag AUROC (duplicates kept, ties by name); "
                         "estimate = mean out-of-bag AUROC of the winner; percentile CI"),
    }
    if j_single is not None:
        s = _summary(oob_all_a[:, j_single], level)
        single_name = cast(str, single)
        out["single"] = {"candidate": single, "naive": naive[single_name],
                         "oob_mean": s["mean"], "ci": s["ci"], "sd": s["sd"],
                         "optimism": naive[single_name] - s["mean"]}
    return out
