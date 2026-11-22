"""Per-fold nested selection, analysis A'.

The global analysis (`lazy_analysis` in fastperm.py, `analysis()` in
scripts/selection_aware_permutation.py) picks one candidate for all outer folds by
its mean inner-validation participant AUROC. A' picks per outer fold: within fold k,
every candidate is fitted on the inner training participants and scored on that
fold's validation participants, the best one (ties broken by name, the global rule)
is refitted on train + validation, and it predicts fold k's test participants.
T_perfold is the participant AUROC of the pooled out-of-fold scores, with the same
aggregation and 10-window minimum as the global statistic.

Pooling scores from different models across folds mixes their score scales (a
class-weighted logistic regression centres near 0.5, an unweighted boosting model
near the prevalence), which the pooled AUROC reads as signal or noise. Each
statistic is therefore also returned as the mean of the per-fold test participant
AUROCs (`*_foldmean`, folds whose test set has one class are left out), from the
same predictions and no extra fits.

A valid fold without any inner AUROC (one-class validation) takes the global pick.
This never happens on NHANES, and with a single candidate it is that candidate.

Every rule shared with the global analysis (fold skip rule, inner fits, inner score
lists, tie rule, aggregation) is the same code path, so with K = 1 the per-fold,
global and fixed statistics are the same number. Outer refits are cached per
(candidate, fold): a fold whose pick is the fixed pipeline is fitted once.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .fastperm import PermContext, _thread_limit

PERFOLD_COLUMNS = ["index", "T_perfold", "T_fixed", "T_global", "selected_global",
                   "perfold_selected", "n_fallback_folds", "T_perfold_foldmean",
                   "T_fixed_foldmean", "T_global_foldmean"]
FLOAT_COLUMNS = ["T_perfold", "T_fixed", "T_global", "T_perfold_foldmean",
                 "T_fixed_foldmean", "T_global_foldmean"]
SKIP = "skip"
ALPHA = 0.05


def _best(scores: dict[str, float]) -> str:
    return max(scores, key=lambda k: (scores[k], k))


def perfold_analysis(y: np.ndarray, pre: dict, candidates: Sequence[dict],
                     build: Callable, aggregate: Callable, fixed_name: str | None,
                     seed: int = 0, with_fixed: bool = True,
                     with_global: bool = False) -> dict:
    """Per-fold selection statistic, plus the fixed and (optionally) global ones.

    `perfold_selected` has one entry per outer fold of `pre["masks"]`: the pick, or
    "skip" when the fold fails the skip rule. The global pick is computed from the
    same inner fits at no extra cost; its outer statistic T_global needs its own
    refits and is computed only with `with_global`.
    """
    groups = pre["groups"]
    branches = pre["branches"]
    by_name = {c["name"]: c for c in candidates}
    inner_scores: dict[str, list[float]] = {c["name"]: [] for c in candidates}
    fold_inner: list[dict[str, float] | None] = []
    valid: list[int] = []
    n_inner = 0

    for k, fm in enumerate(pre["masks"]):
        tr, va = fm["train"], fm["validation"]
        fit = tr | va
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[fit])) < 2:
            fold_inner.append(None)
            continue
        valid.append(k)
        scores: dict[str, float] = {}
        if va.sum() and len(np.unique(y[va])) > 1:
            for spec in candidates:
                X = branches[spec["branch"]]
                m = build(spec, seed)
                m.fit(X[tr], y[tr])
                n_inner += 1
                pv = m.predict_proba(X[va])[:, 1]
                pa, ya, _ = aggregate(pv, groups[va], y[va])
                if len(np.unique(ya)) > 1:
                    s = float(roc_auc_score(ya, pa))
                    inner_scores[spec["name"]].append(s)
                    scores[spec["name"]] = s
        fold_inner.append(scores)

    inner_mean = {k: float(np.mean(v)) for k, v in inner_scores.items() if v}
    selected_global = _best(inner_mean) if inner_mean else None

    picks: dict[int, str | None] = {}
    n_fallback = 0
    for k in valid:
        fk = fold_inner[k]
        if fk:
            picks[k] = _best(fk)
        else:
            picks[k] = selected_global
            n_fallback += 1

    cache: dict[tuple[str, int], np.ndarray] = {}
    n_outer = 0

    def outer(name: str, k: int) -> np.ndarray:
        nonlocal n_outer
        if (name, k) not in cache:
            fm = pre["masks"][k]
            spec = by_name[name]
            X = branches[spec["branch"]]
            m = build(spec, seed)
            fit = fm["train"] | fm["validation"]
            m.fit(X[fit], y[fit])
            n_outer += 1
            cache[(name, k)] = m.predict_proba(X[fm["test"]])[:, 1]
        return cache[(name, k)]

    nan = float("nan")

    def pooled(names: dict[int, str]) -> tuple[float, float, list[float]]:
        """Pooled participant AUROC, and the mean of the per-fold test AUROCs."""
        oof = np.full(len(y), np.nan)
        per_fold: list[float] = []
        for k, name in names.items():
            te = pre["masks"][k]["test"]
            oof[te] = outer(name, k)
            pa, ya, _ = aggregate(oof[te], groups[te], y[te])
            per_fold.append(float(roc_auc_score(ya, pa)) if len(np.unique(ya)) > 1 else nan)
        ok = ~np.isnan(oof)
        finite = [v for v in per_fold if not np.isnan(v)]
        fold_mean = float(np.mean(finite)) if finite else nan
        if ok.sum() < 10:
            return nan, fold_mean, per_fold
        pa, ya, _ = aggregate(oof[ok], groups[ok], y[ok])
        auc = float(roc_auc_score(ya, pa)) if len(np.unique(ya)) > 1 else nan
        return auc, fold_mean, per_fold

    none: tuple[float, float, list[float]] = (nan, nan, [])
    T_perfold, fm_perfold, fa_perfold = (pooled(cast(dict[int, str], picks))
                                         if selected_global is not None else none)
    n_outer_perfold = n_outer
    use_fixed = with_fixed and fixed_name is not None and fixed_name in by_name
    T_fixed, fm_fixed, fa_fixed = (pooled({k: cast(str, fixed_name) for k in valid})
                                if use_fixed else none)
    n_outer_fixed = n_outer - n_outer_perfold
    T_global, fm_global, fa_global = (pooled({k: selected_global for k in valid})
                                      if with_global and selected_global is not None else none)
    n_outer_global = n_outer - n_outer_perfold - n_outer_fixed
    return {
        "T_perfold": T_perfold,
        "T_fixed": T_fixed,
        "T_global": T_global,
        "T_perfold_foldmean": fm_perfold,
        "T_fixed_foldmean": fm_fixed,
        "T_global_foldmean": fm_global,
        "fold_test_auc": {"perfold": fa_perfold, "fixed": fa_fixed, "global": fa_global},
        "selected_global": selected_global,
        "perfold_selected": [(picks[k] or "none") if k in picks else SKIP
                             for k in range(len(pre["masks"]))],
        "fold_inner": fold_inner,
        "inner_mean": inner_mean,
        "n_fallback_folds": n_fallback,
        "n_fits": n_inner + n_outer,
        "n_fits_inner": n_inner,
        "n_fits_outer_perfold": n_outer_perfold,
        "n_fits_outer_fixed_extra": n_outer_fixed,
        "n_fits_outer_global_extra": n_outer_global,
    }


def perfold_permutation_statistic(ctx: PermContext, i: int,
                                  with_global: bool = False) -> dict:
    """Permutation i with the labeling and folds of fastperm.permutation_statistic."""
    t0 = time.perf_counter()
    with _thread_limit(ctx.threads):
        yb = ctx.permute(ctx.feat, ctx.y, ctx.base_seed + i)
        folds_b = ctx.make_folds(ctx.feat, yb, ctx.n_folds, ctx.fold_seed_name)
        pre_b = ctx.precompute(ctx.feat, ctx.cols, folds_b)
        r = perfold_analysis(yb, pre_b, ctx.candidates, ctx.build, ctx.aggregate,
                             ctx.fixed_name, seed=ctx.seed, with_fixed=ctx.with_fixed,
                             with_global=with_global)
    return {"index": int(i),
            "T_perfold": r["T_perfold"],
            "T_fixed": r["T_fixed"],
            "T_global": r["T_global"],
            "T_perfold_foldmean": r["T_perfold_foldmean"],
            "T_fixed_foldmean": r["T_fixed_foldmean"],
            "T_global_foldmean": r["T_global_foldmean"],
            "selected_global": r["selected_global"] or "none",
            "perfold_selected": "|".join(r["perfold_selected"]),
            "n_fallback_folds": r["n_fallback_folds"],
            "n_fits": r["n_fits"],
            "n_fits_inner": r["n_fits_inner"],
            "n_fits_outer_perfold": r["n_fits_outer_perfold"],
            "n_fits_outer_fixed_extra": r["n_fits_outer_fixed_extra"],
            "n_fits_outer_global_extra": r["n_fits_outer_global_extra"],
            "seconds": time.perf_counter() - t0,
            "pid": os.getpid()}


def iter_perfold(ctx: PermContext, indices: Iterable[int], n_jobs: int = 1,
                 chunk_size: int | None = None,
                 with_global: bool = False) -> Iterator[list[dict]]:
    """Chunks of results in index order; the scheme of fastperm.iter_permutations."""
    idx = [int(i) for i in indices]
    if n_jobs == 1:
        for i in idx:
            yield [perfold_permutation_statistic(ctx, i, with_global)]
        return
    from joblib import Parallel, delayed
    size = chunk_size or n_jobs
    stream = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(perfold_permutation_statistic)(ctx, i, with_global) for i in idx)
    buf: list[dict] = []
    try:
        for r in stream:
            buf.append(r)
            if len(buf) >= size:
                yield buf
                buf = []
        if buf:
            yield buf
    finally:
        stream.close()


def read_perfold_checkpoint(path: Path) -> dict[int, dict]:
    """Rows by index, floats read back bit-exactly (round_trip)."""
    if not Path(path).exists():
        return {}
    prev = pd.read_csv(path, float_precision="round_trip",
                       dtype={"selected_global": str, "perfold_selected": str})
    out = {}
    for _, r in prev.iterrows():
        out[int(r["index"])] = {**{c: float(r[c]) for c in FLOAT_COLUMNS},
                                "selected_global": str(r["selected_global"]),
                                "perfold_selected": str(r["perfold_selected"]),
                                "n_fallback_folds": int(r["n_fallback_folds"])}
    return out


def write_perfold_checkpoint(path: Path, done: dict[int, dict]) -> None:
    """PERFOLD_COLUMNS, rows sorted by index, atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"index": k, **{c: v[c] for c in PERFOLD_COLUMNS[1:]}}
            for k, v in sorted(done.items())]
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=PERFOLD_COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, path)


def recycled_typeI(stat: np.ndarray, ref: np.ndarray, alpha: float = ALPHA) -> dict:
    """Each permutation b as a pseudo-observed dataset, tested against the other B - 1.

    p(b) = (1 + #{j != b : ref_j >= stat_b}) / B, as in null recycling. `stat` and `ref` are
    paired by permutation index. With ref = stat the rate is about alpha by exchangeability.
    """
    stat = np.asarray(stat, dtype=float)
    ref = np.asarray(ref, dtype=float)
    B = len(stat)
    ge = (ref[None, :] >= stat[:, None])
    np.fill_diagonal(ge, False)
    p = (1 + ge.sum(axis=1)) / B
    k = int((p <= alpha).sum())
    ci = stats.binomtest(k, B).proportion_ci(confidence_level=0.95, method="exact")
    return {"B": B, "alpha": alpha, "rejections": k, "rate": k / B,
            "ci_95_clopper_pearson": [float(ci.low), float(ci.high)]}
