"""Exact speedups for the selection-aware permutation test.

Three changes, none of which alters a statistic or a fixed-B p-value:

1. Lazy analysis. Selection reads only the inner-validation scores, so every
   candidate needs its inner fit, but only the selected candidate (and,
   optionally, the fixed pipeline) needs the outer refit. With K = 8 candidates
   and 5 outer folds this is 40 + 5 fits instead of 80, or 40 + 10 when the
   fixed pipeline is also refitted and was not the one selected.
2. Parallel permutations. Permutation i depends only on i, so indices run in
   separate processes and results are collected in index order.
3. Besag-Clifford sequential stopping, as an option. It changes B, so it is
   off unless requested, and its p-value is reported under its own name.

There are two options, both off by default: a fixed-pipeline-only mode
(`select=False`, 5 outer fits per permutation, T_fixed identical to the full
run's T_fixed for the same index) and label permutation within blocks
(`permute_within_blocks`, e.g. NHANES survey strata). With `typeI=True` each
permutation also applies the window-naive, window-bootstrap and participant-
bootstrap procedures to the fixed pipeline's out-of-fold scores, so the share of
permutations they reject is their Type-I error on real features. These columns
are additive: the statistics and seeds are unchanged.

The engine does not import `scripts/`. The analysis pieces (candidate builder,
aggregation rule, permutation, fold construction, precomputation) are passed
in, so the caller can hand over the original functions instead of copies.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..utils.seeds import derive_seed

CHECKPOINT_COLUMNS = ["index", "T_selection_aware", "T_fixed", "selected"]
TYPEI_ALPHA = 0.05
TYPEI_PARTICIPANT_BOOT = 1000
TYPEI_WINDOW_BOOT = 400
TYPEI_COLUMNS = ["window_auroc", "window_naive_p", "window_naive_p_two_sided",
                 "window_naive_reject", "participant_boot_lo", "participant_boot_hi",
                 "participant_boot_reject", "window_boot_lo", "window_boot_hi",
                 "window_boot_reject"]


def lazy_analysis(y: np.ndarray, pre: dict, candidates: Sequence[dict],
                  build: Callable, aggregate: Callable, fixed_name: str | None,
                  seed: int = 0, with_fixed: bool = True, select: bool = True,
                  return_oof: bool = False) -> dict:
    """Selection-aware statistic with outer refits only where they are read.

    Mirrors `analysis()` in scripts/selection_aware_permutation.py rule for rule:
    the same fold skip rule, inner fits only when the validation fold has both
    classes, fold-ordered inner score lists, selection by the highest mean inner
    score with ties broken by name, and the same outer minimum of 10 predicted
    windows. `per_candidate_outer` holds only the candidates that were refitted.

    `select=False` skips the inner fits and the selection, and refits only the
    fixed pipeline. T_fixed does not read the inner fits, so it is the same
    number as with `select=True`; T_selection_aware is NaN.
    `return_oof=True` adds the out-of-fold window scores of the refitted
    candidates under "oof"; nothing else changes.
    """
    groups = pre["groups"]
    branches = pre["branches"]
    by_name = {c["name"]: c for c in candidates}
    inner_scores: dict[str, list[float]] = {c["name"]: [] for c in candidates}
    valid_folds: list[dict] = []
    n_inner = 0

    for fm in pre["masks"]:
        tr, va = fm["train"], fm["validation"]
        fit = tr | va
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[fit])) < 2:
            continue
        valid_folds.append(fm)
        if not select or not (va.sum() and len(np.unique(y[va])) > 1):
            continue
        for spec in candidates:
            X = branches[spec["branch"]]
            m = build(spec, seed)
            m.fit(X[tr], y[tr])
            n_inner += 1
            pv = m.predict_proba(X[va])[:, 1]
            pa, ya, _ = aggregate(pv, groups[va], y[va])
            if len(np.unique(ya)) > 1:
                inner_scores[spec["name"]].append(float(roc_auc_score(ya, pa)))

    inner_mean = {k: float(np.mean(v)) for k, v in inner_scores.items() if v}
    selected = max(inner_mean, key=lambda k: (inner_mean[k], k)) if inner_mean else None

    targets: list[str] = []
    if selected is not None:
        targets.append(selected)
    use_fixed = with_fixed and fixed_name is not None and fixed_name in by_name
    if use_fixed and fixed_name is not None and fixed_name not in targets:
        targets.append(fixed_name)

    out: dict[str, float] = {}
    n_outer: dict[str, int] = {}
    oofs: dict[str, np.ndarray] = {}
    for name in targets:
        spec = by_name[name]
        X = branches[spec["branch"]]
        oof = np.full(len(y), np.nan)
        n_outer[name] = 0
        for fm in valid_folds:
            fit, te = fm["train"] | fm["validation"], fm["test"]
            m = build(spec, seed)
            m.fit(X[fit], y[fit])
            n_outer[name] += 1
            oof[te] = m.predict_proba(X[te])[:, 1]
        oofs[name] = oof
        ok = ~np.isnan(oof)
        if ok.sum() < 10:
            continue
        pa, ya, _ = aggregate(oof[ok], groups[ok], y[ok])
        if len(np.unique(ya)) > 1:
            out[name] = float(roc_auc_score(ya, pa))

    n_sel = n_outer.get(selected, 0) if selected is not None else 0
    n_fix = (n_outer.get(fixed_name, 0)
             if (use_fixed and fixed_name is not None and fixed_name != selected) else 0)
    res: dict[str, Any] = {
        "per_candidate_outer": out,
        "inner_mean": inner_mean,
        "selected_by_inner": selected,
        "T_selection_aware": out.get(selected, float("nan")) if selected else float("nan"),
        "T_fixed": (out.get(fixed_name, float("nan"))
                    if use_fixed and fixed_name is not None else float("nan")),
        "n_fits": n_inner + n_sel + n_fix,
        "n_fits_inner": n_inner,
        "n_fits_outer_selected": n_sel,
        "n_fits_outer_fixed": n_fix,
    }
    if return_oof:
        res["oof"] = oofs
    return res


def stratified_boot_ci(y: np.ndarray, p: np.ndarray, rng: np.random.Generator,
                       n: int) -> tuple[float, float]:
    """95% percentile interval of AUROC, positives and negatives resampled apart.

    The rule of `_boot_ci` in scripts/simulation_study.py: fewer than 30 usable
    draws gives (nan, nan).
    """
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        if len(np.unique(y[idx])) > 1:
            draws.append(roc_auc_score(y[idx], p[idx]))
    if len(draws) < 30:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return (float(lo), float(hi))


def typeI_procedures(y: np.ndarray, oof: np.ndarray, groups: np.ndarray,
                     aggregate: Callable, seed_name: str,
                     n_participant_boot: int = TYPEI_PARTICIPANT_BOOT,
                     n_window_boot: int = TYPEI_WINDOW_BOOT) -> dict:
    """The three procedures of the simulation study, on one set of OOF scores.

    window naive           Mann-Whitney U of positive vs negative window scores,
                           every window treated as independent. `window_naive_p` is
                           one-sided (positives higher); the two-sided p is kept too.
                           Reject when the one-sided p <= 0.05.
    window bootstrap       stratified window resampling, n_window_boot draws; reject
                           when the 95% interval of window AUROC excludes 0.5.
    participant bootstrap  window scores averaged per participant (`aggregate`),
                           stratified participant resampling, n_participant_boot
                           draws; reject when the 95% interval excludes 0.5.
    Seeds come from `seed_name`, one fixed stream per procedure.
    """
    ok = ~np.isnan(oof)
    yw, pw = y[ok].astype(int), oof[ok]
    pos, neg = pw[yw == 1], pw[yw == 0]
    p1 = float(stats.mannwhitneyu(pos, neg, alternative="greater").pvalue)
    p2 = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    wlo, whi = stratified_boot_ci(yw, pw, np.random.default_rng(
        derive_seed(f"{seed_name}:window_boot")), n_window_boot)
    pa, ya, _ = aggregate(pw, groups[ok], yw)
    plo, phi = stratified_boot_ci(ya, pa, np.random.default_rng(
        derive_seed(f"{seed_name}:participant_boot")), n_participant_boot)

    def excl(lo: float, hi: float) -> int:
        return int(bool(np.isfinite(lo) and (lo > 0.5 or hi < 0.5)))
    return {"window_auroc": float(roc_auc_score(yw, pw)),
            "window_naive_p": p1, "window_naive_p_two_sided": p2,
            "window_naive_reject": int(p1 <= TYPEI_ALPHA),
            "participant_boot_lo": plo, "participant_boot_hi": phi,
            "participant_boot_reject": excl(plo, phi),
            "window_boot_lo": wlo, "window_boot_hi": whi,
            "window_boot_reject": excl(wlo, whi)}


def typeI_rates(extras: dict[int, dict]) -> dict:
    """Rejection shares over permutations, with Clopper-Pearson 95% intervals."""
    out: dict = {"n_permutations": len(extras), "alpha": TYPEI_ALPHA}
    for proc in ("window_naive", "window_boot", "participant_boot"):
        r = np.array([int(v[f"{proc}_reject"]) for v in extras.values()])
        if not len(r):
            continue
        k, n = int(r.sum()), len(r)
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        out[proc] = {"rejections": k, "rate": k / n,
                     "ci_95_clopper_pearson": [float(ci.low), float(ci.high)]}
    return out


@dataclass(frozen=True)
class PermContext:
    """Everything one permutation needs. Picklable, so it can go to a worker.

    The callables should be the original module-level functions; they pickle by
    reference, so workers import the same code rather than a copy.
    """
    feat: pd.DataFrame
    cols: list[str]
    y: np.ndarray
    n_folds: int
    fold_seed_name: str
    base_seed: int
    candidates: list[dict]
    fixed_name: str | None
    build: Callable
    aggregate: Callable
    permute: Callable
    make_folds: Callable
    precompute: Callable
    seed: int = 0
    with_fixed: bool = True
    threads: int | None = None
    select: bool = True
    typeI: bool = False


def _thread_limit(n: int | None):
    if n is None:
        return nullcontext()
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=n)


def permutation_statistic(ctx: PermContext, i: int) -> dict:
    """Permutation i, exactly as the loop body of selection_aware_permutation.main().

    Labels are permuted with seed base_seed + i, and the label-dependent folds are
    regenerated from the permuted labels with the same seed name.
    """
    t0 = time.perf_counter()
    with _thread_limit(ctx.threads):
        yb = ctx.permute(ctx.feat, ctx.y, ctx.base_seed + i)
        folds_b = ctx.make_folds(ctx.feat, yb, ctx.n_folds, ctx.fold_seed_name)
        pre_b = ctx.precompute(ctx.feat, ctx.cols, folds_b)
        r = lazy_analysis(yb, pre_b, ctx.candidates, ctx.build, ctx.aggregate,
                          ctx.fixed_name, seed=ctx.seed, with_fixed=ctx.with_fixed,
                          select=ctx.select, return_oof=ctx.typeI)
        extra = (typeI_procedures(yb, r["oof"][ctx.fixed_name], pre_b["groups"],
                                  ctx.aggregate, f"typeI:{ctx.base_seed + i}")
                 if ctx.typeI else {})
    return {"index": int(i),
            "T_selection_aware": r["T_selection_aware"],
            "T_fixed": r["T_fixed"],
            "selected": r["selected_by_inner"] or "none",
            "n_fits": r["n_fits"],
            "n_fits_inner": r["n_fits_inner"],
            "n_fits_outer_selected": r["n_fits_outer_selected"],
            "n_fits_outer_fixed": r["n_fits_outer_fixed"],
            "inner_mean": r["inner_mean"],
            "seconds": time.perf_counter() - t0,
            "pid": os.getpid(),
            **extra}


def iter_permutations(ctx: PermContext, indices: Iterable[int], n_jobs: int = 1,
                      chunk_size: int | None = None) -> Iterator[list[dict]]:
    """Yield results chunk by chunk, each chunk in index order.

    n_jobs == 1 runs in this process, one index per chunk. Otherwise one loky
    pool streams all indices in order (joblib's ordered generator), and a chunk
    is yielded every `chunk_size` results. Workers never wait for a chunk to
    finish, so a slow permutation does not idle the other cores. Results do not
    depend on n_jobs or chunk size, because each index carries its own seed.
    Closing the iterator early stops the pool; unfinished indices are dropped.
    """
    idx = [int(i) for i in indices]
    if n_jobs == 1:
        for i in idx:
            yield [permutation_statistic(ctx, i)]
        return
    from joblib import Parallel, delayed
    size = chunk_size or n_jobs
    stream = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(permutation_statistic)(ctx, i) for i in idx)
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


def permute_within_blocks(feat: pd.DataFrame, y: np.ndarray, seed: int,
                          block_of: Mapping) -> np.ndarray:
    """Participant-label permutation restricted to blocks (e.g. survey strata).

    Participants are ordered as in `permute()` of selection_aware_permutation.py
    (groupby study_id). One generator seeded with `seed` visits the blocks in
    sorted order and permutes the labels inside each, so every block keeps its
    label counts. Bind `block_of` (study_id -> block) with functools.partial to
    get the (feat, y, seed) signature the driver calls.
    """
    per = pd.DataFrame({"g": feat["study_id"], "y": y}).groupby("g")["y"].first()
    ids = per.index.to_numpy()
    labels = per.to_numpy()
    blocks = pd.Series(ids).map(block_of)
    if blocks.isna().any():
        missing = ids[blocks.isna().to_numpy()][:5].tolist()
        raise ValueError(f"{int(blocks.isna().sum())} participants have no block, "
                         f"e.g. {missing}")
    b = blocks.to_numpy()
    out = labels.copy()
    g = np.random.default_rng(seed)
    for blk in np.unique(b):
        where = np.flatnonzero(b == blk)
        out[where] = labels[where][g.permutation(len(where))]
    mapping = dict(zip(ids, out, strict=True))
    return feat["study_id"].map(mapping).to_numpy().astype(int)


def besag_clifford(null_stats: Iterable[float], observed: float, h: int = 10,
                   B_max: int | None = None, complete: bool = True) -> dict:
    """Sequential Monte Carlo p-value (Besag and Clifford, 1991, Biometrika 78:301-304).

    Consume permuted statistics in order and count those >= observed. Stop at the
    h-th such value; then p = h / L, with L the number consumed. If B_max values
    are consumed first, or the stream ends and `complete` says it is the whole
    stream, p = (1 + count) / (L + 1), which is the fixed-B plus-one p-value.
    If the stream ends early and is not complete, the decision is pending.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    n_ge = 0
    L = 0
    for t in null_stats:
        if B_max is not None and L >= B_max:
            break
        t = float(t)
        if np.isnan(t):
            raise ValueError("NaN in the permutation stream; drop invalid permutations first")
        L += 1
        if t >= observed:
            n_ge += 1
            if n_ge == h:
                return {"decided": True, "reached_h": True, "L": L, "n_ge": n_ge,
                        "p_value": h / L, "h": h, "B_max": B_max,
                        "rule": "h / L at the h-th exceedance"}
    reached = B_max is not None and L >= B_max
    if reached or complete:
        return {"decided": True, "reached_h": False, "L": L, "n_ge": n_ge,
                "p_value": (1 + n_ge) / (L + 1), "h": h, "B_max": B_max,
                "rule": "(1 + count) / (L + 1), fixed-B plus-one"}
    return {"decided": False, "reached_h": False, "L": L, "n_ge": n_ge,
            "p_value": None, "h": h, "B_max": B_max, "rule": "pending"}


def read_checkpoint(path: Path) -> dict[int, tuple[float, float, str]]:
    """Same schema as selection_aware_permutation.main(), read back bit-exactly.

    pandas' default float parser is not round-trip exact: it moves about 40% of
    AUROC-like values by one ulp. A resumed statistic that tied the observed one
    could then drop out of the `>=` count. `round_trip` returns the written floats
    exactly, so a resumed run gives the same p-value as an uninterrupted one.
    """
    if not Path(path).exists():
        return {}
    prev = pd.read_csv(path, float_precision="round_trip")
    return {int(r["index"]): (float(r["T_selection_aware"]), float(r["T_fixed"]),
                              str(r["selected"])) for _, r in prev.iterrows()}


def read_checkpoint_extras(path: Path) -> dict[int, dict]:
    """The Type-I columns of a checkpoint, per index; empty if it has none."""
    if not Path(path).exists():
        return {}
    prev = pd.read_csv(path, float_precision="round_trip")
    if not set(TYPEI_COLUMNS) <= set(prev.columns):
        return {}
    prev = prev.dropna(subset=["window_naive_p"])
    return {int(r["index"]): {c: r[c] for c in TYPEI_COLUMNS} for _, r in prev.iterrows()}


def write_checkpoint(path: Path, done: dict[int, tuple[float, float, str]],
                     extras: dict[int, dict] | None = None) -> None:
    """Same columns and float repr as the original checkpoint, rows sorted by index.

    With `extras`, the Type-I columns follow the original four; an index
    without extras gets empty cells there.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = CHECKPOINT_COLUMNS + (TYPEI_COLUMNS if extras else [])
    rows = [{"index": k, "T_selection_aware": v[0], "T_fixed": v[1], "selected": v[2],
             **((extras or {}).get(k, {}))}
            for k, v in sorted(done.items())]
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=cols).to_csv(tmp, index=False)
    os.replace(tmp, path)
