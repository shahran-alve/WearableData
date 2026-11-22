#!/usr/bin/env python
"""Fast selection-aware permutation test.

Same test as scripts/selection_aware_permutation.py, computed faster. The
candidate family, the model builder, the aggregation rule, the permutation
and the fold construction are imported from the original scripts, not copied.

Subcommands
  run        the permutation test: lazy refits, parallel permutations, and an
             optional Besag-Clifford stop. Reads and writes the same checkpoint
             CSV as the original script, so either script can extend it.
  equality   original analysis() vs the lazy path on the same labels and folds.
  timing     wall clock and model fits per permutation for three configurations
             (original, lazy, lazy + n_jobs), run interleaved.
  checks     writes reports/bigdata/engine_checks.json from the two CSVs above and a
             pytest junit XML.
  observed   the observed statistics: full analysis(), the fixed and
             selected candidates' out-of-fold scores, window-naive Mann-Whitney p,
             participant bootstrap intervals, and the three Type-I procedures.
  compare    bitwise comparison of two checkpoints over an index range.

Options of `run`, all off by default so the default run is unchanged:
  --block-col C     permute participant labels within blocks of column C of
                    <workspace>/validation/participant_metadata.csv (e.g. sdmvstra);
                    permutation seeds come from "selperm:<ws>:within:<C>".
  --fixed-only      refit only the fixed pipeline (5 fits per permutation). T_fixed
                    for index i equals the full run's T_fixed for index i.
  --typeI           per permutation, also store the window-naive, window-bootstrap
                    and participant-bootstrap outcomes on the fixed pipeline's
                    out-of-fold scores (extra checkpoint columns).
  --deadline T      stop after the first checkpoint written at or after local time T.
  --observed-from   read the observed statistics from an `observed` JSON (checked
                    against the sha256 of features.csv) instead of recomputing them.
  --no-observed / --report-only / --perm-log   see --help.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import math
import os
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import participant_level_eval as ple
import selection_aware_permutation as sap

from wearabledata.evaluation.fastperm import (
    TYPEI_COLUMNS,
    PermContext,
    _thread_limit,
    besag_clifford,
    iter_permutations,
    lazy_analysis,
    permutation_statistic,
    permute_within_blocks,
    read_checkpoint,
    read_checkpoint_extras,
    stratified_boot_ci,
    typeI_procedures,
    typeI_rates,
    write_checkpoint,
)
from wearabledata.evaluation.statistics import grouped_bootstrap_ci
from wearabledata.utils.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]
BIGDATA = ROOT / "reports" / "bigdata"


# ------------------------------------------------------------------ shared
def load_workspace(ws: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Same loading as selection_aware_permutation.main()."""
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat.columns]
    return feat, cols, feat["label"].to_numpy()


def fold_seed_name(ws: Path) -> str:
    return f"cv:{ws.name}:primary"


def perm_base_seed(ws: Path) -> int:
    return derive_seed(f"selperm:{ws.name}")


def make_context(ws: Path, feat: pd.DataFrame, cols: list[str], y: np.ndarray,
                 n_folds: int, threads: int | None, block_of: dict | None = None,
                 block_col: str | None = None, select: bool = True,
                 typeI: bool = False) -> PermContext:
    """Bind the original functions and the original seed scheme.

    Without `block_of` this is the unrestricted context: `sap.permute` and the
    seed "selperm:<ws>". With it, labels are permuted within blocks and the
    seed name is "selperm:<ws>:within:<block_col>".
    """
    if block_of is None:
        permute, base = sap.permute, perm_base_seed(ws)
    else:
        permute = functools.partial(permute_within_blocks, block_of=block_of)
        base = derive_seed(f"selperm:{ws.name}:within:{block_col}")
    return PermContext(feat=feat, cols=cols, y=y, n_folds=n_folds,
                       fold_seed_name=fold_seed_name(ws), base_seed=base,
                       candidates=list(sap.CANDIDATES), fixed_name=sap.FIXED_PIPELINE,
                       build=sap.build, aggregate=ple.aggregate, permute=permute,
                       make_folds=sap.make_folds, precompute=sap.precompute,
                       threads=threads, select=select, typeI=typeI)


def load_blocks(ws: Path, block_col: str, block_file: str | None) -> dict:
    """study_id -> block from participant_metadata.csv (or --block-file)."""
    path = Path(block_file) if block_file else ws / "validation" / "participant_metadata.csv"
    meta = pd.read_csv(path)
    if block_col not in meta.columns:
        raise SystemExit(f"{path.name} has no column {block_col!r}")
    if meta["study_id"].duplicated().any():
        raise SystemExit(f"{path.name}: duplicate study_id")
    return dict(zip(meta["study_id"].astype(str), meta[block_col], strict=True))


def default_checkpoint(ws: Path, fixed_only: bool, block_col: str | None) -> Path:
    """Default checkpoint name; fixed-only and within-block runs get their own names."""
    stem = "fixed_null" if fixed_only else "selection_aware_null"
    if block_col:
        stem += f"_within_{block_col}"
    return ws / "results" / f"{stem}_checkpoint.csv"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_observed(path: Path, ws: Path) -> dict:
    """Observed statistics from an `observed` JSON computed on the same features.csv."""
    ob = json.loads(Path(path).read_text())
    sha = sha256_path(ws / "interim" / "features.csv")
    if ob["features_sha256"] != sha:
        raise SystemExit(f"{path} was computed on a different features.csv "
                         f"({ob['features_sha256'][:12]} vs {sha[:12]})")
    return ob["analysis_exact"]


class _CountingBuild:
    """Wraps sap.build. analysis() calls build once per fit, so calls == fits."""

    def __init__(self, build):
        self.build = build
        self.n = 0

    def __call__(self, spec, seed=0):
        self.n += 1
        return self.build(spec, seed)


def original_analysis_counted(feat, cols, y, folds, pre) -> tuple[dict, int]:
    """The original analysis(), unmodified, with its model fits counted."""
    counter = _CountingBuild(sap.build)
    original = sap.build
    sap.build = counter
    try:
        r = sap.analysis(feat, cols, y, folds, pre=pre)
    finally:
        sap.build = original
    return r, counter.n


def original_permutation(ws: Path, feat, cols, y, i: int, n_folds: int) -> dict:
    """The loop body of selection_aware_permutation.main() for index i."""
    t0 = time.perf_counter()
    yb = sap.permute(feat, y, perm_base_seed(ws) + i)
    folds_b = sap.make_folds(feat, yb, n_folds, fold_seed_name(ws))
    r, n = original_analysis_counted(feat, cols, yb, folds_b,
                                     sap.precompute(feat, cols, folds_b))
    return {"index": i, "T_selection_aware": r["T_selection_aware"], "T_fixed": r["T_fixed"],
            "selected": r["selected_by_inner"] or "none", "n_fits": n,
            "inner_mean": r["inner_mean"], "seconds": time.perf_counter() - t0}


def _absdiff(a: float, b: float) -> float:
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    return abs(a - b)


def _inner_diff(a: dict, b: dict) -> tuple[bool, float]:
    same_keys = set(a) == set(b)
    diffs = [_absdiff(a[k], b[k]) for k in set(a) & set(b)]
    return same_keys, (max(diffs) if diffs else 0.0)


def _system_times() -> tuple[float, float] | None:
    """(idle, total) CPU time of the whole machine, Windows only."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                 ctypes.byref(user)):
        return None

    def v(ft):
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime
    return float(v(idle)), float(v(kernel) + v(user))   # kernel time includes idle


def _busy(t0: tuple[float, float] | None) -> float:
    """Machine-wide CPU busy fraction since t0 (all processes, not just this one)."""
    t1 = _system_times()
    if t0 is None or t1 is None or t1[1] <= t0[1]:
        return float("nan")
    return round(1 - (t1[0] - t0[0]) / (t1[1] - t0[1]), 3)


def _is_synthetic(ws: Path) -> bool:
    return (ws / "interim" / "SYNTHETIC.json").exists()


# ------------------------------------------------------------------ run
def cmd_run(a: argparse.Namespace) -> int:
    ws = Path(a.workspace)
    B = a.permutations
    feat, cols, y = load_workspace(ws)
    fixed_only = bool(getattr(a, "fixed_only", False))
    block_col = getattr(a, "block_col", None)
    typeI = bool(getattr(a, "typeI", False))
    null_name = f"within_{block_col}" if block_col else "unrestricted"

    # Observed statistic from the original analysis(), so the descriptive block
    # (per-candidate outer AUROC, oracle maximum) matches the original report.
    threads = a.threads_per_task or None     # 0 = leave the thread pools alone
    obs_check: dict | None = None
    if getattr(a, "observed_from", None):
        obs = load_observed(Path(a.observed_from), ws)
        obs_check = {"source": str(a.observed_from)}
    elif getattr(a, "no_observed", False):
        obs = None
    elif fixed_only:
        folds = sap.make_folds(feat, y, a.folds, fold_seed_name(ws))
        pre = sap.precompute(feat, cols, folds)
        with _thread_limit(threads):
            obs = lazy_analysis(y, pre, sap.CANDIDATES, sap.build, ple.aggregate,
                                sap.FIXED_PIPELINE, select=False)
        obs_check = {"source": "lazy_analysis(select=False): fixed pipeline only"}
    else:
        folds = sap.make_folds(feat, y, a.folds, fold_seed_name(ws))
        pre = sap.precompute(feat, cols, folds)
        with _thread_limit(threads):
            obs = sap.analysis(feat, cols, y, folds, pre=pre)
            lazy_obs = lazy_analysis(y, pre, sap.CANDIDATES, sap.build, ple.aggregate,
                                     sap.FIXED_PIPELINE)
        obs_check = {
            "same_selected": lazy_obs["selected_by_inner"] == obs["selected_by_inner"],
            "abs_diff_T_selection_aware": _absdiff(lazy_obs["T_selection_aware"],
                                                   obs["T_selection_aware"]),
            "abs_diff_T_fixed": _absdiff(lazy_obs["T_fixed"], obs["T_fixed"]),
        }
        if not (obs_check["same_selected"]
                and obs_check["abs_diff_T_selection_aware"] <= 1e-12
                and obs_check["abs_diff_T_fixed"] <= 1e-12):
            raise RuntimeError(f"lazy analysis differs from analysis() on the observed "
                               f"labels: {obs_check}")

    block_of = load_blocks(ws, block_col, getattr(a, "block_file", None)) if block_col else None
    ctx = make_context(ws, feat, cols, y, a.folds, threads, block_of=block_of,
                       block_col=block_col, select=not fixed_only, typeI=typeI)
    block_report = None
    if block_of is not None:
        # Permutation 0 must keep every block's label counts.
        yb = ctx.permute(feat, y, ctx.base_seed)
        per = pd.DataFrame({"g": feat["study_id"], "y": y, "yb": yb}).groupby("g").first()
        per["block"] = per.index.map(block_of)
        cnt = per.groupby("block")[["y", "yb"]].sum()
        block_report = {"column": block_col, "n_blocks": int(len(cnt)),
                        "permutation_seed_name": f"selperm:{ws.name}:within:{block_col}",
                        "perm0_block_positive_counts_preserved":
                            bool((cnt["y"] == cnt["yb"]).all())}
        if not block_report["perm0_block_positive_counts_preserved"]:
            raise RuntimeError("within-block permutation changed a block's label count")
    ckpt = Path(a.checkpoint) if a.checkpoint else default_checkpoint(ws, fixed_only, block_col)
    done = read_checkpoint(ckpt)
    extras = read_checkpoint_extras(ckpt) if typeI else {}
    invalid: set[int] = set()
    k = 1 if (fixed_only or a.sequential_stat == "fixed") else 0
    if a.sequential and obs is None:
        raise SystemExit("--sequential needs the observed statistic")
    obs_stat = None if obs is None else (obs["T_selection_aware"] if k == 0 else obs["T_fixed"])
    deadline = (dt.datetime.fromisoformat(a.deadline)
                if getattr(a, "deadline", None) else None)
    perm_log = Path(a.perm_log) if getattr(a, "perm_log", None) else None

    def sequential_state() -> dict:
        vals, idxs, complete = [], [], True
        for i in range(B):
            if i in done:
                vals.append(done[i][k])
                idxs.append(i)
            elif i not in invalid:
                complete = False
                break
        res = besag_clifford(vals, obs_stat, h=a.h, B_max=B, complete=complete)
        res["stop_index"] = idxs[res["L"] - 1] if (res["decided"] and res["L"]) else None
        return res

    seq = sequential_state() if a.sequential else None
    todo = [i for i in range(B) if i not in done]
    t_start = time.perf_counter()
    n_new, fits_new, stopped_on_budget, stopped_on_deadline = 0, 0, False, False
    report_only = bool(getattr(a, "report_only", False))
    if deadline is not None and dt.datetime.now() >= deadline:
        stopped_on_deadline = True
    if not (seq and seq["decided"]) and not report_only and not stopped_on_deadline and todo:
        gen = iter_permutations(ctx, todo, n_jobs=a.n_jobs, chunk_size=a.chunk_size)
        try:
            for chunk in gen:
                log_rows = []
                for r in chunk:
                    n_new += 1
                    fits_new += r["n_fits"]
                    log_rows.append({"index": r["index"], "seconds": r["seconds"],
                                     "n_fits": r["n_fits"], "selected": r["selected"],
                                     "pid": r["pid"], "T_fixed": r["T_fixed"],
                                     "T_selection_aware": r["T_selection_aware"],
                                     "elapsed": time.perf_counter() - t_start,
                                     "logged_at": dt.datetime.now().isoformat(
                                         timespec="seconds")})
                    if np.isnan(r["T_fixed"]) or (not fixed_only
                                                  and np.isnan(r["T_selection_aware"])):
                        invalid.add(r["index"])
                        continue
                    done[r["index"]] = (r["T_selection_aware"], r["T_fixed"], r["selected"])
                    if typeI:
                        extras[r["index"]] = {c: r[c] for c in TYPEI_COLUMNS}
                write_checkpoint(ckpt, done, extras if typeI else None)
                if perm_log is not None:
                    perm_log.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(log_rows).to_csv(perm_log, mode="a", index=False,
                                                  header=not perm_log.exists())
                if a.sequential:
                    seq = sequential_state()
                    if seq["decided"]:
                        break
                if a.time_budget and time.perf_counter() - t_start > a.time_budget:
                    stopped_on_budget = True
                    break
                if deadline is not None and dt.datetime.now() >= deadline:
                    stopped_on_deadline = True
                    break
        finally:
            gen.close()
    wall = time.perf_counter() - t_start

    in_range = dict(sorted((i, v) for i, v in done.items() if i < B))
    resolved_all = all(i in done or i in invalid for i in range(B))
    sel_null = np.array([v[0] for v in in_range.values()])
    fix_null = np.array([v[1] for v in in_range.values()])

    default_path = not (fixed_only or block_col or typeI)
    report: dict = {
        "workspace": str(ws),
        "engine": ("lazy analysis: inner fits for every candidate, outer refit only for "
                   "the selected candidate and the fixed pipeline; the observed statistic "
                   "comes from the original analysis()"),
        "candidate_family": [c["name"] for c in sap.CANDIDATES],
        "fixed_pipeline": sap.FIXED_PIPELINE,
    }
    if not default_path:
        report["null"] = null_name
        report["mode"] = "fixed_only" if fixed_only else "selection_aware"
        report["permutation_seed_name"] = (f"selperm:{ws.name}:within:{block_col}"
                                           if block_col else f"selperm:{ws.name}")
        report["block"] = block_report
    if obs is not None:
        report["observed"] = {
            "selected_model": obs.get("selected_by_inner"),
            "T_selection_aware": round(obs["T_selection_aware"], 4),
            "T_fixed": round(obs["T_fixed"], 4),
            "T_oracle_max": round(obs.get("T_oracle_max", float("nan")), 4),
            "per_candidate_outer": {c: round(v, 4)
                                    for c, v in obs["per_candidate_outer"].items()},
        }
        if not default_path:
            report["observed_exact"] = {"T_selection_aware": obs["T_selection_aware"],
                                        "T_fixed": obs["T_fixed"]}
    report.update({
        "observed_lazy_check": obs_check,
        "B_requested": B,
        "B_completed": len(in_range),
        "all_indices_resolved": resolved_all,
        "stopped_on_time_budget": stopped_on_budget,
    })
    if not default_path:
        report["stopped_on_deadline"] = stopped_on_deadline
        report["deadline"] = getattr(a, "deadline", None)
        report["completed_indices_are_prefix"] = sorted(in_range) == list(range(len(in_range)))
    stopped_early = bool(a.sequential and seq and seq["reached_h"] and not resolved_all)
    if len(in_range) >= 2 and not stopped_early and obs is not None:
        report["fixed_pipeline_null"] = sap.mc_p(fix_null, obs["T_fixed"])
        if not fixed_only:
            report["selection_aware_null"] = sap.mc_p(sel_null, obs["T_selection_aware"])
            report["selection_optimism"] = {
                "null_mean_shift": round(float(sel_null.mean() - fix_null.mean()), 4),
                "null_p95_shift": round(float(np.percentile(sel_null, 95)
                                              - np.percentile(fix_null, 95)), 4),
                "null_max_shift": round(float(sel_null.max() - fix_null.max()), 4),
            }
    elif len(in_range) >= 2 and obs is None:
        report["null_summary_no_observed"] = {
            "fixed": {"mean": float(fix_null.mean()), "sd": float(fix_null.std(ddof=1)),
                      "max": float(fix_null.max())},
            **({} if fixed_only else {"selection_aware": {
                "mean": float(sel_null.mean()), "sd": float(sel_null.std(ddof=1)),
                "max": float(sel_null.max())}})}
    elif stopped_early:
        report["fixed_B_note"] = ("sequential rule stopped before B; fixed-B p-values are "
                                  "not reported because B was chosen by the data")
    report["sequential"] = ({"enabled": True, "method": "Besag and Clifford (1991)",
                             "statistic": a.sequential_stat, "observed": obs_stat, **seq}
                            if a.sequential else {"enabled": False})
    report["execution"] = {
        "n_jobs": a.n_jobs, "threads_per_task": threads,
        "chunk_size": a.chunk_size or (a.n_jobs if a.n_jobs > 1 else 1),
        "new_permutations": n_new, "wall_seconds": round(wall, 2),
        "fits_new_total": fits_new,
        "fits_per_permutation_mean": round(fits_new / n_new, 2) if n_new else None,
    }
    if typeI:
        report["real_typeI"] = {
            "procedures": ("fixed pipeline out-of-fold scores of each permutation; window "
                           "naive = one-sided Mann-Whitney p <= 0.05; window and participant "
                           "bootstrap = stratified percentile 95% interval of AUROC excludes "
                           "0.5 (400 and 1000 resamples)"),
            **typeI_rates({i: extras[i] for i in in_range if i in extras})}
    report["checkpoint"] = str(ckpt)
    report["data_source"] = (f"SYNTHETIC:{ws.name}" if _is_synthetic(ws) else f"REAL:{ws.name}")
    default_out = ("selection_aware_permutation_fast.json" if default_path
                   else ckpt.name.replace("_checkpoint.csv", "_fast.json"))
    out = Path(a.out) if a.out else ws / "results" / default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in report
                      if key in ("observed", "B_completed", "fixed_pipeline_null",
                                 "selection_aware_null", "sequential", "execution",
                                 "null", "mode", "real_typeI", "stopped_on_deadline")},
                     indent=2))
    return 0


# ------------------------------------------------------------------ observed
def cmd_observed(a: argparse.Namespace) -> int:
    """Observed-label statistics on one workspace."""
    ws = Path(a.workspace)
    feat, cols, y = load_workspace(ws)
    threads = a.threads_per_task or None
    folds = sap.make_folds(feat, y, a.folds, fold_seed_name(ws))
    pre = sap.precompute(feat, cols, folds)
    groups = pre["groups"]
    with _thread_limit(threads):
        t0 = time.perf_counter()
        obs, n_fits = original_analysis_counted(feat, cols, y, folds, pre)
        t_analysis = time.perf_counter() - t0
        t0 = time.perf_counter()
        lz = lazy_analysis(y, pre, sap.CANDIDATES, sap.build, ple.aggregate,
                           sap.FIXED_PIPELINE, return_oof=True)
        t_lazy = time.perf_counter() - t0
    lazy_check = {
        "same_selected": lz["selected_by_inner"] == obs["selected_by_inner"],
        "T_selection_aware_bitwise_equal": lz["T_selection_aware"] == obs["T_selection_aware"],
        "T_fixed_bitwise_equal": lz["T_fixed"] == obs["T_fixed"],
    }
    roles = {"fixed": sap.FIXED_PIPELINE, "selected": obs["selected_by_inner"]}
    per_role: dict = {}
    t0 = time.perf_counter()
    for role, name in roles.items():
        oof = lz["oof"][name]
        ok = ~np.isnan(oof)
        pa, ya, ids = ple.aggregate(oof[ok], groups[ok], y[ok])
        auc = float(roc_auc_score_(ya, pa))
        strat = stratified_boot_ci(ya, pa, np.random.default_rng(
            derive_seed(f"pb:a6:{ws.name}:{role}:stratified")), a.bootstrap)
        grouped = grouped_bootstrap_ci(ya, pa, ids, roc_auc_score_, n_resamples=a.bootstrap,
                                       seed_name="pb:primary:auc")
        tp = typeI_procedures(y, oof, groups, ple.aggregate,
                              f"typeI:observed:{ws.name}" + ("" if role == "fixed"
                                                             else f":{role}"))
        per_role[role] = {
            "candidate": name,
            "participant_auroc": auc,
            "equals_analysis_bitwise": auc == obs["per_candidate_outer"][name],
            "n_windows_scored": int(ok.sum()),
            "participant_bootstrap_stratified": {
                "ci_95": list(strat), "n_resamples": a.bootstrap,
                "construction": ("positives and negatives resampled separately, "
                                 "percentile interval")},
            "participant_bootstrap_grouped": {
                "ci_95": [grouped["ci_low"], grouped["ci_high"]],
                "n_resamples": a.bootstrap,
                "n_valid_resamples": grouped["n_valid_resamples"],
                "construction": ("grouped_bootstrap_ci as in participant_level_eval.py, "
                                 "seed name pb:primary:auc")},
            "window_naive": {"window_auroc": tp["window_auroc"],
                             "mann_whitney_p_one_sided": tp["window_naive_p"],
                             "mann_whitney_p_two_sided": tp["window_naive_p_two_sided"],
                             "n_positive_windows": int((y[ok] == 1).sum()),
                             "n_negative_windows": int((y[ok] == 0).sum())},
            "typeI_procedures_on_observed": tp,
        }
        if role == "selected" and name == sap.FIXED_PIPELINE:
            per_role[role]["note"] = "the selected candidate is the fixed pipeline"
    t_boot = time.perf_counter() - t0
    report = {
        "workspace": str(ws),
        "features_sha256": sha256_path(ws / "interim" / "features.csv"),
        "n_participants": int(pd.Series(groups).nunique()),
        "n_positive": int(pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].first().sum()),
        "n_windows": int(len(feat)),
        "n_feature_columns": len(cols),
        "folds": a.folds, "fold_seed_name": fold_seed_name(ws),
        "threads_per_task": threads,
        "analysis_exact": {"T_selection_aware": obs["T_selection_aware"],
                           "T_fixed": obs["T_fixed"],
                           "selected_by_inner": obs["selected_by_inner"],
                           "T_oracle_max": obs["T_oracle_max"],
                           "per_candidate_outer": obs["per_candidate_outer"],
                           "inner_mean": obs["inner_mean"]},
        "analysis_n_fits": n_fits,
        "lazy_check": lazy_check,
        "roles": per_role,
        "seconds": {"analysis": round(t_analysis, 1), "lazy": round(t_lazy, 1),
                    "bootstrap_and_procedures": round(t_boot, 1)},
        "data_source": f"REAL:{ws.name}" if not _is_synthetic(ws) else f"SYNTHETIC:{ws.name}",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n_participants", "n_positive", "analysis_exact",
                                               "lazy_check", "seconds")}, indent=2))
    return 0


def roc_auc_score_(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


# ------------------------------------------------------------------ compare
def _bits(v) -> str:
    if isinstance(v, (float, np.floating)):
        return float(v).hex()
    if isinstance(v, (int, np.integer)):
        return float(v).hex()
    return str(v)


def cmd_compare(a: argparse.Namespace) -> int:
    """Bitwise comparison of two checkpoints on indices first..last."""
    A = pd.read_csv(a.a, float_precision="round_trip").set_index("index")
    Bc = pd.read_csv(a.b, float_precision="round_trip").set_index("index")
    cols = [c for c in A.columns if c in Bc.columns]
    rows = []
    for i in range(a.first, a.last + 1):
        if i not in A.index or i not in Bc.index:
            rows.append({"label": a.label, "index": i, "column": "(row)", "value_a": "",
                         "value_b": "", "bitwise_equal": False,
                         "note": "missing in " + ("a" if i not in A.index else "b")})
            continue
        for c in cols:
            va, vb = A.at[i, c], Bc.at[i, c]
            rows.append({"label": a.label, "index": i, "column": c, "value_a": repr(va),
                         "value_b": repr(vb), "bitwise_equal": _bits(va) == _bits(vb),
                         "note": ""})
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, mode="a" if a.append and out.exists() else "w", index=False,
              header=not (a.append and out.exists()))
    summary = {"label": a.label, "n_values": int(len(df)),
               "n_bitwise_equal": int(df["bitwise_equal"].sum()),
               "all_equal": bool(df["bitwise_equal"].all()), "out": str(out)}
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_equal"] else 1


# ------------------------------------------------------------------ equality
def _equality_task(ws_str: str, index: int, n_folds: int, threads: int | None) -> dict:
    ws = Path(ws_str)
    feat, cols, y = load_workspace(ws)
    meta = json.loads((ws / "interim" / "SYNTHETIC.json").read_text()) \
        if _is_synthetic(ws) else {}
    with _thread_limit(threads):
        if index < 0:     # observed labels, as in selection_aware_permutation.main()
            yb = y
            folds_b = sap.make_folds(feat, yb, n_folds, fold_seed_name(ws))
            t0 = time.perf_counter()
            orig, n_orig = original_analysis_counted(feat, cols, yb, folds_b,
                                                     sap.precompute(feat, cols, folds_b))
            t_orig = time.perf_counter() - t0
            t0 = time.perf_counter()
            lz = lazy_analysis(yb, sap.precompute(feat, cols, folds_b), sap.CANDIDATES,
                               sap.build, ple.aggregate, sap.FIXED_PIPELINE)
            t_lazy = time.perf_counter() - t0
            lazy = {"T_selection_aware": lz["T_selection_aware"], "T_fixed": lz["T_fixed"],
                    "selected": lz["selected_by_inner"] or "none", "n_fits": lz["n_fits"],
                    "inner_mean": lz["inner_mean"]}
        else:             # permutation i: original loop body vs the driver's code path
            o = original_permutation(ws, feat, cols, y, index, n_folds)
            orig = {"T_selection_aware": o["T_selection_aware"], "T_fixed": o["T_fixed"],
                    "selected_by_inner": o["selected"], "inner_mean": o["inner_mean"]}
            n_orig, t_orig = o["n_fits"], o["seconds"]
            lazy = permutation_statistic(make_context(ws, feat, cols, y, n_folds, None), index)
            t_lazy = lazy["seconds"]
    same_keys, inner_diff = _inner_diff(orig["inner_mean"], lazy["inner_mean"])
    return {
        "workspace": ws.name,
        "effect": meta.get("effect"),
        "labels": "observed" if index < 0 else "permuted",
        "index": index,
        "selected_original": orig["selected_by_inner"] or "none",
        "selected_lazy": lazy["selected"],
        "selected_equal": (orig["selected_by_inner"] or "none") == lazy["selected"],
        "T_selection_aware_original": orig["T_selection_aware"],
        "T_selection_aware_lazy": lazy["T_selection_aware"],
        "abs_diff_T_selection_aware": _absdiff(orig["T_selection_aware"],
                                               lazy["T_selection_aware"]),
        "T_fixed_original": orig["T_fixed"],
        "T_fixed_lazy": lazy["T_fixed"],
        "abs_diff_T_fixed": _absdiff(orig["T_fixed"], lazy["T_fixed"]),
        "inner_keys_equal": same_keys,
        "max_abs_diff_inner_mean": inner_diff,
        "n_fits_original": n_orig,
        "n_fits_lazy": lazy["n_fits"],
        "seconds_original": round(t_orig, 3),
        "seconds_lazy": round(t_lazy, 3),
        "threads": threads,
    }


def cmd_equality(a: argparse.Namespace) -> int:
    from joblib import Parallel, delayed
    tasks = [(str(Path(w)), i) for w in a.workspaces for i in range(-1, a.perms)]
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(_equality_task)(w, i, a.folds, a.threads_per_task) for w, i in tasks)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    worst = float(df[["abs_diff_T_selection_aware", "abs_diff_T_fixed",
                      "max_abs_diff_inner_mean"]].to_numpy().max())
    print(json.dumps({"n_compared": len(df), "max_abs_diff": worst,
                      "n_selected_mismatch": int((~df["selected_equal"]).sum()),
                      "n_inner_keys_mismatch": int((~df["inner_keys_equal"]).sum()),
                      "wall_seconds": round(time.perf_counter() - t0, 1),
                      "out": str(out)}, indent=2))
    return 0


# ------------------------------------------------------------------ timing
def _warm() -> int:
    import participant_level_eval  # noqa: F401
    import selection_aware_permutation  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    return os.getpid()


def _probe_fit(feat, cols, y, ws: Path, n_folds: int, threads: int | None) -> dict:
    """One fixed-pipeline fit (outer fold 0 of the observed design) at a thread setting."""
    folds = sap.make_folds(feat, y, n_folds, fold_seed_name(ws))
    pre = sap.precompute(feat, cols, folds)
    fm = pre["masks"][0]
    spec = next(c for c in sap.CANDIDATES if c["name"] == sap.FIXED_PIPELINE)
    X = pre["branches"][spec["branch"]]
    fit = fm["train"] | fm["validation"]
    t0 = time.perf_counter()
    with _thread_limit(threads):
        p = sap.build(spec, 0).fit(X[fit], y[fit]).predict_proba(X[fm["test"]])[:, 1]
    return {"seconds": time.perf_counter() - t0, "pred": p}


def cmd_timing(a: argparse.Namespace) -> int:
    from joblib import Parallel, delayed
    ws = Path(a.workspace)
    feat, cols, y = load_workspace(ws)
    idx = list(range(a.perms))
    serial_threads = a.serial_threads or None
    ctx_serial = make_context(ws, feat, cols, y, a.folds, threads=serial_threads)
    ctx_par = make_context(ws, feat, cols, y, a.folds, threads=a.threads_per_task)
    blocks = [idx[s:s + a.block] for s in range(0, len(idx), a.block)]
    batches = [idx[s:s + a.n_jobs] for s in range(0, len(idx), a.n_jobs)]
    # Spread the parallel batches evenly between the serial blocks.
    after: dict[int, list[int]] = {}
    for b in range(len(batches)):
        pos = max(0, int(math.floor((b + 0.5) * len(blocks) / len(batches))) - 1)
        after.setdefault(pos, []).append(b)

    rows: list[dict] = []

    def warm_pool(tag: str, rnd: int) -> None:
        # Pool start-up is a one-off cost per run, not a per-permutation cost. loky
        # retires workers idle for 300 s, which the serial blocks can exceed, so
        # the pool is re-warmed before every parallel batch and the time logged.
        t0, s0 = time.perf_counter(), _system_times()
        pids = Parallel(n_jobs=a.n_jobs, backend="loky")(
            delayed(_warm)() for _ in range(a.n_jobs))
        rows.append({"config": "lazy_parallel_pool_warmup", "index": -1, "batch": tag,
                     "round": rnd, "task_seconds": np.nan,
                     "batch_wall_seconds": time.perf_counter() - t0, "batch_size": 0,
                     "n_jobs": a.n_jobs, "threads_per_task": a.threads_per_task,
                     "machine_cpu_busy": _busy(s0), "n_workers_seen": len(set(pids))})

    def row(cfg, r, batch, rnd, wall, size, n_jobs, threads, busy):
        return {"config": cfg, "index": r["index"], "batch": batch, "round": rnd,
                "task_seconds": r["seconds"], "batch_wall_seconds": wall,
                "batch_size": size, "n_jobs": n_jobs, "threads_per_task": threads,
                "machine_cpu_busy": busy, "n_fits": r["n_fits"],
                "n_fits_inner": r.get("n_fits_inner"),
                "n_fits_outer_selected": r.get("n_fits_outer_selected"),
                "n_fits_outer_fixed": r.get("n_fits_outer_fixed"),
                "T_selection_aware": r["T_selection_aware"], "T_fixed": r["T_fixed"],
                "selected": r["selected"], "pid": r.get("pid", os.getpid())}

    # Thread probe: the same fixed-pipeline fit at default threads and at one
    # thread, alternated. It documents how much default OpenMP threading costs
    # on this machine at this load, and that predictions do not change.
    for rep in range(a.probe_reps):
        order = (None, 1) if rep % 2 == 0 else (1, None)
        preds = {}
        for th in order:
            s0 = _system_times()
            pr = _probe_fit(feat, cols, y, ws, a.folds, th)
            preds[th] = pr["pred"]
            rows.append({"config": ("probe_fit_default_threads" if th is None
                                    else "probe_fit_1_thread"),
                         "index": -1, "batch": f"probe:{rep}", "round": -1,
                         "task_seconds": pr["seconds"], "batch_wall_seconds": pr["seconds"],
                         "batch_size": 1, "n_jobs": 1, "threads_per_task": th,
                         "machine_cpu_busy": _busy(s0), "n_fits": 1})
            print(f"probe rep={rep} threads={th} {pr['seconds']:.2f}s", flush=True)
        rows[-1]["probe_max_abs_pred_diff"] = float(np.max(np.abs(preds[None] - preds[1])))

    for j, block in enumerate(blocks):
        for i in block:
            order = ("original", "lazy") if i % 2 == 0 else ("lazy", "original")
            for cfg in order:
                s0 = _system_times()
                if cfg == "original":
                    with _thread_limit(serial_threads):
                        r = original_permutation(ws, feat, cols, y, i, a.folds)
                else:
                    r = permutation_statistic(ctx_serial, i)
                rows.append(row(cfg, r, f"{cfg}:{i}", j, r["seconds"], 1, 1, serial_threads,
                                _busy(s0)))
                print(f"{cfg:8s} i={i:3d} {r['seconds']:7.2f}s fits={r['n_fits']}", flush=True)
        for b in after.get(j, []):
            warm_pool(f"warmup:{b}", j)
            t1, s0 = time.perf_counter(), _system_times()
            res = [r for chunk in iter_permutations(ctx_par, batches[b], n_jobs=a.n_jobs,
                                                     chunk_size=len(batches[b]))
                   for r in chunk]
            wall = time.perf_counter() - t1
            busy = _busy(s0)
            rows += [row("lazy_parallel", r, f"parallel:{b}", j, wall, len(batches[b]),
                         a.n_jobs, a.threads_per_task, busy) for r in res]
            print(f"parallel batch {b} ({len(res)} perms) {wall:7.2f}s", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(json.dumps(timing_summary(df), indent=2))
    return 0


def timing_summary(df: pd.DataFrame) -> dict:
    runs = df[df["config"].isin(["original", "lazy", "lazy_parallel"])]
    sec: dict[str, float] = {}
    for cfg in ("original", "lazy"):
        sec[cfg] = float(runs.loc[runs.config == cfg, "task_seconds"].mean())
    par = runs[runs.config == "lazy_parallel"]
    walls = par.drop_duplicates("batch")["batch_wall_seconds"].sum()
    sec["lazy_parallel"] = float(walls / len(par)) if len(par) else float("nan")
    # Per-index agreement across the three configurations.
    keyed = runs.set_index(["index", "config"])
    diffs = []
    for stat in ("T_selection_aware", "T_fixed"):
        wide = keyed[stat].unstack()
        for c1, c2 in (("original", "lazy"), ("lazy", "lazy_parallel"),
                       ("original", "lazy_parallel")):
            diffs += [_absdiff(x, z) for x, z in zip(wide[c1], wide[c2], strict=True)]
    sel = keyed["selected"].unstack()
    sel_mismatch = int((sel.nunique(axis=1, dropna=False) > 1).sum())
    orig_fits = runs.loc[runs.config == "original", "n_fits"]
    lazy_rows = runs[runs.config.isin(["lazy", "lazy_parallel"])]
    return {
        "sec_per_perm": {k: round(v, 3) for k, v in sec.items()},
        "speedup_vs_original": {k: round(sec["original"] / v, 2) for k, v in sec.items()},
        "fits_per_perm": {
            "original": int(orig_fits.max()), "original_min": int(orig_fits.min()),
            "lazy": int(lazy_rows["n_fits"].max()),
            "lazy_mean": round(float(lazy_rows["n_fits"].mean()), 2),
            "lazy_selection_only": int((lazy_rows["n_fits_inner"]
                                        + lazy_rows["n_fits_outer_selected"]).max()),
        },
        "n_indices": int(runs["index"].nunique()),
        "machine_cpu_busy_mean": {
            cfg: round(float(df.loc[df.config == cfg, "machine_cpu_busy"].mean()), 3)
            for cfg in ("original", "lazy", "lazy_parallel")
            if "machine_cpu_busy" in df.columns},
        "thread_probe": _probe_summary(df),
        "max_abs_diff_across_configs": float(max(diffs)) if diffs else float("nan"),
        "n_selected_mismatch_across_configs": sel_mismatch,
        "n_workers_used_parallel": int(par["pid"].nunique()),
    }


def _probe_summary(df: pd.DataFrame) -> dict:
    d = df[df.config == "probe_fit_default_threads"]["task_seconds"]
    o = df[df.config == "probe_fit_1_thread"]["task_seconds"]
    if not len(d):
        return {}
    diff = df.get("probe_max_abs_pred_diff", pd.Series(dtype=float)).dropna()
    return {"reps": int(len(d)),
            "fixed_pipeline_fit_sec_default_threads": round(float(d.mean()), 3),
            "fixed_pipeline_fit_sec_1_thread": round(float(o.mean()), 3),
            "max_abs_pred_diff": float(diff.max()) if len(diff) else float("nan")}


# ------------------------------------------------------------------ checks
def cmd_checks(a: argparse.Namespace) -> int:
    eq = pd.read_csv(a.equality)
    tm = pd.read_csv(a.timing)
    ts = timing_summary(tm)
    suite = ET.parse(a.junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    n = int(suite.get("tests"))
    failed = int(suite.get("failures")) + int(suite.get("errors"))
    skipped = int(suite.get("skipped"))
    eq_max = float(eq[["abs_diff_T_selection_aware", "abs_diff_T_fixed",
                       "max_abs_diff_inner_mean"]].to_numpy().max())
    checks = {
        "max_abs_diff": eq_max,
        "n_compared": int(len(eq)),
        "fits_per_perm": {k: ts["fits_per_perm"][k]
                          for k in ("original", "lazy", "lazy_mean", "lazy_selection_only")},
        "sec_per_perm": ts["sec_per_perm"],
        "n_jobs": int(tm.loc[tm.config == "lazy_parallel", "n_jobs"].iloc[0]),
        "tests_passed": n - failed - skipped,
        "tests_failed": failed,
        "tests_skipped": skipped,
        "n_selected_mismatch": int((~eq["selected_equal"].astype(bool)).sum()),
        "timing_run_max_abs_diff_across_configs": ts["max_abs_diff_across_configs"],
        "timing_run_n_selected_mismatch": ts["n_selected_mismatch_across_configs"],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0


# ------------------------------------------------------------------ cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="the permutation test")
    r.add_argument("--workspace", default=str(ROOT / "reports" / "real_depresjon"))
    r.add_argument("--permutations", type=int, default=1000,
                   help="B for the fixed-B test; B_max under --sequential")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--threads-per-task", type=int, default=1,
                   help="BLAS/OpenMP threads per permutation (0 = no limit). Results do "
                        "not depend on it; default OpenMP threading is slow under load")
    r.add_argument("--chunk-size", type=int, default=None,
                   help="permutations per checkpoint write; default n_jobs")
    r.add_argument("--time-budget", type=float, default=None)
    r.add_argument("--sequential", action="store_true",
                   help="Besag-Clifford stop at h exceedances (off by default)")
    r.add_argument("--h", type=int, default=10)
    r.add_argument("--sequential-stat", choices=["selection_aware", "fixed"],
                   default="selection_aware")
    r.add_argument("--checkpoint", default=None,
                   help="default: <workspace>/results/selection_aware_null_checkpoint.csv "
                        "(variants: fixed_null[_within_<col>]_checkpoint.csv, "
                        "selection_aware_null_within_<col>_checkpoint.csv)")
    r.add_argument("--out", default=None)
    r.add_argument("--block-col", default=None,
                   help="permute labels within blocks of this participant_metadata column "
                        "(off by default)")
    r.add_argument("--block-file", default=None,
                   help="default: <workspace>/validation/participant_metadata.csv")
    r.add_argument("--fixed-only", action="store_true",
                   help="refit only the fixed pipeline (fixed-pipeline null)")
    r.add_argument("--typeI", action="store_true",
                   help="also store window-naive / window-bootstrap / participant-bootstrap "
                        "outcomes per permutation (extra checkpoint columns)")
    r.add_argument("--deadline", default=None,
                   help="local ISO time; stop after the first checkpoint at or after it")
    r.add_argument("--observed-from", default=None,
                   help="observed statistics from an `observed` JSON")
    r.add_argument("--no-observed", action="store_true",
                   help="skip the observed statistic (null only, no p-values)")
    r.add_argument("--report-only", action="store_true",
                   help="compute no new permutations; rewrite the report from the checkpoint")
    r.add_argument("--perm-log", default=None,
                   help="append one row per finished permutation (seconds, fits, pid)")
    r.set_defaults(func=cmd_run)

    o = sub.add_parser("observed", help="observed statistics")
    o.add_argument("--workspace", default=str(ROOT / "reports" / "real_nhanes"))
    o.add_argument("--folds", type=int, default=5)
    o.add_argument("--threads-per-task", type=int, default=1)
    o.add_argument("--bootstrap", type=int, default=10000)
    o.add_argument("--out", default=str(BIGDATA / "observed.json"))
    o.set_defaults(func=cmd_observed)

    q = sub.add_parser("compare", help="bitwise comparison of two checkpoints")
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--last", type=int, default=4)
    q.add_argument("--label", default="")
    q.add_argument("--append", action="store_true")
    q.add_argument("--out", default=str(BIGDATA / "remote_equality.csv"))
    q.set_defaults(func=cmd_compare)

    e = sub.add_parser("equality", help="original analysis() vs the lazy path")
    e.add_argument("--workspaces", nargs="+", default=[
        str(ROOT / "reports" / "synthetic_depresjon" / "synthetic_depresjon_null"),
        str(ROOT / "reports" / "synthetic_depresjon")])
    e.add_argument("--perms", type=int, default=10)
    e.add_argument("--folds", type=int, default=5)
    e.add_argument("--n-jobs", type=int, default=11)
    e.add_argument("--threads-per-task", type=int, default=1)
    e.add_argument("--out", default=str(BIGDATA / "lazy_equality.csv"))
    e.set_defaults(func=cmd_equality)

    t = sub.add_parser("timing", help="original vs lazy vs lazy + n_jobs")
    t.add_argument("--workspace", default=str(ROOT / "reports" / "synthetic_depresjon"))
    t.add_argument("--perms", type=int, default=20)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--n-jobs", type=int, default=10)
    t.add_argument("--threads-per-task", type=int, default=1)
    t.add_argument("--block", type=int, default=5, help="serial indices per block")
    t.add_argument("--serial-threads", type=int, default=1,
                   help="thread limit for the serial original and lazy runs (0 = none)")
    t.add_argument("--probe-reps", type=int, default=1,
                   help="alternating default-thread vs 1-thread fixed-pipeline fits")
    t.add_argument("--out", default=str(BIGDATA / "timing.csv"))
    t.set_defaults(func=cmd_timing)

    c = sub.add_parser("checks", help="write the engine checks JSON")
    c.add_argument("--equality", default=str(BIGDATA / "lazy_equality.csv"))
    c.add_argument("--timing", default=str(BIGDATA / "timing.csv"))
    c.add_argument("--junit", required=True, help="pytest --junitxml output")
    c.add_argument("--out", default=str(BIGDATA / "engine_checks.json"))
    c.set_defaults(func=cmd_checks)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")   # loky workers inherit this
    raise SystemExit(main())
