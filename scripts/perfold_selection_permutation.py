#!/usr/bin/env python
"""Per-fold vs global selection.

Analysis A' selects the candidate inside each outer fold by that fold's inner-
validation participant AUROC (src/wearabledata/evaluation/perfold.py). The global
analysis selects once, by the mean over folds. The candidate family, model builder,
aggregation, permutation, folds and seeds are the originals, imported: permutation
i here is the same labeling as permutation i of the unrestricted global-selection run
(`fast_selection_permutation.py run`, seed "selperm:<ws>").

Subcommands
  observed   A' on the observed labels, with T_global and T_fixed from the same fits;
             optionally compared bitwise with the global-selection `observed` JSON.
  run        the per-fold null: parallel, checkpointed, optional deadline.
  report     per-fold null vs global null (global-selection checkpoint) vs fixed null
             (fixed checkpoint): p-values, null shift, recycled Type-I of the
             fixed null, pick frequencies, per-index consistency with the global run.
  checks     writes reports/bigdata/perfold_checks.json.

Option of `run`, off by default so the default run is unchanged:
  --block-col C     permute participant labels within blocks of column C of
                    <workspace>/validation/participant_metadata.csv (e.g. sdmvstra),
                    with the context of `fast_selection_permutation.py run --block-col C`
                    (seed "selperm:<ws>:within:<C>"), so permutation i is the labeling
                    of the global-selection within-block run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import fast_selection_permutation as fsp
import numpy as np
import pandas as pd
import participant_level_eval as ple
import selection_aware_permutation as sap
from scipy import stats

from wearabledata.evaluation.fastperm import _thread_limit, read_checkpoint
from wearabledata.evaluation.perfold import (
    FLOAT_COLUMNS,
    PERFOLD_COLUMNS,
    SKIP,
    iter_perfold,
    perfold_analysis,
    read_perfold_checkpoint,
    recycled_typeI,
    write_perfold_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
BIGDATA = ROOT / "reports" / "bigdata"


# ------------------------------------------------------------------ observed
def cmd_observed(a: argparse.Namespace) -> int:
    ws = Path(a.workspace)
    feat, cols, y = fsp.load_workspace(ws)
    threads = a.threads_per_task or None
    folds = sap.make_folds(feat, y, a.folds, fsp.fold_seed_name(ws))
    pre = sap.precompute(feat, cols, folds)
    t0 = time.perf_counter()
    with _thread_limit(threads):
        r = perfold_analysis(y, pre, sap.CANDIDATES, sap.build, ple.aggregate,
                             sap.FIXED_PIPELINE, with_global=True)
    secs = time.perf_counter() - t0
    sha = fsp.sha256_path(ws / "interim" / "features.csv")
    report = {
        "workspace": str(ws),
        "host": socket.gethostname(),
        "features_sha256": sha,
        "n_participants": int(pd.Series(pre["groups"]).nunique()),
        "n_windows": int(len(feat)),
        "folds": a.folds, "fold_seed_name": fsp.fold_seed_name(ws),
        "threads_per_task": threads,
        "candidate_family": [c["name"] for c in sap.CANDIDATES],
        "fixed_pipeline": sap.FIXED_PIPELINE,
        "T_perfold": r["T_perfold"],
        "T_global": r["T_global"],
        "T_fixed": r["T_fixed"],
        "T_perfold_foldmean": r["T_perfold_foldmean"],
        "T_global_foldmean": r["T_global_foldmean"],
        "T_fixed_foldmean": r["T_fixed_foldmean"],
        "fold_test_auc": r["fold_test_auc"],
        "selected_global": r["selected_global"],
        "perfold_selected": r["perfold_selected"],
        "n_fallback_folds": r["n_fallback_folds"],
        "fold_inner": r["fold_inner"],
        "inner_mean": r["inner_mean"],
        "n_fits": {k: r[k] for k in ("n_fits", "n_fits_inner", "n_fits_outer_perfold",
                                     "n_fits_outer_fixed_extra", "n_fits_outer_global_extra")},
        "seconds": round(secs, 1),
    }
    if a.compare_observed:
        ob = json.loads(Path(a.compare_observed).read_text())
        ae = ob["analysis_exact"]
        diffs = [abs(r["inner_mean"][k] - ae["inner_mean"][k]) for k in r["inner_mean"]]
        report["observed_comparison"] = {
            "source": str(a.compare_observed),
            "same_features_sha256": ob["features_sha256"] == sha,
            "T_global_bitwise_equal": r["T_global"] == ae["T_selection_aware"],
            "T_fixed_bitwise_equal": r["T_fixed"] == ae["T_fixed"],
            "selected_global_equal": r["selected_global"] == ae["selected_by_inner"],
            "inner_mean_keys_equal": set(r["inner_mean"]) == set(ae["inner_mean"]),
            "inner_mean_bitwise_equal": all(r["inner_mean"][k] == ae["inner_mean"].get(k)
                                            for k in r["inner_mean"]),
            "inner_mean_max_abs_diff": max(diffs) if diffs else 0.0,
        }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report
                      if k in ("T_perfold", "T_global", "T_fixed", "T_perfold_foldmean",
                               "T_global_foldmean", "T_fixed_foldmean", "fold_test_auc",
                               "selected_global", "perfold_selected", "n_fits", "seconds",
                               "observed_comparison")},
                     indent=2))
    return 0


# ------------------------------------------------------------------ run
def _null_summary(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _block_report(ctx, feat: pd.DataFrame, y: np.ndarray, block_of: dict, block_col: str,
                  ws: Path) -> dict:
    """Permutation 0 must keep every block's label count (the check of fsp.cmd_run)."""
    yb = ctx.permute(feat, y, ctx.base_seed)
    per = pd.DataFrame({"g": feat["study_id"], "y": y, "yb": yb}).groupby("g").first()
    per["block"] = per.index.map(block_of)
    cnt = per.groupby("block")[["y", "yb"]].sum()
    rep = {"column": block_col, "n_blocks": int(len(cnt)),
           "permutation_seed_name": f"selperm:{ws.name}:within:{block_col}",
           "perm0_block_positive_counts_preserved": bool((cnt["y"] == cnt["yb"]).all())}
    if not rep["perm0_block_positive_counts_preserved"]:
        raise RuntimeError("within-block permutation changed a block's label count")
    return rep


def cmd_run(a: argparse.Namespace) -> int:
    ws = Path(a.workspace)
    B = a.permutations
    feat, cols, y = fsp.load_workspace(ws)
    threads = a.threads_per_task or None
    block_col = getattr(a, "block_col", None)
    block_report = None
    if block_col:
        block_of = fsp.load_blocks(ws, block_col, None)
        ctx = fsp.make_context(ws, feat, cols, y, a.folds, threads, block_of=block_of,
                               block_col=block_col)
        block_report = _block_report(ctx, feat, y, block_of, block_col, ws)
        default_ckpt = ws / "results" / f"perfold_null_within_{block_col}_checkpoint.csv"
    else:
        ctx = fsp.make_context(ws, feat, cols, y, a.folds, threads)
        default_ckpt = ws / "results" / "perfold_null_checkpoint.csv"
    ckpt = Path(a.checkpoint) if a.checkpoint else default_ckpt
    done = read_perfold_checkpoint(ckpt)
    invalid: set[int] = set()
    deadline = dt.datetime.fromisoformat(a.deadline) if a.deadline else None
    perm_log = Path(a.perm_log) if a.perm_log else None
    todo = [i for i in range(B) if i not in done]
    t_start = time.perf_counter()
    n_new, fits_new, task_sec = 0, 0, 0.0
    stopped_on_deadline = deadline is not None and dt.datetime.now() >= deadline
    if todo and not stopped_on_deadline:
        gen = iter_perfold(ctx, todo, n_jobs=a.n_jobs, chunk_size=a.chunk_size,
                           with_global=a.with_global)
        try:
            for chunk in gen:
                log_rows = []
                for r in chunk:
                    n_new += 1
                    fits_new += r["n_fits"]
                    task_sec += r["seconds"]
                    log_rows.append({k: r[k] for k in (
                        "index", "seconds", "n_fits", "n_fits_inner", "n_fits_outer_perfold",
                        "n_fits_outer_fixed_extra", "n_fits_outer_global_extra", "pid",
                        "T_perfold", "T_fixed", "T_global", "T_perfold_foldmean",
                        "T_fixed_foldmean", "T_global_foldmean", "selected_global",
                        "perfold_selected")}
                        | {"elapsed": time.perf_counter() - t_start,
                           "logged_at": dt.datetime.now().isoformat(timespec="seconds")})
                    if np.isnan(r["T_perfold"]) or np.isnan(r["T_fixed"]):
                        invalid.add(r["index"])
                        continue
                    done[r["index"]] = {c: r[c] for c in PERFOLD_COLUMNS[1:]}
                write_perfold_checkpoint(ckpt, done)
                if perm_log is not None:
                    perm_log.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(log_rows).to_csv(perm_log, mode="a", index=False,
                                                  header=not perm_log.exists())
                if deadline is not None and dt.datetime.now() >= deadline:
                    stopped_on_deadline = True
                    break
        finally:
            gen.close()
    wall = time.perf_counter() - t_start

    in_range = dict(sorted((i, v) for i, v in done.items() if i < B))
    report: dict = {
        "workspace": str(ws),
        "analysis": ("per-fold selection: in each outer fold, the candidate with the highest "
                     "inner-validation participant AUROC of that fold is refitted and "
                     "predicts that fold's test participants"),
        "null": f"within_{block_col}" if block_col else "unrestricted",
        "permutation_seed_name": (f"selperm:{ws.name}:within:{block_col}" if block_col
                                  else f"selperm:{ws.name}"),
        **({"block": block_report} if block_col else {}),
        "fold_seed_name": fsp.fold_seed_name(ws),
        "candidate_family": [c["name"] for c in sap.CANDIDATES],
        "fixed_pipeline": sap.FIXED_PIPELINE,
        "B_requested": B,
        "B_completed": len(in_range),
        "completed_indices_are_prefix": sorted(in_range) == list(range(len(in_range))),
        "invalid_indices_this_call": sorted(invalid),
        "stopped_on_deadline": stopped_on_deadline,
        "deadline": a.deadline,
    }
    if len(in_range) >= 2:
        cols_ = [c for c in FLOAT_COLUMNS
                 if not any(np.isnan(v[c]) for v in in_range.values())]
        nulls = {c: np.array([v[c] for v in in_range.values()]) for c in cols_}
        report["null_summary"] = {c: _null_summary(v) for c, v in nulls.items()}
        if a.observed_from:
            ob = json.loads(Path(a.observed_from).read_text())
            report["tests"] = {c: sap.mc_p(v, ob[c]) for c, v in nulls.items()}
    report["execution"] = {
        "host": socket.gethostname(), "n_jobs": a.n_jobs, "threads_per_task": threads,
        "chunk_size": a.chunk_size or (a.n_jobs if a.n_jobs > 1 else 1),
        "with_global": a.with_global, "new_permutations": n_new,
        "wall_seconds": round(wall, 2),
        "task_seconds_mean": round(task_sec / n_new, 2) if n_new else None,
        "fits_per_permutation_mean": round(fits_new / n_new, 2) if n_new else None,
    }
    report["checkpoint"] = str(ckpt)
    out = Path(a.out) if a.out else ckpt.with_name(ckpt.name.replace("_checkpoint.csv",
                                                                       ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report
                      if k in ("B_completed", "null_summary", "tests",
                               "stopped_on_deadline", "execution")}, indent=2))
    return 0


# ------------------------------------------------------------------ report
def _pvalue(null: np.ndarray, obs: float) -> dict:
    n_ge = int((null >= obs).sum())
    return {"observed": obs, "B": int(len(null)), "n_ge": n_ge,
            "p_value": (1 + n_ge) / (len(null) + 1), **_null_summary(null)}


def _family(obs: dict, nulls: dict[str, np.ndarray]) -> dict:
    """p-values, null shift vs the fixed null, and recycled Type-I, for one statistic family."""
    fx = nulls["fixed"]
    sd_fx = float(fx.std(ddof=1))
    out: dict = {"tests": {k: _pvalue(v, obs[k]) for k, v in nulls.items()}}
    out["null_shift_vs_fixed"] = {
        k: {"mean_shift": float(v.mean() - fx.mean()),
            "mean_shift_sd_units": float((v.mean() - fx.mean()) / sd_fx),
            "p95_shift": float(np.percentile(v, 95) - np.percentile(fx, 95)),
            "p95_shift_sd_units": float((np.percentile(v, 95) - np.percentile(fx, 95)) / sd_fx)}
        for k, v in nulls.items() if k != "fixed"}
    out["recycled_typeI"] = {f"{k}_vs_fixed_null": recycled_typeI(v, fx)
                             for k, v in nulls.items()}
    out["recycled_typeI"].update({f"{k}_vs_own_null": recycled_typeI(v, v)
                                  for k, v in nulls.items() if k != "fixed"})
    if "global" in nulls:
        d = nulls["perfold"] - nulls["global"]
        npos, nneg = int((d > 0).sum()), int((d < 0).sum())
        out["paired_perfold_minus_global"] = {
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)), "median": float(np.median(d)),
            "n_positive": npos, "n_negative": nneg, "n_zero": int((d == 0).sum()),
            "sign_test_p": (float(stats.binomtest(npos, npos + nneg).pvalue)
                            if npos + nneg else None),
            "corr": float(np.corrcoef(nulls["perfold"], nulls["global"])[0, 1])}
    return out


def cmd_report(a: argparse.Namespace) -> int:
    """Two statistic families on the same permutations.

    pooled    participant AUROC of the pooled out-of-fold scores (the paper's global
              statistic; per-fold picks mix model score scales across folds)
    foldmean  mean of the per-fold test participant AUROCs
    Global values come from this checkpoint when it has them (run --with-global);
    otherwise the pooled global null comes from the global-selection checkpoint.
    """
    ob = json.loads(Path(a.observed).read_text())
    pf = read_perfold_checkpoint(Path(a.checkpoint))
    gl = read_checkpoint(Path(a.global_checkpoint)) if a.global_checkpoint else {}
    fx = read_checkpoint(Path(a.fixed_checkpoint)) if a.fixed_checkpoint else {}
    idx = sorted(pf)
    own_global = bool(idx) and not any(np.isnan(pf[i]["T_global"]) for i in idx)
    if gl:
        idx = [i for i in idx if i in gl]
    if fx:
        idx = [i for i in idx if i in fx]
    if len(idx) < 2:
        raise SystemExit(f"only {len(idx)} common permutation indices")

    def col(c: str) -> np.ndarray:
        return np.array([pf[i][c] for i in idx])
    pooled = {"perfold": col("T_perfold"), "fixed": col("T_fixed")}
    if own_global:
        pooled["global"] = col("T_global")
    elif gl:
        pooled["global"] = np.array([gl[i][0] for i in idx])
    families = {"pooled": _family({"perfold": ob["T_perfold"], "global": ob["T_global"],
                                   "fixed": ob["T_fixed"]}, pooled)}
    families["pooled"]["global_source"] = ("this checkpoint" if own_global
                                           else str(a.global_checkpoint) if gl else None)
    if own_global:
        fm = {"perfold": col("T_perfold_foldmean"), "global": col("T_global_foldmean"),
              "fixed": col("T_fixed_foldmean")}
        families["foldmean"] = _family({"perfold": ob["T_perfold_foldmean"],
                                        "global": ob["T_global_foldmean"],
                                        "fixed": ob["T_fixed_foldmean"]}, fm)
    picks = [pf[i]["perfold_selected"].split("|") for i in idx]
    flat = Counter(p for row in picks for p in row if p != SKIP)
    n_folds = sum(flat.values())
    report: dict = {
        "B_common": len(idx),
        "indices": {"first": idx[0], "last": idx[-1],
                    "prefix": idx == list(range(len(idx)))},
        "B_perfold_checkpoint": len(pf),
        "observed_source": str(a.observed),
        "observed": {k: ob[k] for k in ("T_perfold", "T_global", "T_fixed",
                                        "T_perfold_foldmean", "T_global_foldmean",
                                        "T_fixed_foldmean", "perfold_selected",
                                        "selected_global", "fold_test_auc")},
        "families": families,
        "picks_under_null": {
            "perfold_pick_share": {k: round(v / n_folds, 4) for k, v in flat.most_common()},
            "global_pick_share": {k: round(v / len(idx), 4) for k, v in Counter(
                pf[i]["selected_global"] for i in idx).most_common()},
            "share_all_folds_same_pick": float(np.mean([len(set(r) - {SKIP}) == 1
                                                        for r in picks])),
            "share_all_folds_equal_global_pick": float(np.mean(
                [set(r) - {SKIP} == {pf[i]["selected_global"]}
                 for r, i in zip(picks, idx, strict=False)])),
            "fallback_folds_total": int(sum(pf[i]["n_fallback_folds"] for i in idx)),
        },
    }
    consistency: dict = {}
    if fx:
        consistency["T_fixed_bitwise_equal_to_fixed_checkpoint"] = {
            "source": str(a.fixed_checkpoint),
            "n_equal": int(sum(pf[i]["T_fixed"] == fx[i][1] for i in idx)), "n": len(idx)}
    if gl:
        consistency["T_fixed_bitwise_equal_to_global_checkpoint"] = {
            "source": str(a.global_checkpoint),
            "n_equal": int(sum(pf[i]["T_fixed"] == gl[i][1] for i in idx)), "n": len(idx)}
        consistency["selected_global_equal_to_global_checkpoint"] = {
            "n_equal": int(sum(pf[i]["selected_global"] == gl[i][2] for i in idx)),
            "n": len(idx)}
        if own_global:
            consistency["T_global_bitwise_equal_to_global_checkpoint"] = {
                "n_equal": int(sum(pf[i]["T_global"] == gl[i][0] for i in idx)),
                "n": len(idx)}
    report["consistency_with_global_run"] = consistency
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"B_common": report["B_common"],
                      "consistency_with_global_run": consistency,
                      **{f: {k: fam[k] for k in ("tests", "null_shift_vs_fixed")}
                         for f, fam in families.items()}}, indent=2))
    return 0


# ------------------------------------------------------------------ checks
def cmd_checks(a: argparse.Namespace) -> int:
    ob = json.loads(Path(a.observed).read_text())
    log = pd.read_csv(a.perm_log)
    checks: dict = {
        "T_obs_perfold": ob["T_perfold"],
        "T_obs_global": ob["T_global"],
        "perfold_selected": ob["perfold_selected"],
        "sec_per_perm": round(float(log["seconds"].mean()), 1),
        "run": {"host": a.host, "pid": a.pid, "log": a.log, "B_target": a.b_target},
        "T_obs_fixed": ob["T_fixed"],
        "T_obs_foldmean": {k: ob[f"T_{k}_foldmean"] for k in ("perfold", "global", "fixed")},
        "selected_global": ob["selected_global"],
        "sec_per_perm_note": (f"mean single-core seconds per permutation over the "
                              f"{len(log)} permutations in {Path(a.perm_log).name}, "
                              f"n_jobs {a.n_jobs}"),
        "fits_per_perm_mean": round(float(log["n_fits"].mean()), 2),
        "observed_matches_global_run": ob.get("observed_comparison"),
    }
    for extra in a.observed_remote or []:
        rem = json.loads(Path(extra).read_text())
        checks.setdefault("observed_reproduced", {})[rem["host"]] = {
            "source": str(extra),
            **{f"{k}_bitwise_equal": rem[k] == ob[k]
               for k in ("T_perfold", "T_global", "T_fixed", "T_perfold_foldmean",
                         "T_global_foldmean", "T_fixed_foldmean")},
            "perfold_selected_equal": rem["perfold_selected"] == ob["perfold_selected"],
            "observed_comparison": rem.get("observed_comparison")}
    if a.report:
        rep = json.loads(Path(a.report).read_text())
        checks["first_rows"] = {"B_common": rep["B_common"],
                                "consistency_with_global_run": rep["consistency_with_global_run"]}
    if a.junit:
        suite = ET.parse(a.junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        n, bad = int(suite.get("tests")), int(suite.get("failures")) + int(suite.get("errors"))
        checks["tests_passed"] = n - bad - int(suite.get("skipped"))
        checks["tests_failed"] = bad
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
    nhanes = str(ROOT / "reports" / "real_nhanes")

    o = sub.add_parser("observed", help="A' on the observed labels")
    o.add_argument("--workspace", default=nhanes)
    o.add_argument("--folds", type=int, default=5)
    o.add_argument("--threads-per-task", type=int, default=1)
    o.add_argument("--compare-observed", default=None,
                   help="global-selection `observed` JSON to compare with")
    o.add_argument("--out", default=str(BIGDATA / "perfold_observed.json"))
    o.set_defaults(func=cmd_observed)

    r = sub.add_parser("run", help="the per-fold null (unrestricted, or within blocks)")
    r.add_argument("--workspace", default=nhanes)
    r.add_argument("--permutations", type=int, default=999)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--chunk-size", type=int, default=None,
                   help="permutations per checkpoint write; default n_jobs")
    r.add_argument("--threads-per-task", type=int, default=1,
                   help="BLAS/OpenMP threads per permutation (0 = no limit)")
    r.add_argument("--deadline", default=None,
                   help="local ISO time; stop after the first checkpoint at or after it")
    r.add_argument("--checkpoint", default=None,
                   help="default: <workspace>/results/perfold_null_checkpoint.csv")
    r.add_argument("--perm-log", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--with-global", action="store_true",
                   help="also refit the global pick and store T_global (off by default)")
    r.add_argument("--observed-from", default=None, help="per-fold observed JSON, for p-values")
    r.add_argument("--block-col", default=None,
                   help="permute labels within blocks of this participant_metadata column, "
                        "as in the global-selection within-block run (off by default)")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="per-fold vs global vs fixed nulls")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--observed", default=str(BIGDATA / "perfold_observed.json"))
    p.add_argument("--global-checkpoint", default=None, help="global-selection checkpoint")
    p.add_argument("--fixed-checkpoint", default=None, help="fixed-pipeline checkpoint")
    p.add_argument("--out", default=str(BIGDATA / "perfold_report.json"))
    p.set_defaults(func=cmd_report)

    c = sub.add_parser("checks", help="write the per-fold checks JSON")
    c.add_argument("--observed", default=str(BIGDATA / "perfold_observed.json"))
    c.add_argument("--observed-remote", nargs="*", default=None)
    c.add_argument("--perm-log", required=True)
    c.add_argument("--report", default=None)
    c.add_argument("--junit", default=None)
    c.add_argument("--host", required=True)
    c.add_argument("--pid", type=int, required=True)
    c.add_argument("--log", required=True)
    c.add_argument("--b-target", type=int, default=999)
    c.add_argument("--n-jobs", type=int, default=56)
    c.add_argument("--out", default=str(BIGDATA / "perfold_checks.json"))
    c.set_defaults(func=cmd_checks)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")   # loky workers inherit this
    raise SystemExit(main())
