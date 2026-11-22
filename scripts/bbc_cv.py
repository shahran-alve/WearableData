#!/usr/bin/env python
"""Participant-level BBC-CV on one workspace.

The permutation test says whether there is signal. BBC-CV (Tsamardinos et al., Machine
Learning 2018) says how large it is once the optimism of picking the best of K candidates
is removed, with a percentile interval. It is descriptive; the test stays the permutation p.

Subcommands
  run      out-of-fold participant scores of all candidates on the observed labels, with
           the original folds and seeds (40 outer fits for K = 8, in parallel), then
           BBC-CV. Writes reports/bigdata/bbccv_<tag>.json and bbccv_<tag>_scores.csv.
           --scores-from reuses a scores CSV instead of refitting.
           --compare-observed LABEL=PATH checks the per-candidate pooled AUROCs against an
           `observed` JSON of fast_selection_permutation.py (the same features.csv is required).
           --verify-analysis also runs the original analysis() (80 fits) and compares.
  checks   reports/bigdata/bbccv_checks.json from the run JSONs and the pytest junit XML.

The candidate family, builder, aggregation and fold construction are imported from
scripts/selection_aware_permutation.py and scripts/participant_level_eval.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import participant_level_eval as ple
import selection_aware_permutation as sap
from sklearn.metrics import roc_auc_score

from wearabledata.evaluation.bbccv import bbc_cv, oof_matrix
from wearabledata.evaluation.fastperm import _thread_limit

ROOT = Path(__file__).resolve().parents[1]
BIGDATA = ROOT / "reports" / "bigdata"
TOL = 1e-12


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_workspace(ws: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Same loading as selection_aware_permutation.main()."""
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat.columns]
    return feat, cols, feat["label"].to_numpy()


def compare(ours: dict[str, float], ref: dict[str, float]) -> dict:
    per = {}
    for name in sorted(set(ours) | set(ref)):
        a, b = ours.get(name), ref.get(name)
        if a is None or b is None:
            per[name] = {"bbccv": a, "observed": b, "bitwise": False, "abs_diff": None}
            continue
        per[name] = {"bbccv": a, "observed": b, "bitwise": a == b, "abs_diff": abs(a - b)}
    diffs = [v["abs_diff"] for v in per.values() if v["abs_diff"] is not None]
    complete = all(v["abs_diff"] is not None for v in per.values())
    return {"bitwise_all": complete and all(v["bitwise"] for v in per.values()),
            "within_1e-12": complete and max(diffs, default=0.0) <= TOL,
            "max_abs_diff": max(diffs, default=None),
            "n_bitwise": sum(v["bitwise"] for v in per.values()),
            "n_candidates": len(per), "per_candidate": per}


def scores_frame(ids, y, S, names) -> pd.DataFrame:
    df = pd.DataFrame(S, columns=names)
    df.insert(0, "label", np.asarray(y).astype(int))
    df.insert(0, "study_id", ids)
    return df


def cmd_run(a: argparse.Namespace) -> int:
    ws = Path(a.workspace)
    feat_path = ws / "interim" / "features.csv"
    sha = sha256_path(feat_path)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    host = a.host_label or socket.gethostname()
    t_start = time.perf_counter()
    report: dict = {"tag": a.tag, "workspace": str(ws), "features_sha256": sha,
                    "host": host, "data_source": ("SYNTHETIC:" if (ws / "interim" /
                                                  "SYNTHETIC.json").exists() else "REAL:")
                    + ws.name}
    refs: list[tuple[str, dict]] = []
    for item in a.compare_observed or []:
        label, _, path = item.partition("=")
        ob = json.loads(Path(path).read_text())
        if ob["features_sha256"] != sha:
            raise SystemExit(f"{path} was computed on a different features.csv")
        refs.append((label, {"path": path, **ob["analysis_exact"]}))

    if a.scores_from:
        df = pd.read_csv(a.scores_from, float_precision="round_trip")
        names = [c for c in df.columns if c not in ("study_id", "label")]
        S, y_p, ids = df[names].to_numpy(float), df["label"].to_numpy(int), df["study_id"]
        per_candidate = {n: float(roc_auc_score(y_p, S[:, j]))
                         for j, n in enumerate(names)}
        report["scores"] = {"source": str(a.scores_from)}
    else:
        feat, cols, y = load_workspace(ws)
        folds = sap.make_folds(feat, y, a.folds, f"cv:{ws.name}:primary")
        pre = sap.precompute(feat, cols, folds)
        om = oof_matrix(y, pre, sap.CANDIDATES, sap.build, ple.aggregate, seed=0,
                        n_jobs=a.n_jobs, threads=a.threads_per_task)
        S, y_p, ids, names = om["S"], om["y"], om["ids"], om["names"]
        per_candidate = om["per_candidate_outer"]
        report["scores"] = {
            "source": "fitted", "n_windows": int(len(feat)), "n_fits": om["n_fits"],
            "n_valid_folds": om["n_valid_folds"], "folds": a.folds,
            "fold_seed_name": f"cv:{ws.name}:primary", "n_jobs": a.n_jobs,
            "threads_per_task": a.threads_per_task, "n_workers_seen": om["n_workers_seen"],
            "wall_seconds": round(om["wall_seconds"], 1),
            "fit_seconds_by_candidate": {k: round(v, 1) for k, v in om["fit_seconds"].items()}}
        scores_path = out_dir / f"bbccv_{a.tag}_scores.csv"
        scores_frame(ids, y_p, S, names).to_csv(scores_path, index=False)
        report["scores"]["csv"] = str(scores_path)
        if a.verify_analysis:
            t0 = time.perf_counter()
            with _thread_limit(a.threads_per_task):
                orig = sap.analysis(feat, cols, y, folds, pre=pre)
            refs.append(("analysis()", {"path": "scripts/selection_aware_permutation.py "
                                        "analysis(), this run", **orig}))
            report["verify_analysis_seconds"] = round(time.perf_counter() - t0, 1)

    report["n_participants"] = int(len(y_p))
    report["n_positive"] = int(np.sum(y_p))
    report["candidates"] = names
    report["per_candidate_outer"] = per_candidate
    report["compare_observed"] = {
        label: {"path": r["path"], "selected_by_inner": r.get("selected_by_inner"),
                **compare(per_candidate, r["per_candidate_outer"])} for label, r in refs}

    single = a.single or next((r.get("selected_by_inner") for _, r in refs
                               if r.get("selected_by_inner")), None) or "rf_leaf5"
    t0 = time.perf_counter()
    bbc = bbc_cv(S, y_p, names, B=a.B, seed_name=a.seed_name or f"bbccv:{ws.name}",
                 single=single)
    bbc["seconds"] = round(time.perf_counter() - t0, 1)
    bbc["single_is_global_rule_pick"] = any(r.get("selected_by_inner") == single
                                            for _, r in refs)
    report["bbc"] = bbc
    report["seconds_total"] = round(time.perf_counter() - t_start, 1)
    out = out_dir / f"bbccv_{a.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    show = {k: bbc[k] for k in ("naive_max", "naive_winner", "bbc_corrected", "ci",
                                "optimism", "winner_share", "n_valid_resamples")}
    show["single"] = bbc.get("single")
    show["compare_observed"] = {k: {kk: v[kk] for kk in ("bitwise_all", "max_abs_diff",
                                                           "n_bitwise")}
                                for k, v in report["compare_observed"].items()}
    print(json.dumps(show, indent=2))
    print(f"wrote {out}")
    return 0


def _junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    n = int(suite.get("tests"))
    failed = int(suite.get("failures")) + int(suite.get("errors"))
    skipped = int(suite.get("skipped"))
    return {"passed": n - failed - skipped, "failed": failed, "skipped": skipped}


def cmd_checks(a: argparse.Namespace) -> int:
    nh = json.loads(Path(a.nhanes).read_text())
    local = json.loads(Path(a.nhanes_local).read_text()) if a.nhanes_local else None
    null = json.loads(Path(a.null).read_text())["bbc"]
    eff = json.loads(Path(a.effect).read_text())["bbc"]
    b = nh["bbc"]

    def same_host(rep: dict) -> tuple[bool, dict]:
        cmp = rep["compare_observed"]
        hit = cmp.get(rep["host"])
        return (bool(hit and hit["bitwise_all"]),
                {lbl: {"bitwise_all": v["bitwise_all"], "max_abs_diff": v["max_abs_diff"],
                       "n_bitwise": v["n_bitwise"]} for lbl, v in cmp.items()})

    ok_remote, detail_remote = same_host(nh)
    detail = {nh["host"]: detail_remote}
    ok = ok_remote
    if local is not None:
        ok_local, detail_local = same_host(local)
        detail[local["host"]] = detail_local
        ok = ok and ok_local
    junit = _junit(Path(a.junit))
    checks = {
        "nhanes": {"naive_max": b["naive_max"], "bbc_corrected": b["bbc_corrected"],
                   "ci": b["ci"], "optimism": b["optimism"],
                   "winner_share": b["winner_share"],
                   "matches_observed_per_candidate": ok,
                   "matches_observed_detail": detail},
        "synthetic_null": {"naive_max": null["naive_max"],
                           "bbc_corrected": null["bbc_corrected"]},
        "synthetic_effect": {"naive_max": eff["naive_max"],
                             "bbc_corrected": eff["bbc_corrected"]},
        "tests_passed": junit["passed"],
    }
    out = Path(a.out)
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="scores + BBC-CV on one workspace")
    r.add_argument("--workspace", default=str(ROOT / "reports" / "real_nhanes"))
    r.add_argument("--tag", required=True, help="output names bbccv_<tag>.json, _scores.csv")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--threads-per-task", type=int, default=1)
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--seed-name", default=None, help="default bbccv:<workspace name>")
    r.add_argument("--single", default=None,
                   help="single-candidate bootstrap; default the compared global pick")
    r.add_argument("--scores-from", default=None)
    r.add_argument("--compare-observed", action="append", metavar="LABEL=PATH")
    r.add_argument("--verify-analysis", action="store_true")
    r.add_argument("--host-label", default=None, help="default socket.gethostname()")
    r.add_argument("--out-dir", default=str(BIGDATA))
    c = sub.add_parser("checks", help="write bbccv_checks.json")
    c.add_argument("--nhanes", default=str(BIGDATA / "bbccv_nhanes.json"))
    c.add_argument("--nhanes-local", default=None)
    c.add_argument("--null", default=str(BIGDATA / "bbccv_synthetic_null.json"))
    c.add_argument("--effect", default=str(BIGDATA / "bbccv_synthetic_effect.json"))
    c.add_argument("--junit", default=str(BIGDATA / "bbccv_pytest_junit.xml"))
    c.add_argument("--out", default=str(BIGDATA / "bbccv_checks.json"))
    a = ap.parse_args()
    return {"run": cmd_run, "checks": cmd_checks}[a.cmd](a)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
