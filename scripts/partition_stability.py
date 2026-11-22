#!/usr/bin/env python
"""Repeated grouped-partition stability.

The grouped, stratified five-fold schedule is regenerated under many independent
seeds, the complete participant-level procedure is rerun for each, and the
distribution of the estimate is reported. Two statistics per seed:

  fixed        the prespecified histogram gradient-boosting pipeline
  selected     the best candidate in the prespecified family, chosen by mean
               inner-validation participant AUROC (the rule used by the
               selection-aware permutation test)

Comparing the two separates the spread due to partitioning from the spread due
to partitioning plus selection. The script also reports the fraction of seeds
whose estimate exceeds 0.70, 0.80, and 0.90.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from selection_aware_permutation import FIXED_PIPELINE, analysis, precompute

from wearabledata.features.splits import nested_grouped_cv

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT / "reports" / "real_depresjon"))
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fixed-only", action="store_true",
                    help="skip the candidate family; much faster")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ws = Path(a.workspace)
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat.columns]
    y = feat["label"].to_numpy()

    out = Path(a.out) if a.out else ws / "results" / "partition_stability.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if out.exists():
        done = {int(r["seed"]): dict(r) for _, r in pd.read_csv(out).iterrows()}

    start = time.perf_counter()
    for s in range(a.seeds):
        if s in done:
            continue
        if a.time_budget and time.perf_counter() - start > a.time_budget:
            break
        folds = nested_grouped_cv(feat, outer_folds=a.folds,
                                  seed_name=f"partition:{ws.name}:{s}")
        pre = precompute(feat, cols, folds)
        # Fold composition diagnostic: the smallest positive count in any test fold.
        min_pos = min(int(feat[feat["study_id"].isin(f["test"])]
                          .groupby("study_id")["label"].first().sum()) for f in folds)
        r = analysis(feat, cols, y, folds, pre=pre)
        rec = {"seed": s,
               "auroc_fixed": r["per_candidate_outer"].get(FIXED_PIPELINE, np.nan),
               "auroc_selected": r["T_selection_aware"],
               "selected_model": r["selected_by_inner"],
               "auroc_oracle_max": r["T_oracle_max"],
               "min_positives_in_a_test_fold": min_pos,
               "data_source": f"REAL:{ws.name}"}
        done[s] = rec
        pd.DataFrame(list(done.values())).to_csv(out, index=False)

    df = pd.DataFrame(list(done.values()))
    if df.empty:
        print("no seeds completed; rerun to extend")
        return 0

    summary = {"workspace": str(ws), "n_seeds": int(len(df)), "folds": a.folds}
    for col, label in (("auroc_fixed", "fixed_pipeline"),
                       ("auroc_selected", "selection_aware"),
                       ("auroc_oracle_max", "oracle_best_candidate")):
        v = df[col].dropna().to_numpy()
        if len(v) == 0:
            continue
        summary[label] = {
            "median": round(float(np.median(v)), 4),
            "iqr": [round(float(np.percentile(v, 25)), 4),
                    round(float(np.percentile(v, 75)), 4)],
            "p5_p95": [round(float(np.percentile(v, 5)), 4),
                       round(float(np.percentile(v, 95)), 4)],
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "range": round(float(v.max() - v.min()), 4),
            "frac_above_0.70": round(float((v > 0.70).mean()), 3),
            "frac_above_0.80": round(float((v > 0.80).mean()), 3),
            "frac_above_0.90": round(float((v > 0.90).mean()), 3)}
    if "selected_model" in df:
        summary["selected_model_frequency"] = {
            k: round(float(v), 3)
            for k, v in df["selected_model"].value_counts(normalize=True).items()}
    # Does fold class balance explain the spread?
    if df["min_positives_in_a_test_fold"].nunique() > 1:
        ok = df["auroc_fixed"].notna()
        summary["corr_auroc_vs_min_fold_positives"] = round(float(np.corrcoef(
            df.loc[ok, "min_positives_in_a_test_fold"], df.loc[ok, "auroc_fixed"])[0, 1]), 4)
    summary["reading"] = (
        "the spread across seeds is partition variability under an otherwise "
        "identical procedure; the selection-aware column adds the variability "
        "contributed by choosing a winner among candidates")
    summary["data_source"] = f"REAL:{ws.name}"

    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
