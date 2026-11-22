#!/usr/bin/env python
"""Selection-aware participant-label permutation test.

Two nulls are computed and reported:

  fixed-pipeline null      one pipeline chosen before analysis; its complete
                           fitting process is repeated under permutation.
  selection-aware null     for every permutation the whole candidate family is
                           evaluated and the winner is selected by the same inner
                           rule used on the observed labels.

The gap between them measures the optimism added by model selection.

Folds are grouped and stratified on the labels, so fold construction is part of
the analysis and is repeated for every permuted label vector, with the same
deterministic splitter and seed schedule. Reusing the observed-label folds would
stratify each permuted fit on labels that no longer apply and bias the null
downward; an earlier version did this and gave null means of 0.471 and 0.443.
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
from participant_level_eval import aggregate
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from wearabledata.features.splits import mask_for, nested_grouped_cv
from wearabledata.utils.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Pre-specified candidate family. Frozen before the final analysis; the same
# family and the same inner-selection rule are used for observed and permuted
# labels. The feature branch is part of the candidate definition.
# --------------------------------------------------------------------------
CANDIDATES: list[dict] = [
    {"name": "logreg_C0.1_all", "family": "logreg", "C": 0.1, "branch": "all"},
    {"name": "logreg_C1_all", "family": "logreg", "C": 1.0, "branch": "all"},
    {"name": "logreg_C1_population", "family": "logreg", "C": 1.0, "branch": "population"},
    {"name": "logreg_C1_relative", "family": "logreg", "C": 1.0, "branch": "relative"},
    {"name": "rf_leaf1", "family": "rf", "min_samples_leaf": 1, "branch": "all"},
    {"name": "rf_leaf5", "family": "rf", "min_samples_leaf": 5, "branch": "all"},
    {"name": "hgb_lr0.1_leaf31", "family": "hgb", "learning_rate": 0.1,
     "max_leaf_nodes": 31, "max_iter": 200, "branch": "all"},
    {"name": "hgb_lr0.03_leaf15", "family": "hgb", "learning_rate": 0.03,
     "max_leaf_nodes": 15, "max_iter": 300, "branch": "all"},
]
FIXED_PIPELINE = "hgb_lr0.1_leaf31"     # the prespecified primary pipeline


def branch_columns(cols: list[str], branch: str) -> list[str]:
    if branch == "population":
        return [c for c in cols if not c.startswith(("pz__", "dev__", "chg__"))]
    if branch == "relative":
        rel = [c for c in cols if c.startswith(("pz__", "dev__", "chg__"))]
        qual = [c for c in cols if c.startswith("cov__")
                or c in ("window_coverage", "valid_days")]
        return rel + qual
    return cols


def build(spec: dict, seed: int = 0) -> Pipeline:
    steps: list = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if spec["family"] == "logreg":
        steps += [("scale", RobustScaler()),
                  ("clf", LogisticRegression(C=spec["C"], max_iter=2000,
                                             class_weight="balanced",
                                             random_state=seed))]
    elif spec["family"] == "rf":
        steps.append(("clf", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=spec["min_samples_leaf"],
            class_weight="balanced_subsample", n_jobs=1, random_state=seed)))
    else:
        steps.append(("clf", HistGradientBoostingClassifier(
            learning_rate=spec["learning_rate"], max_leaf_nodes=spec["max_leaf_nodes"],
            max_iter=spec["max_iter"], early_stopping=False, random_state=seed)))
    return Pipeline(steps)


def precompute(feat: pd.DataFrame, cols: list[str], folds: list[dict]) -> dict:
    """Materialise branch matrices and fold masks once.

    Repeated `feat.loc[mask, subset]` slicing dominated runtime: the permutation
    loop touches the same matrices thousands of times and pandas copies each one.
    """
    branches = {b: feat[branch_columns(cols, b)].to_numpy(dtype=float)
                for b in ("all", "population", "relative")}
    masks = [{k: mask_for(feat, f[k]) for k in ("train", "validation", "test")}
             for f in folds]
    return {"branches": branches, "masks": masks,
            "groups": feat["study_id"].to_numpy()}


def make_folds(feat: pd.DataFrame, y: np.ndarray, n_folds: int, seed_name: str) -> list[dict]:
    """Grouped-stratified folds built from the supplied labels.

    Called with the observed labels for T_obs and with each permuted vector under
    the null, so the label-dependent part of the design is inside A.
    """
    tmp = feat[["study_id"]].copy()
    tmp["label"] = y
    return nested_grouped_cv(tmp, outer_folds=n_folds, seed_name=seed_name)


def analysis(feat: pd.DataFrame, cols: list[str], y: np.ndarray, folds: list[dict],
             seed: int = 0, pre: dict | None = None) -> dict:
    """The complete analysis algorithm A.

    For each candidate: fit on the inner training participants, score the inner
    validation participants (the selection signal), and refit on train+validation
    to predict the outer test fold. Selection uses inner-validation participant
    AUROC only; the outer statistic is never consulted for selection.
    """
    pre = pre or precompute(feat, cols, folds)
    groups = pre["groups"]
    inner_scores: dict[str, list[float]] = {c["name"]: [] for c in CANDIDATES}
    oof: dict[str, np.ndarray] = {c["name"]: np.full(len(feat), np.nan)
                                  for c in CANDIDATES}

    for fm in pre["masks"]:
        tr, va, te = fm["train"], fm["validation"], fm["test"]
        fit = tr | va
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[fit])) < 2:
            continue
        for spec in CANDIDATES:
            X = pre["branches"][spec["branch"]]
            # inner selection signal: fitted on train, scored on validation
            if va.sum() and len(np.unique(y[va])) > 1:
                m = build(spec, seed)
                m.fit(X[tr], y[tr])
                pv = m.predict_proba(X[va])[:, 1]
                pa, ya, _ = aggregate(pv, groups[va], y[va])
                if len(np.unique(ya)) > 1:
                    inner_scores[spec["name"]].append(float(roc_auc_score(ya, pa)))
            # outer prediction: refit on train+validation
            m = build(spec, seed)
            m.fit(X[fit], y[fit])
            oof[spec["name"]][te] = m.predict_proba(X[te])[:, 1]

    out: dict[str, float] = {}
    for spec in CANDIDATES:
        p = oof[spec["name"]]
        ok = ~np.isnan(p)
        if ok.sum() < 10:
            continue
        pa, ya, _ = aggregate(p[ok], groups[ok], y[ok])
        if len(np.unique(ya)) > 1:
            out[spec["name"]] = float(roc_auc_score(ya, pa))

    inner_mean = {k: float(np.mean(v)) for k, v in inner_scores.items() if v}
    selected = max(inner_mean, key=lambda k: (inner_mean[k], k)) if inner_mean else None
    return {
        "per_candidate_outer": out,
        "inner_mean": inner_mean,
        "selected_by_inner": selected,
        "T_selection_aware": out.get(selected, float("nan")) if selected else float("nan"),
        "T_fixed": out.get(FIXED_PIPELINE, float("nan")),
        "T_oracle_max": max(out.values()) if out else float("nan"),
    }


def permute(feat: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    per = pd.DataFrame({"g": feat["study_id"], "y": y}).groupby("g")["y"].first()
    g = np.random.default_rng(seed)
    mapping = dict(zip(per.index.to_numpy(), g.permutation(per.to_numpy()), strict=True))
    return feat["study_id"].map(mapping).to_numpy().astype(int)


def mc_p(null: np.ndarray, obs: float) -> dict:
    n_ge = int((null >= obs).sum())
    B = len(null)
    p = (1 + n_ge) / (B + 1)
    lo, hi = stats.binomtest(n_ge, B).proportion_ci(confidence_level=0.95)
    return {"observed": round(obs, 4), "B": B, "n_ge": n_ge,
            "p_value": round(p, 5),
            "p_ci": [round(float((1 + lo * B) / (B + 1)), 5),
                     round(float((1 + hi * B) / (B + 1)), 5)],
            "smallest_attainable_p": round(1 / (B + 1), 5),
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std(ddof=1)), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_max": round(float(null.max()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT / "reports" / "real_depresjon"))
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ws = Path(a.workspace)
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat.columns]
    y = feat["label"].to_numpy()
    folds = nested_grouped_cv(feat, outer_folds=a.folds, seed_name=f"cv:{ws.name}:primary")

    folds = make_folds(feat, y, a.folds, f"cv:{ws.name}:primary")
    pre = precompute(feat, cols, folds)
    obs = analysis(feat, cols, y, folds, pre=pre)
    ckpt = ws / "results" / "selection_aware_null_checkpoint.csv"
    done: dict[int, tuple[float, float, str]] = {}
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        done = {int(r["index"]): (float(r["T_selection_aware"]), float(r["T_fixed"]),
                                  str(r["selected"])) for _, r in prev.iterrows()}

    base = derive_seed(f"selperm:{ws.name}")
    start = time.perf_counter()
    for i in range(a.permutations):
        if i in done:
            continue
        if a.time_budget and time.perf_counter() - start > a.time_budget:
            break
        yb = permute(feat, y, base + i)
        # Regenerate the label-dependent design from the permuted labels.
        folds_b = make_folds(feat, yb, a.folds, f"cv:{ws.name}:primary")
        r = analysis(feat, cols, yb, folds_b,
                     pre=precompute(feat, cols, folds_b))
        if np.isnan(r["T_selection_aware"]) or np.isnan(r["T_fixed"]):
            continue
        done[i] = (r["T_selection_aware"], r["T_fixed"], r["selected_by_inner"] or "none")
        pd.DataFrame([{"index": k, "T_selection_aware": v[0], "T_fixed": v[1],
                       "selected": v[2]} for k, v in done.items()]).to_csv(ckpt, index=False)

    if not done:
        print("no permutations completed yet; rerun to extend the checkpoint")
        return 0
    sel_null = np.array([v[0] for v in done.values()])
    fix_null = np.array([v[1] for v in done.values()])
    picks = pd.Series([v[2] for v in done.values()]).value_counts(normalize=True)

    # ---- paired comparison at MATCHED B on the SAME permutation ids -----
    delta = sel_null - fix_null
    n = len(delta)
    boot = np.random.default_rng(derive_seed("paired_delta"))
    draws = np.array([delta[boot.integers(0, n, n)].mean() for _ in range(4000)])
    paired = {
        "matched_B": int(n),
        "note": ("both statistics are computed on the same permutation ids, so the "
                 "comparison is paired and free of the Monte Carlo-resolution "
                 "artefact that comparing p-values at different B would introduce"),
        "delta_mean": round(float(delta.mean()), 4),
        "delta_median": round(float(np.median(delta)), 4),
        "delta_bootstrap_ci": [round(float(np.percentile(draws, 2.5)), 4),
                               round(float(np.percentile(draws, 97.5)), 4)],
        "frac_delta_positive": round(float((delta > 0).mean()), 4),
        "sign_test_p": round(float(stats.binomtest(int((delta > 0).sum()), n).pvalue), 5),
        "upper_tail": {
            "p95_fixed": round(float(np.percentile(fix_null, 95)), 4),
            "p95_selection": round(float(np.percentile(sel_null, 95)), 4),
            "p95_shift": round(float(np.percentile(sel_null, 95)
                                     - np.percentile(fix_null, 95)), 4),
            "p99_fixed": round(float(np.percentile(fix_null, 99)), 4),
            "p99_selection": round(float(np.percentile(sel_null, 99)), 4),
            "max_fixed": round(float(fix_null.max()), 4),
            "max_selection": round(float(sel_null.max()), 4)},
        "matched_p_values": {
            "fixed": round(float((1 + (fix_null >= obs["T_fixed"]).sum()) / (n + 1)), 5),
            "selection_aware": round(float((1 + (sel_null >= obs["T_selection_aware"]).sum())
                                           / (n + 1)), 5),
            "smallest_attainable": round(1 / (n + 1), 5)},
        "caution": ("a shift in the null mean is not by itself selection optimism: "
                    "inner selection inside outer training folds is proper nested "
                    "evaluation. Whether the upper tail moves is the quantity that "
                    "affects inference, and it requires far more permutations than "
                    "are currently available"),
    }

    report = {
        "workspace": str(ws),
        "paired_comparison": paired,
        "fold_policy": ("regenerated design: grouped-stratified folds are rebuilt "
                        "from each permuted label vector with the same splitting "
                        "algorithm and seed schedule, because stratification reads "
                        "the labels and is therefore part of A"),
        "candidate_family": [c["name"] for c in CANDIDATES],
        "n_candidates": len(CANDIDATES),
        "inner_selection_rule": "highest mean inner-validation participant AUROC",
        "observed": {
            "selected_model": obs["selected_by_inner"],
            "T_selection_aware": round(obs["T_selection_aware"], 4),
            "T_fixed": round(obs["T_fixed"], 4),
            "T_oracle_max": round(obs["T_oracle_max"], 4),
            "per_candidate_outer": {k: round(v, 4)
                                    for k, v in obs["per_candidate_outer"].items()},
        },
        "fixed_pipeline_null": mc_p(fix_null, obs["T_fixed"]),
        "selection_aware_null": mc_p(sel_null, obs["T_selection_aware"]),
        "selection_optimism": {
            "null_mean_shift": round(float(sel_null.mean() - fix_null.mean()), 4),
            "null_p95_shift": round(float(np.percentile(sel_null, 95)
                                          - np.percentile(fix_null, 95)), 4),
            "null_max_shift": round(float(sel_null.max() - fix_null.max()), 4),
            "reading": ("a positive shift is the optimism that selecting among "
                        f"{len(CANDIDATES)} candidates adds under the null; a fixed-"
                        "pipeline null does not account for it"),
        },
        "selected_model_frequency_under_null": {k: round(float(v), 4)
                                                for k, v in picks.head(8).items()},
        "data_source": f"REAL:{ws.name}",
    }
    out = Path(a.out) if a.out else ws / "results" / "selection_aware_permutation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("observed", "fixed_pipeline_null", "selection_aware_null",
                       "selection_optimism")}, indent=2))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
