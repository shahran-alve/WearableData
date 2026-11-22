#!/usr/bin/env python
"""Participant-level evaluation and a fixed-pipeline permutation test.

The outcome is defined per participant, so the evaluation produces one score per
held-out participant. Out-of-fold window probabilities are averaged into a
participant score, and every reported metric is computed over participants.
Pooling windows instead would weight a person with 40 windows four times as much
as a person with 10.

The primary statistic is fixed in advance: participant-level out-of-fold AUROC of
a histogram gradient-boosting pipeline. For each permutation the participant
labels are shuffled and the whole pipeline is rerun (imputation, scaling, model
fit, aggregation). The p-value is the plus-one Monte Carlo estimate

    p = (1 + #{permuted statistics >= observed}) / (B + 1)

reported with a binomial interval for the Monte Carlo error.

Because the primary statistic comes from one fixed pipeline, no model selection
is repeated inside the permutation. Other models are reported descriptively.
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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from wearabledata.evaluation.statistics import grouped_bootstrap_ci
from wearabledata.features.splits import mask_for, nested_grouped_cv
from wearabledata.utils.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]

# ---- pre-specified primary pipeline (frozen before any result was inspected) --
PRIMARY = dict(learning_rate=0.1, max_leaf_nodes=31, max_iter=200)
AGGREGATION = "mean"          # participant score = mean of that person's window probabilities


def primary_pipeline(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", HistGradientBoostingClassifier(random_state=seed, early_stopping=False,
                                               **PRIMARY)),
    ])


def aggregate(scores: np.ndarray, groups: np.ndarray, labels: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window scores -> one score per participant. Labels are constant within person."""
    df = pd.DataFrame({"g": groups, "p": scores, "y": labels}).dropna(subset=["p"])
    agg = df.groupby("g").agg(p=("p", AGGREGATION), y=("y", "first"),
                              n_windows=("p", "size"))
    return (agg["p"].to_numpy(), agg["y"].to_numpy().astype(int),
            agg.index.to_numpy())


def make_folds(feat: pd.DataFrame, y: np.ndarray, n_folds: int, seed_name: str) -> list[dict]:
    """Grouped-stratified folds built from the supplied labels.

    Stratification reads the labels, so fold construction belongs inside the
    analysis algorithm A. Reusing folds stratified on the observed labels while
    permuting labels mis-stratifies every permuted fit and biases the null.
    """
    tmp = feat[["study_id"]].copy()
    tmp["label"] = y
    return nested_grouped_cv(tmp, outer_folds=n_folds, seed_name=seed_name)


def run_cv(feat: pd.DataFrame, cols: list[str], y: np.ndarray, folds: list[dict],
           seed: int = 0, X: np.ndarray | None = None,
           masks: list[dict] | None = None) -> np.ndarray:
    """Out-of-fold window probabilities.

    `X` and `masks` let a caller hoist the DataFrame-to-array conversion and the
    fold masks out of the permutation loop. Repeated `feat.loc[mask, cols]`
    slicing copies the whole matrix each time and dominated the runtime.
    """
    Xa = feat[cols].to_numpy(dtype=float) if X is None else X
    fm = masks or [{"tr": mask_for(feat, f["train"] + f["validation"]),
                    "te": mask_for(feat, f["test"])} for f in folds]
    oof = np.full(len(feat), np.nan)
    for m_ in fm:
        tr, te = m_["tr"], m_["te"]
        if len(np.unique(y[tr])) < 2:
            continue
        model = primary_pipeline(seed)
        model.fit(Xa[tr], y[tr])
        oof[te] = model.predict_proba(Xa[te])[:, 1]
    return oof


def participant_metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    out = {"n_participants": int(len(y)), "n_positive": int(y.sum()),
           "n_negative": int((1 - y).sum()), "prevalence": round(float(y.mean()), 4)}
    if len(np.unique(y)) < 2:
        return {**out, "status": "SINGLE_CLASS"}
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update({
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auprc_baseline": round(float(y.mean()), 4),
        # Probability quality, primary per E10; ECE is appendix-only at this n.
        "brier": round(float(brier_score_loss(y, p)), 4),
        "brier_baseline_prevalence": round(float(np.mean((y - y.mean()) ** 2)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, yhat)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else float("nan"), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else float("nan"), 4),
        "macro_f1": round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    })
    return out


def choose_threshold(feat: pd.DataFrame, cols: list[str], y: np.ndarray,
                     folds: list[dict], seed: int = 0) -> float:
    """Threshold from inner validation participants only -- never from test folds."""
    ps, ys = [], []
    for f in folds:
        tr = mask_for(feat, f["train"])
        va = mask_for(feat, f["validation"])
        if len(np.unique(y[tr])) < 2 or va.sum() == 0:
            continue
        m = primary_pipeline(seed)
        m.fit(feat.loc[tr, cols], y[tr])
        pv = m.predict_proba(feat.loc[va, cols])[:, 1]
        pa, ya, _ = aggregate(pv, feat.loc[va, "study_id"].to_numpy(), y[va])
        ps.append(pa)
        ys.append(ya)
    if not ps:
        return 0.5
    p, yv = np.concatenate(ps), np.concatenate(ys)
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        s = f1_score(yv, (p >= t).astype(int), average="macro", zero_division=0)
        if s > best:
            best, best_t = s, float(t)
    return best_t


def permutation_test(feat: pd.DataFrame, cols: list[str], y: np.ndarray,
                     folds: list[dict], observed: float, B: int, seed_name: str,
                     checkpoint: Path | None = None,
                     time_budget_s: float | None = None,
                     n_folds: int = 5, fold_seed_name: str = "cv") -> dict:
    """Shuffle the participant->label map; rerun the whole pipeline each time.

    Permutations are checkpointed to disk. Each permutation index is derived
    from a fixed seed, so resuming appends new indices rather than repeating
    completed ones, and the accumulated null is identical to an uninterrupted
    run of the same length.
    """
    per = pd.DataFrame({"g": feat["study_id"], "y": y}).groupby("g")["y"].first()
    ids = per.index.to_numpy()
    labels = per.to_numpy()
    base = derive_seed(seed_name)
    Xa = feat[cols].to_numpy(dtype=float)

    done: dict[int, float] = {}
    if checkpoint is not None and checkpoint.exists():
        prev = pd.read_csv(checkpoint)
        done = dict(zip(prev["index"].astype(int), prev["auroc"].astype(float),
                        strict=True))
    start = time.perf_counter()
    stopped_early = False
    for i in range(B):
        if i in done:
            continue
        if time_budget_s is not None and time.perf_counter() - start > time_budget_s:
            stopped_early = True
            break
        g = np.random.default_rng(base + i)
        mapping = dict(zip(ids, g.permutation(labels), strict=True))
        yp = feat["study_id"].map(mapping).to_numpy().astype(int)
        # Regenerate the label-dependent design from the permuted labels.
        folds_b = make_folds(feat, yp, n_folds, fold_seed_name)
        masks_b = [{"tr": mask_for(feat, f["train"] + f["validation"]),
                    "te": mask_for(feat, f["test"])} for f in folds_b]
        oof = run_cv(feat, cols, yp, folds_b, X=Xa, masks=masks_b)
        pa, ya, _ = aggregate(oof, feat["study_id"].to_numpy(), yp)
        if len(np.unique(ya)) < 2:
            continue
        done[i] = float(roc_auc_score(ya, pa))
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"index": list(done), "auroc": list(done.values())}).to_csv(
                checkpoint, index=False)
    null = np.asarray(sorted(done.values()))
    if len(null) == 0:
        return {"status": "NO_VALID_PERMUTATIONS"}
    n_ge = int((null >= observed).sum())
    B_eff = len(null)
    p = (1 + n_ge) / (B_eff + 1)
    lo, hi = stats.binomtest(n_ge, B_eff).proportion_ci(confidence_level=0.95)
    return {
        "test": ("participant-level permutation; grouped-stratified folds regenerated "
                 "from each permuted labelling; full pipeline rerun"),
        "statistic": "participant-level out-of-fold AUROC (fixed HGB pipeline)",
        "observed": round(observed, 4),
        "B_requested": B, "B_effective": B_eff,
        "n_permuted_ge_observed": n_ge,
        "p_value_plus_one": round(p, 5),
        "p_value_monte_carlo_ci": [round(float((1 + lo * B_eff) / (B_eff + 1)), 5),
                                   round(float((1 + hi * B_eff) / (B_eff + 1)), 5)],
        "smallest_attainable_p": round(1 / (B_eff + 1), 5),
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(ddof=1)), 4),
        "null_quantiles": {q: round(float(np.percentile(null, 100 * q)), 4)
                           for q in (0.025, 0.5, 0.95, 0.975)},
        "null_max": round(float(null.max()), 4),
        "null_note": ("quantiles are empirical percentiles of the permutation "
                      "distribution and carry Monte Carlo error at this B; they are "
                      "descriptive, and inference uses the p-value above"),
        "completed": not stopped_early and len(null) >= B,
        "stopped_on_time_budget": stopped_early,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT / "reports" / "real_depresjon"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--subset-participants", default=None,
                    help="JSON list of study_ids: restrict the cohort (matched analyses)")
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds; checkpoint and exit so the run can be resumed")
    ap.add_argument("--skip-observed", action="store_true",
                    help="only extend the permutation checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ws = Path(a.workspace)
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat.columns]
    if a.subset_participants:
        keep = set(json.loads(Path(a.subset_participants).read_text()))
        feat = feat[feat["study_id"].isin(keep)].reset_index(drop=True)
    y = feat["label"].to_numpy()
    groups = feat["study_id"].to_numpy()

    folds = make_folds(feat, y, a.folds, f"cv:{ws.name}:{a.tag}")
    threshold = choose_threshold(feat, cols, y, folds)

    # ---- observed statistic, and its stability across training seeds -----
    by_seed = []
    for s in a.seeds:
        oof = run_cv(feat, cols, y, folds, seed=s)
        pa, ya, pid = aggregate(oof, groups, y)
        by_seed.append({"seed": s, **participant_metrics(pa, ya, threshold)})
        if s == a.seeds[0]:
            p_primary, y_primary, ids_primary = pa, ya, pid
    seed_df = pd.DataFrame(by_seed)
    observed = float(seed_df.loc[seed_df.seed == a.seeds[0], "auroc"].iloc[0])

    # ---- thresholded-metric intervals (stratified participant bootstrap) --
    def _thresh_metrics(yv, pv, thr):
        yhat = (pv >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, yhat, labels=[0, 1]).ravel()
        return {"balanced_accuracy": float(balanced_accuracy_score(yv, yhat)),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
                "n_true_positives_detected": int(tp)}

    gboot = np.random.default_rng(derive_seed(f"threshboot:{a.tag}"))
    pos = np.flatnonzero(y_primary == 1)
    neg = np.flatnonzero(y_primary == 0)
    draws: dict[str, list[float]] = {}
    for _ in range(a.bootstrap):
        # Stratified: resample positives and negatives separately so every
        # replicate contains both classes; percentile intervals.
        idx = np.r_[gboot.choice(pos, len(pos), replace=True),
                    gboot.choice(neg, len(neg), replace=True)]
        for k, v in _thresh_metrics(y_primary[idx], p_primary[idx], threshold).items():
            if k != "n_true_positives_detected":
                draws.setdefault(k, []).append(v)
    thresh_ci = {k: [round(float(np.nanpercentile(v, 2.5)), 4),
                     round(float(np.nanpercentile(v, 97.5)), 4)]
                 for k, v in draws.items()}
    thresh_point = _thresh_metrics(y_primary, p_primary, threshold)

    # ---- participant bootstrap (descriptive) -----------------------------
    ci_auc = grouped_bootstrap_ci(y_primary, p_primary, ids_primary, roc_auc_score,
                                  n_resamples=a.bootstrap, seed_name=f"pb:{a.tag}:auc")
    ci_pr = grouped_bootstrap_ci(y_primary, p_primary, ids_primary,
                                 average_precision_score, n_resamples=a.bootstrap,
                                 seed_name=f"pb:{a.tag}:pr")
    discard = 1 - ci_auc["n_valid_resamples"] / a.bootstrap

    # ---- the test --------------------------------------------------------
    ckpt = ws / "results" / f"permutation_null_{a.tag}_checkpoint.csv"
    test = permutation_test(feat, cols, y, folds, observed, a.permutations,
                            f"perm:{ws.name}:{a.tag}", checkpoint=ckpt,
                            time_budget_s=a.time_budget, n_folds=a.folds,
                            fold_seed_name=f"cv:{ws.name}:{a.tag}")

    report = {
        "workspace": str(ws), "tag": a.tag,
        "unit_of_analysis": "participant",
        "aggregation_rule": AGGREGATION,
        "primary_pipeline": PRIMARY,
        "n_participants": int(pd.Series(groups).nunique()),
        "n_windows": int(len(feat)),
        "windows_per_participant": {
            "median": float(pd.Series(groups).value_counts().median()),
            "min": int(pd.Series(groups).value_counts().min()),
            "max": int(pd.Series(groups).value_counts().max())},
        "threshold_source": "inner validation participants only",
        "participant_metrics": by_seed[0],
        "seed_stability": {"seeds": a.seeds,
                           "auroc_mean": round(float(seed_df["auroc"].mean()), 4),
                           "auroc_sd": round(float(seed_df["auroc"].std(ddof=1)), 4)
                           if len(seed_df) > 1 else 0.0},
        "thresholded_metrics": {
            "threshold": round(float(threshold), 4),
            "point": {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in thresh_point.items()},
            "ci_95_percentile": thresh_ci,
            "interval_construction": ("stratified participant bootstrap, positives "
                                      "and negatives resampled separately, percentile "
                                      "interval"),
            "caution": ("the threshold is selected on very small inner-validation "
                        "sets and is not stable; these intervals do not include "
                        "threshold-selection variability"),
        },
        "participant_bootstrap": {
            "auroc": [round(ci_auc["ci_low"], 4), round(ci_auc["ci_high"], 4)],
            "auprc": [round(ci_pr["ci_low"], 4), round(ci_pr["ci_high"], 4)],
            "n_resamples": a.bootstrap,
            "discarded_single_class_resample_rate": round(float(discard), 4),
            "role": "descriptive only; inference uses the permutation p-value"},
        "permutation_test": test,
        "data_source": f"REAL:{ws.name}",
    }

    out = Path(a.out) if a.out else ws / "results" / f"participant_level_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    pd.DataFrame({"study_id": ids_primary, "label": y_primary,
                  "participant_score": p_primary}).to_csv(
        out.with_name(out.stem + "_predictions.csv"), index=False)
    print(json.dumps({k: report[k] for k in
                      ("n_participants", "participant_metrics", "seed_stability",
                       "participant_bootstrap", "permutation_test")}, indent=2))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
