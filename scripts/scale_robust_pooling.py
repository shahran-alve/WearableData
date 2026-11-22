#!/usr/bin/env python
"""Scale-robust pooling for the participant-level statistic.

The concern is concrete: when different outer folds select different model
families, the probabilities they emit need not live on a common scale, and
pooling them before computing a single AUROC can create or destroy apparent
discrimination for reasons that have nothing to do with the data.

The fix used here is to rank-normalise \\emph{within each outer fold} before
pooling. Each participant's score is replaced by its within-fold quantile
position, so any strictly monotone per-fold rescaling of the model's output
leaves the statistic unchanged. Because AUROC depends only on the ordering of
scores, this preserves the estimand within a fold while removing the
cross-fold comparability assumption.

Both the observed statistic and its permutation null are recomputed under the
rank-pooled definition, so the comparison is like-for-like rather than a
post-hoc re-scoring of an existing null.
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
from participant_level_eval import aggregate, make_folds, primary_pipeline
from scipy import stats
from sklearn.metrics import roc_auc_score

from wearabledata.features.splits import mask_for
from wearabledata.utils.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]


def foldwise_rank_scores(feat: pd.DataFrame, cols: list[str], y: np.ndarray,
                         folds: list[dict], X: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray]:
    """Participant scores expressed as within-fold quantile ranks, then pooled."""
    groups = feat["study_id"].to_numpy()
    scores, labels, ids = [], [], []
    for f in folds:
        tr = mask_for(feat, f["train"] + f["validation"])
        te = mask_for(feat, f["test"])
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        m = primary_pipeline(0)
        m.fit(X[tr], y[tr])
        p_win = m.predict_proba(X[te])[:, 1]
        p_part, y_part, id_part = aggregate(p_win, groups[te], y[te])
        if len(p_part) < 2:
            continue
        # Within-fold quantile rank in (0, 1): invariant to any monotone rescaling
        # of that fold's probability output.
        r = stats.rankdata(p_part) / (len(p_part) + 1.0)
        scores.append(r)
        labels.append(y_part)
        ids.append(id_part)
    if not scores:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT / "reports" / "real_depresjon"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--tag", default="rankpooled")
    a = ap.parse_args()

    ws = Path(a.workspace)
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat.columns]
    y = feat["label"].to_numpy()
    X = feat[cols].to_numpy(dtype=float)

    folds = make_folds(feat, y, a.folds, f"cv:{ws.name}:primary")
    r, yr, _ = foldwise_rank_scores(feat, cols, y, folds, X)
    obs = float(roc_auc_score(yr, r))

    ckpt = ws / "results" / f"permutation_null_{a.tag}_checkpoint.csv"
    done: dict[int, float] = {}
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        done = dict(zip(prev["index"].astype(int), prev["auroc"].astype(float), strict=True))

    per = pd.DataFrame({"g": feat["study_id"], "y": y}).groupby("g")["y"].first()
    ids, labs = per.index.to_numpy(), per.to_numpy()
    base = derive_seed(f"rankperm:{ws.name}")
    start = time.perf_counter()
    for i in range(a.permutations):
        if i in done:
            continue
        if a.time_budget and time.perf_counter() - start > a.time_budget:
            break
        g = np.random.default_rng(base + i)
        mapping = dict(zip(ids, g.permutation(labs), strict=True))
        yp = feat["study_id"].map(mapping).to_numpy().astype(int)
        folds_b = make_folds(feat, yp, a.folds, f"cv:{ws.name}:primary")
        rp, yp_r, _ = foldwise_rank_scores(feat, cols, yp, folds_b, X)
        if len(rp) == 0 or len(np.unique(yp_r)) < 2:
            continue
        done[i] = float(roc_auc_score(yp_r, rp))
        pd.DataFrame({"index": list(done), "auroc": list(done.values())}).to_csv(
            ckpt, index=False)

    null = np.array(sorted(done.values()))
    out: dict = {
        "workspace": str(ws),
        "statistic": ("participant AUROC after within-fold quantile-rank "
                      "normalisation, then pooling"),
        "rationale": ("invariant to any strictly monotone per-fold rescaling, so "
                      "it does not assume probabilities are comparable across "
                      "outer folds that may select different model families"),
        "observed_rank_pooled": round(obs, 4),
        "n_participants": int(len(yr)),
        "data_source": f"REAL:{ws.name}",
    }
    if len(null):
        n_ge = int((null >= obs).sum())
        B = len(null)
        lo, hi = stats.binomtest(n_ge, B).proportion_ci(confidence_level=0.95)
        out.update({
            "B": B, "n_ge": n_ge,
            "p_value": round((1 + n_ge) / (B + 1), 5),
            "p_ci": [round(float((1 + lo * B) / (B + 1)), 5),
                     round(float((1 + hi * B) / (B + 1)), 5)],
            "smallest_attainable_p": round(1 / (B + 1), 5),
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std(ddof=1)), 4),
            "null_max": round(float(null.max()), 4)})
    (ws / "results" / f"scale_robust_{a.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
