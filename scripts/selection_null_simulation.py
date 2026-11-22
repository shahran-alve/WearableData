#!/usr/bin/env python
"""Simulation: does repeating model selection inside the null matter?

Under the null, the number of candidate pipelines K is varied and two reference
distributions are compared for the same observed statistic:

  frozen    the pipeline selected on the observed labels is held fixed, and only
            that pipeline is refitted under each permutation.
  repeated  the whole candidate search is rerun inside every permutation.

If selection had no effect, the two nulls would coincide. Reported per K: the
Type-I error under each null and the shift in the upper tail.
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
from simulation_study import _folds, simulate, to_participant
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
# Candidate family: regularisation strength crossed with feature subset. All are
# reasonable a priori choices, which is the situation the test must handle.
GRID = [(C, frac) for C in (0.03, 0.3, 3.0, 30.0) for frac in (1.0, 0.5, 0.75, 0.25)]


def _fit_eval(df, y, feats, folds, C):
    oof = np.full(len(df), np.nan)
    for te in folds:
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", LogisticRegression(C=C, max_iter=400,
                                                 class_weight="balanced"))])
        m.fit(df.loc[tr, feats], y[tr])
        oof[te] = m.predict_proba(df.loc[te, feats])[:, 1]
    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    p, ya = to_participant(df.loc[ok], oof[ok], y[ok])
    return np.nan if len(np.unique(ya)) < 2 else float(roc_auc_score(ya, p))


def evaluate(df, y, K, rng):
    """Return (best statistic over K candidates, index of the winner)."""
    allf = [c for c in df.columns if c.startswith("f")]
    folds = _folds(df, rng)
    vals = []
    for C, frac in GRID[:K]:
        feats = allf[:max(2, int(round(frac * len(allf))))]
        vals.append(_fit_eval(df, y, feats, folds, C))
    vals = np.array(vals, dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan, -1
    j = int(np.nanargmax(vals))
    return float(vals[j]), j


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", type=int, default=40)
    ap.add_argument("--K", type=int, nargs="*", default=[1, 4, 8, 16])
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--permutations", type=int, default=49)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "simulation"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "selection_null_replicates.csv"
    done = {}
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        done = {(int(r["K"]), int(r["seed"])): dict(r) for _, r in prev.iterrows()}

    start = time.perf_counter()
    for K in a.K:
        for s in range(a.replicates):
            if (K, s) in done:
                continue
            if a.time_budget and time.perf_counter() - start > a.time_budget:
                break
            seed = 7000 + 100 * K + s
            rng = np.random.default_rng(seed)
            df = simulate(a.participants, 0.0, rng)      # delta = 0: pure null
            y = df["label"].to_numpy()
            obs, win = evaluate(df, y, K, np.random.default_rng(seed + 1))
            if not np.isfinite(obs):
                continue
            allf = [c for c in df.columns if c.startswith("f")]
            C_w, frac_w = GRID[win]
            feats_w = allf[:max(2, int(round(frac_w * len(allf))))]

            per = df.groupby("participant")["label"].first()
            ids, labs = per.index.to_numpy(), per.to_numpy()
            gp = np.random.default_rng(seed + 2)
            ge_frozen = ge_repeat = 0
            null_f, null_r = [], []
            for _ in range(a.permutations):
                mp = dict(zip(ids, gp.permutation(labs), strict=True))
                yp = df["participant"].map(mp).to_numpy().astype(int)
                fo = _folds(df, np.random.default_rng(seed + 1))
                vf = _fit_eval(df, yp, feats_w, fo, C_w)       # frozen winner
                vr, _ = evaluate(df, yp, K, np.random.default_rng(seed + 1))  # repeat search
                if np.isfinite(vf):
                    null_f.append(vf)
                    ge_frozen += int(vf >= obs)
                if np.isfinite(vr):
                    null_r.append(vr)
                    ge_repeat += int(vr >= obs)
            if not null_f or not null_r:
                continue
            done[(K, s)] = {
                "K": K, "seed": s, "observed": obs,
                "p_frozen": (1 + ge_frozen) / (len(null_f) + 1),
                "p_repeated": (1 + ge_repeat) / (len(null_r) + 1),
                "null_mean_frozen": float(np.mean(null_f)),
                "null_mean_repeated": float(np.mean(null_r)),
                "null_p95_frozen": float(np.percentile(null_f, 95)),
                "null_p95_repeated": float(np.percentile(null_r, 95)),
                "data_source": "SIMULATION"}
            pd.DataFrame(list(done.values())).to_csv(ckpt, index=False)

    df = pd.DataFrame(list(done.values()))
    if df.empty:
        print("no replicates yet")
        return 0
    rows = []
    for K, g in df.groupby("K"):
        rec = {"K": int(K), "n_replicates": int(len(g))}
        for lab in ("frozen", "repeated"):
            rej = (g[f"p_{lab}"] <= ALPHA)
            lo, hi = stats.binomtest(int(rej.sum()), len(g)).proportion_ci(0.95)
            rec[f"type1_{lab}"] = round(float(rej.mean()), 4)
            rec[f"type1_{lab}_ci"] = [round(float(lo), 3), round(float(hi), 3)]
            rec[f"null_mean_{lab}"] = round(float(g[f"null_mean_{lab}"].mean()), 4)
            rec[f"null_p95_{lab}"] = round(float(g[f"null_p95_{lab}"].mean()), 4)
        rec["p95_shift"] = round(rec["null_p95_repeated"] - rec["null_p95_frozen"], 4)
        rows.append(rec)
    summ = pd.DataFrame(rows).sort_values("K")
    summ.to_csv(out / "selection_null_summary.csv", index=False)
    (out / "selection_null_summary.json").write_text(json.dumps(
        {"alpha": ALPHA, "delta": 0.0, "rows": rows,
         "reading": ("frozen = permute after choosing a model; repeated = rerun the "
                     "search inside every permutation. Divergence with K is the "
                     "selection effect."),
         "data_source": "SIMULATION"}, indent=2))
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
