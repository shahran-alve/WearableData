#!/usr/bin/env python
"""Simulation study: Type-I error and power of four inference procedures.

Synthetic participants each contribute a variable number of correlated windows.
A participant-level effect of size `delta` shifts the feature means of positive
participants, and `delta = 0` is the null. A large participant-level random
intercept makes windows from one person much more alike than windows from
different people.

Procedures compared on the same simulated data:

  window_naive        window-level AUROC against 0.5 by a Mann-Whitney test that
                      treats every window as independent.
  window_bootstrap    window-level AUROC with a window bootstrap; reject when the
                      95% interval excludes 0.5.
  participant_boot    participant-level AUROC with a participant bootstrap; reject
                      when the 95% interval excludes 0.5.
  permutation         participant-level AUROC with the participant-label
                      permutation test.

At `delta = 0` the rejection rate is the Type-I error, which should be near 0.05.
Above zero it is the power.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05


def simulate(n_participants: int, delta: float, rng: np.random.Generator,
             n_features: int = 12, prevalence: float = 0.42,
             windows_lo: int = 8, windows_hi: int = 30) -> pd.DataFrame:
    """Grouped longitudinal data with a known participant-level effect."""
    n_pos = max(3, int(round(prevalence * n_participants)))
    labels = np.r_[np.ones(n_pos), np.zeros(n_participants - n_pos)].astype(int)
    rng.shuffle(labels)
    rows = []
    for i, y in enumerate(labels):
        m = int(rng.integers(windows_lo, windows_hi + 1))
        # Participant random intercept: large relative to the effect, which is
        # what makes window-level pooling so misleading.
        intercept = rng.normal(0, 1.0, size=n_features)
        signal = np.zeros(n_features)
        signal[:4] = delta * y          # effect concentrated in four features
        for _ in range(m):
            x = intercept + signal + rng.normal(0, 0.6, size=n_features)
            rows.append({"participant": f"P{i:04d}", "label": int(y),
                         **{f"f{k}": x[k] for k in range(n_features)}})
    return pd.DataFrame(rows)


def _folds(df: pd.DataFrame, rng: np.random.Generator, k: int = 5) -> list[np.ndarray]:
    """Grouped stratified folds built from the supplied labels."""
    per = df.groupby("participant")["label"].first()
    assign = {}
    for lab in (0, 1):
        ids = per[per == lab].index.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        for j, pid in enumerate(ids):
            assign[pid] = j % k
    return [df["participant"].map(assign).to_numpy() == f for f in range(k)]


def _estimator(kind: str):
    if kind == "tree":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=60, min_samples_leaf=3,
                                      class_weight="balanced_subsample",
                                      n_jobs=1, random_state=0)
    return LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")


def oof_scores(df: pd.DataFrame, y: np.ndarray, rng: np.random.Generator,
               learner: str = "linear") -> np.ndarray:
    feats = [c for c in df.columns if c.startswith("f")]
    oof = np.full(len(df), np.nan)
    for te in _folds(df, rng):
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        # Regularized logistic regression rather than a tree ensemble: the
        # simulation asks whether the *inference procedure* controls error, not
        # which learner is strongest, and a linear fit makes 10^4 model fits
        # affordable. The conclusion is about the test, not the classifier.
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", _estimator(learner))])
        m.fit(df.loc[tr, feats], y[tr])
        oof[te] = m.predict_proba(df.loc[te, feats])[:, 1]
    return oof


def to_participant(df: pd.DataFrame, scores: np.ndarray, y: np.ndarray):
    t = pd.DataFrame({"g": df["participant"], "p": scores, "y": y}).dropna(subset=["p"])
    agg = t.groupby("g").agg(p=("p", "mean"), y=("y", "first"))
    return agg["p"].to_numpy(), agg["y"].to_numpy().astype(int)


def _boot_ci(y: np.ndarray, p: np.ndarray, rng: np.random.Generator,
             n: int = 400) -> tuple[float, float]:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        if len(np.unique(y[idx])) > 1:
            draws.append(roc_auc_score(y[idx], p[idx]))
    if len(draws) < 30:
        return (np.nan, np.nan)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def one_replicate(n_participants: int, delta: float, seed: int, B: int,
                  learner: str = "linear", windows_lo: int = 8,
                  windows_hi: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    df = simulate(n_participants, delta, rng, windows_lo=windows_lo,
                  windows_hi=windows_hi)
    y = df["label"].to_numpy()
    scores = oof_scores(df, y, np.random.default_rng(seed + 7), learner)
    ok = ~np.isnan(scores)
    if len(np.unique(y[ok])) < 2:
        return {}
    out = {"n_participants": n_participants, "delta": delta, "seed": seed,
           "learner": learner, "n_windows": int(len(df))}

    # --- window-level, treated as independent -----------------------------
    w_auc = float(roc_auc_score(y[ok], scores[ok]))
    pos, neg = scores[ok][y[ok] == 1], scores[ok][y[ok] == 0]
    u_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    out["window_auroc"] = w_auc
    out["reject_window_naive"] = bool(u_p <= ALPHA)

    rb = np.random.default_rng(seed + 11)
    lo, hi = _boot_ci(y[ok], scores[ok], rb)
    out["reject_window_bootstrap"] = bool(np.isfinite(lo) and (lo > 0.5 or hi < 0.5))

    # --- participant level -------------------------------------------------
    p_part, y_part = to_participant(df.loc[ok], scores[ok], y[ok])
    obs = float(roc_auc_score(y_part, p_part))
    out["participant_auroc"] = obs
    lo, hi = _boot_ci(y_part, p_part, np.random.default_rng(seed + 13))
    out["reject_participant_bootstrap"] = bool(np.isfinite(lo) and (lo > 0.5 or hi < 0.5))

    # --- permutation test ---------------------------------------------------
    per = df.groupby("participant")["label"].first()
    ids, labs = per.index.to_numpy(), per.to_numpy()
    gp = np.random.default_rng(seed + 17)
    ge = 0
    for _ in range(B):
        mapping = dict(zip(ids, gp.permutation(labs), strict=True))
        yp = df["participant"].map(mapping).to_numpy().astype(int)
        sp = oof_scores(df, yp, np.random.default_rng(seed + 7), learner)
        okp = ~np.isnan(sp)
        if len(np.unique(yp[okp])) < 2:
            continue
        pp, yy = to_participant(df.loc[okp], sp[okp], yp[okp])
        if len(np.unique(yy)) > 1 and roc_auc_score(yy, pp) >= obs:
            ge += 1
    out["permutation_p"] = (1 + ge) / (B + 1)
    out["reject_permutation"] = bool(out["permutation_p"] <= ALPHA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", type=int, nargs="*", default=[40])
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.0, 0.3, 0.6])
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--learner", default="linear", choices=["linear", "tree"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "simulation"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "simulation_replicates.csv"
    done = {}
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        done = {(int(r["n_participants"]), float(r["delta"]), int(r["seed"]),
                 str(r.get("learner", "linear"))): dict(r)
                for _, r in prev.iterrows()}

    start = time.perf_counter()
    stopped = False
    for n in a.participants:
        for d in a.deltas:
            for s in range(a.replicates):
                seed = 1000 * s + int(round(d * 100)) + n
                if (n, d, seed, a.learner) in done:
                    continue
                if a.time_budget and time.perf_counter() - start > a.time_budget:
                    stopped = True
                    break
                r = one_replicate(n, d, seed, a.permutations, a.learner)
                if r:
                    done[(n, d, seed, a.learner)] = r
                    pd.DataFrame(list(done.values())).to_csv(ckpt, index=False)
            if stopped:
                break
        if stopped:
            break

    df = pd.DataFrame(list(done.values()))
    if df.empty:
        print("no replicates yet")
        return 0

    rows = []
    if "learner" not in df:
        df["learner"] = "linear"
    for (n, d, lr), g in df.groupby(["n_participants", "delta", "learner"]):
        rec = {"n_participants": int(n), "delta": float(d), "learner": lr,
               "n_replicates": int(len(g)),
               "truth": "null" if d == 0 else "signal",
               "mean_window_auroc": round(float(g["window_auroc"].mean()), 4),
               "mean_participant_auroc": round(float(g["participant_auroc"].mean()), 4)}
        for proc in ("window_naive", "window_bootstrap", "participant_bootstrap",
                     "permutation"):
            col = f"reject_{proc}"
            if col not in g:
                continue
            rate = float(g[col].mean())
            k, nn = int(g[col].sum()), len(g)
            lo, hi = stats.binomtest(k, nn).proportion_ci(confidence_level=0.95)
            rec[f"{proc}_rate"] = round(rate, 4)
            rec[f"{proc}_ci"] = [round(float(lo), 4), round(float(hi), 4)]
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values(["learner", "n_participants", "delta"])
    summary.to_csv(out / "simulation_summary.csv", index=False)
    (out / "simulation_summary.json").write_text(json.dumps({
        "alpha": ALPHA, "permutations_per_replicate": a.permutations,
        "design": ("participant-level latent effect on 4 of 12 features; "
                   "participant random intercept SD 1.0; window noise SD 0.6; "
                   "8-30 windows per participant"),
        "reading": ("at delta = 0 the rejection rate is Type-I error and should be "
                    "near 0.05; above zero it is power"),
        "rows": rows, "data_source": "SIMULATION"}, indent=2))
    print(summary.to_string(index=False))
    if stopped:
        print("\n(stopped on time budget; rerun to extend the checkpoint)")
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
