#!/usr/bin/env python
"""Synthetic Depresjon-shaped workspace for testing the permutation engines.

SYNTHETIC DATA. Nothing here is a measurement. It exists because the Depresjon
features are not available locally, and the exact-speedup checks need a workspace
with the same shape and column names.

Shape: 55 participants (23 positive), 8-30 windows per participant summing to
788, and the 86 feature names in configs/depresjon_feature_columns.json.
Each base signal has a participant random intercept, AR(1) noise across a
person's windows, and a label shift of `effect` on four signals. Relative
features are computed from the person's own past windows, so the first window
of each person has NaNs, and 0.5% of base cells are set to NaN at random, so
the imputer has work to do.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from wearabledata.utils.seeds import rng

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLUMNS = ROOT / "configs" / "depresjon_feature_columns.json"
EFFECT_SIGNALS = 4          # label shift applies to the first four base signals
NAN_RATE = 0.005            # random missing cells in base features


def window_counts(n: int, total: int, lo: int, hi: int, g: np.random.Generator) -> np.ndarray:
    """Integers in [lo, hi] with a right-skewed shape and an exact sum."""
    if not n * lo <= total <= n * hi:
        raise ValueError("total windows not reachable within [lo, hi]")
    counts = np.clip(lo + g.negative_binomial(2, 2 / (2 + (total / n - lo)), n), lo, hi)
    while counts.sum() != total:
        i = g.integers(n)
        if counts.sum() < total and counts[i] < hi:
            counts[i] += 1
        elif counts.sum() > total and counts[i] > lo:
            counts[i] -= 1
    return counts


def generate(cols: list[str], n_participants: int = 55, n_positive: int = 23,
             n_windows: int = 788, min_windows: int = 8, max_windows: int = 30,
             effect: float = 1.0, seed_name: str = "synthetic_depresjon") -> pd.DataFrame:
    g = rng(f"synthetic_workspace:{seed_name}")
    signals = [c[: -len("__mean")] for c in cols
               if c.endswith("__mean") and not c.startswith(("pz__", "dev__", "chg__"))]
    k = len(signals)
    counts = window_counts(n_participants, n_windows, min_windows, max_windows, g)
    labels = np.zeros(n_participants, dtype=int)
    labels[g.permutation(n_participants)[:n_positive]] = 1
    shift = np.zeros(k)
    shift[:EFFECT_SIGNALS] = effect * np.array([1.0, -1.0] * (EFFECT_SIGNALS // 2))

    frames = []
    for p in range(n_participants):
        m = int(counts[p])
        intercept = g.normal(size=k)
        spread = g.normal(size=k)
        noise = np.empty((m, k))
        noise[0] = g.normal(scale=0.5, size=k)
        for w in range(1, m):   # overlapping windows: strongly autocorrelated
            noise[w] = 0.7 * noise[w - 1] + g.normal(scale=0.5 * np.sqrt(1 - 0.49), size=k)
        x = intercept + labels[p] * shift + noise
        rows: dict[str, np.ndarray] = {}
        exp_mean = np.cumsum(x, axis=0) / np.arange(1, m + 1)[:, None]
        exp_sd = np.array([x[: w + 1].std(axis=0, ddof=1) if w else np.full(k, np.nan)
                           for w in range(m)])
        cov = 1 - 0.3 * g.beta(1, 4, size=(m, k))
        for j, s in enumerate(signals):
            rows[f"{s}__mean"] = x[:, j]
            rows[f"{s}__sd"] = np.exp(0.3 * spread[j] + 0.2 * g.normal(size=m))
            rows[f"{s}__slope"] = np.r_[0.0, np.diff(x[:, j])] / 7 + g.normal(scale=0.05, size=m)
            dev = x[:, j] - exp_mean[:, j]
            rows[f"dev__{s}__mean"] = np.where(np.arange(m) == 0, np.nan, dev)
            rows[f"pz__{s}__mean"] = dev / exp_sd[:, j]
            rows[f"chg__{s}__mean"] = np.r_[np.nan, np.diff(x[:, j])]
            rows[f"cov__{s}"] = cov[:, j]
        wc = cov.mean(axis=1)
        rows["window_coverage"] = wc
        rows["valid_days"] = np.clip(np.round(7 * wc), 5, 7)
        frame = pd.DataFrame(rows)
        frame.insert(0, "label", labels[p])
        frame.insert(0, "window_index", np.arange(m))
        frame.insert(0, "study_id", f"SYN{p:03d}")
        frames.append(frame)

    feat = pd.concat(frames, ignore_index=True)
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        raise ValueError(f"generator does not produce columns: {missing}")
    base = [f"{s}__{a}" for s in signals for a in ("mean", "sd", "slope")]
    holes = g.random((len(feat), len(base))) < NAN_RATE
    feat[base] = feat[base].mask(holes)
    return feat[["study_id", "window_index", "label", *cols]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "synthetic_depresjon"))
    ap.add_argument("--columns-from", default=str(DEFAULT_COLUMNS))
    ap.add_argument("--effect", type=float, default=1.0,
                    help="label shift on four base signals, in intercept-SD units; 0 is the null")
    ap.add_argument("--participants", type=int, default=55)
    ap.add_argument("--positives", type=int, default=23)
    ap.add_argument("--windows", type=int, default=788)
    ap.add_argument("--seed-name", default="synthetic_depresjon")
    a = ap.parse_args()

    cols = json.loads(Path(a.columns_from).read_text())
    feat = generate(cols, a.participants, a.positives, a.windows, effect=a.effect,
                    seed_name=a.seed_name)
    interim = Path(a.out) / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    feat.to_csv(interim / "features.csv", index=False)
    (interim / "feature_columns.json").write_text(json.dumps(cols, indent=2))
    per = feat.groupby("study_id")
    meta = {
        "SYNTHETIC": True,
        "note": "generated by scripts/make_synthetic_workspace.py; not a measurement",
        "effect": a.effect, "seed_name": a.seed_name,
        "columns_from": Path(a.columns_from).resolve().relative_to(ROOT).as_posix()
        if Path(a.columns_from).resolve().is_relative_to(ROOT) else a.columns_from,
        "n_participants": int(per.ngroups),
        "n_positive": int(per["label"].first().sum()),
        "n_windows": int(len(feat)),
        "windows_per_participant": {"min": int(per.size().min()), "max": int(per.size().max()),
                                    "median": float(per.size().median())},
        "n_feature_columns": len(cols),
        "n_nan_cells": int(feat[cols].isna().sum().sum()),
    }
    (interim / "SYNTHETIC.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
