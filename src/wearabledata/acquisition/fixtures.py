"""Synthetic fixture generator for the workspace tests.

Fixture data only exercises code paths. Every artifact produced from it is
stamped `data_source: SYNTHETIC_FIXTURE`, and no reported result comes from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.seeds import rng

FIXTURE_STAMP = "SYNTHETIC_FIXTURE"


def make_daily_panel(n_participants: int = 30, days: int = 42, seed_name: str = "fixture",
                     missing_rate: float = 0.12, effect: float = 1.0,
                     shuffle_labels: bool = False) -> pd.DataFrame:
    """Participant x day panel with a weak, known signal and realistic missingness.

    The effect size is deliberately small (separable to roughly AUROC 0.7, not 1.0).
    A saturated fixture is a bad canary: with perfect separation a leakage bug or a
    broken personalization transform still yields a perfect score, so nothing is
    detected. `shuffle_labels=True` produces the permutation null used by the
    integration test, where any model scoring meaningfully above chance indicates
    leakage somewhere in the pipeline.
    """
    g = rng(seed_name)
    rows = []
    # Participant-level latent risk drives both label and behaviour, so a real
    # personalization effect exists and leakage tests are meaningful.
    risk = g.binomial(1, 0.35, size=n_participants)
    # Independent per-feature participant offsets. A single shared offset would make
    # the nuisance one-dimensional while the risk effect spans many features, which
    # is unrealistically easy; real between-person variation is high-dimensional.
    person_offset = g.normal(0, 1.0, size=(n_participants, 9))
    for i in range(n_participants):
        base_act = 6500 - 320 * effect * risk[i] + 1200 * person_offset[i, 0]
        base_sleep = 7.2 - 0.28 * effect * risk[i] + 0.75 * person_offset[i, 1]
        # `_i=i` binds the participant index at definition time; a bare closure over
        # the loop variable would be correct here but is a well-known footgun.
        def draw(base, risk_coef, person_coef, person_dim, noise_sd, _i=i):
            """One daily value: base + risk shift + participant offset + noise."""
            return float(base + risk_coef * effect * risk[_i]
                         + person_coef * person_offset[_i, person_dim]
                         + g.normal(0, noise_sd))

        for d in range(days):
            weekly = 0.15 * np.sin(2 * np.pi * d / 7.0)
            drift = 0.004 * effect * d * (-1 if risk[i] else 1)
            rows.append({
                "study_id": f"P{i:03d}",
                "day_index": d,
                "activity_total": max(0.0, base_act * (1 + weekly + drift) + g.normal(0, 900)),
                "sedentary_fraction": float(np.clip(
                    draw(0.62, 0.022, 0.055, 2, 0.075), 0, 1)),
                "sleep_duration": max(0.0, base_sleep + g.normal(0, 1.25)),
                "sleep_midpoint": draw(3.6, 0.20, 0.9, 3, 1.1),
                "unlock_count": max(0.0, draw(62, 5.5, 18, 4, 22)),
                "conversation_minutes": max(0.0, draw(95, -9.0, 26, 5, 30)),
                "radius_of_gyration": max(0.0, draw(3.1, -0.30, 1.0, 6, 1.2)),
                "location_entropy": max(0.0, draw(1.5, -0.12, 0.36, 7, 0.42)),
                "stress_lag_mean": float(np.clip(draw(2.4, 0.26, 0.7, 8, 0.85), 1, 5)),
                "label": int(risk[i]),
            })
    df = pd.DataFrame(rows)
    if shuffle_labels:
        perm = g.permutation(n_participants)
        remap = {f"P{i:03d}": int(risk[perm[i]]) for i in range(n_participants)}
        df["label"] = df["study_id"].map(remap)
    feature_cols = [c for c in df.columns if c not in ("study_id", "day_index", "label")]
    mask = g.random((len(df), len(feature_cols))) < missing_rate
    df.loc[:, feature_cols] = df[feature_cols].mask(pd.DataFrame(mask, columns=feature_cols,
                                                                 index=df.index))
    df.attrs["data_source"] = FIXTURE_STAMP
    return df
