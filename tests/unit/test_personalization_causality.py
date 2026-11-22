import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from wearabledata.features.personalization import (
    apply_personalization,
    personal_zscore,
    personalization_columns,
)


def _wins(n=4, k=12):
    return pd.DataFrame([{"study_id": f"P{i}", "cutoff_day": t, "v": float(t + 10 * i)}
                         for i in range(n) for t in range(k)])


def test_personal_zscore_ignores_current_and_future_values():
    df = _wins()
    z = personal_zscore(df, ["v"], min_history=2)
    joined = pd.concat([df.reset_index(drop=True), z.reset_index(drop=True)], axis=1)
    # First two windows per participant have insufficient prior history -> NaN.
    first = joined.groupby("study_id").head(2)
    assert first["pz__v"].isna().all()


def test_perturbing_the_future_does_not_change_the_past():
    df = _wins()
    z1 = personal_zscore(df, ["v"], min_history=2).reset_index(drop=True)
    df2 = df.copy()
    df2.loc[df2["cutoff_day"] >= 8, "v"] = 9999.0
    z2 = personal_zscore(df2, ["v"], min_history=2).reset_index(drop=True)
    early = df["cutoff_day"].to_numpy() < 8
    a, b = z1.loc[early, "pz__v"], z2.loc[early, "pz__v"]
    assert np.allclose(a.fillna(-1), b.fillna(-1))


def test_no_cross_participant_information():
    df = _wins()
    z1 = personal_zscore(df, ["v"], min_history=2).reset_index(drop=True)
    df2 = df.copy()
    df2.loc[df2["study_id"] == "P3", "v"] *= 1000
    z2 = personal_zscore(df2, ["v"], min_history=2).reset_index(drop=True)
    keep = (df["study_id"] != "P3").to_numpy()
    assert np.allclose(z1.loc[keep, "pz__v"].fillna(-1), z2.loc[keep, "pz__v"].fillna(-1))


def test_apply_personalization_adds_expected_columns(windows):
    cols = [c for c in windows.columns if c.endswith("__mean")][:3]
    out = apply_personalization(windows, cols)
    added = personalization_columns(out)
    assert len(added) == 3 * len(cols)
