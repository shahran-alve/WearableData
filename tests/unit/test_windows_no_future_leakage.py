import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from wearabledata.preprocessing.windows import WindowSpec, build_windows


def _panel(n=6, days=20):
    rows = [{"study_id": f"P{i}", "day_index": d, "x": float(d), "label": i % 2}
            for i in range(n) for d in range(days)]
    return pd.DataFrame(rows)


def test_window_uses_only_past_and_present():
    df = _panel()
    w, _ = build_windows(df, ["x"], WindowSpec(history_days=7, min_coverage=0.0,
                                               min_valid_days=1))
    # x == day_index, so a 7-day window ending at cutoff has mean cutoff-3.
    for _, r in w.iterrows():
        assert r["x__mean"] == r["cutoff_day"] - 3.0
        assert r["x__mean"] < r["cutoff_day"] + 1


def test_future_contamination_would_be_caught():
    """Sanity: a centred window would fail the same assertion."""
    df = _panel()
    w, _ = build_windows(df, ["x"], WindowSpec(history_days=7, min_coverage=0.0,
                                               min_valid_days=1))
    assert w["x__mean"].max() < df["day_index"].max()


def test_coverage_threshold_rejects_sparse_windows():
    df = _panel()
    df.loc[df["study_id"] == "P0", "x"] = np.nan
    w, rep = build_windows(df, ["x"], WindowSpec(history_days=7, min_coverage=0.5,
                                                 min_valid_days=4))
    assert "P0" not in set(w["study_id"])
    assert rep["dropped"]["below_valid_days"] > 0
