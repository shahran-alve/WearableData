"""Participant-safe temporal windowing.

Two invariants are enforced by construction, not by convention:
  * a window is built only from days strictly at or before its cutoff;
  * a window is emitted only if modality coverage clears the configured floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    history_days: int = 7
    stride_days: int = 1
    min_coverage: float = 0.5
    min_valid_days: int = 4


def _agg_block(block: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in feature_cols:
        s = pd.to_numeric(block[c], errors="coerce")
        out[f"{c}__mean"] = float(s.mean()) if s.notna().any() else np.nan
        out[f"{c}__sd"] = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
        out[f"{c}__slope"] = _slope(s)
        out[f"cov__{c}"] = float(s.notna().mean())
    return out


def _slope(s: pd.Series) -> float:
    y = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    return float(np.polyfit(x, y[ok], 1)[0])


def build_windows(panel: pd.DataFrame, feature_cols: list[str], spec: WindowSpec,
                  group_col: str = "study_id", time_col: str = "day_index",
                  label_col: str = "label") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Emit one row per (participant, cutoff day) using only past-or-present days."""
    records, dropped = [], {"below_coverage": 0, "below_valid_days": 0, "short_history": 0}
    for gid, sub in panel.groupby(group_col, sort=True):
        sub = sub.sort_values(time_col)
        days = sub[time_col].to_numpy()
        for cutoff in range(int(days.min()) + spec.history_days - 1,
                            int(days.max()) + 1, spec.stride_days):
            lo = cutoff - spec.history_days + 1
            block = sub[(sub[time_col] >= lo) & (sub[time_col] <= cutoff)]
            if len(block) < spec.history_days:
                dropped["short_history"] += 1
                continue
            valid_days = int(block[feature_cols].notna().any(axis=1).sum())
            if valid_days < spec.min_valid_days:
                dropped["below_valid_days"] += 1
                continue
            coverage = float(block[feature_cols].notna().to_numpy().mean())
            if coverage < spec.min_coverage:
                dropped["below_coverage"] += 1
                continue
            rec: dict[str, Any] = {group_col: gid, "cutoff_day": int(cutoff),
                                   "window_coverage": coverage, "valid_days": valid_days}
            rec.update(_agg_block(block, feature_cols))
            if label_col in sub.columns:
                # Label attaches to the participant-window; it is never an input.
                rec[label_col] = int(block[label_col].iloc[-1])
            records.append(rec)
    out = pd.DataFrame(records)
    report = {
        "n_windows": int(len(out)),
        "n_participants": int(out[group_col].nunique()) if len(out) else 0,
        "dropped": dropped,
        "spec": spec.__dict__,
        "windows_per_participant": (out.groupby(group_col).size().describe().to_dict()
                                    if len(out) else {}),
    }
    return out, report
