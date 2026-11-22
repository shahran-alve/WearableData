"""Leakage-safe personalization transforms.

The rule that makes or breaks this project: for every participant -- including
validation and test participants -- a personal baseline statistic at cutoff t may
use only that participant's own windows with cutoff < t. Nothing is fitted across
participants here, so these transforms are legitimate for held-out people.
Global scaling and feature selection are a separate step fitted on train only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _expanding_shifted(g: pd.Series, fn: str, min_periods: int) -> pd.Series:
    r = g.shift(1).expanding(min_periods=min_periods)
    return getattr(r, fn)()


def personal_zscore(df: pd.DataFrame, cols: list[str], group_col: str = "study_id",
                    order_col: str = "cutoff_day", min_history: int = 2) -> pd.DataFrame:
    """z = (x_t - mean(x_{<t})) / sd(x_{<t}), strictly causal per participant."""
    df = df.sort_values([group_col, order_col])
    out = pd.DataFrame(index=df.index)
    for c in cols:
        grp = df.groupby(group_col)[c]
        mu = grp.transform(lambda s: _expanding_shifted(s, "mean", min_history))
        sd = grp.transform(lambda s: _expanding_shifted(s, "std", min_history))
        out[f"pz__{c}"] = (df[c] - mu) / (sd + EPS)
    return out


def deviation_from_rolling_median(df: pd.DataFrame, cols: list[str], window: int = 7,
                                  group_col: str = "study_id",
                                  order_col: str = "cutoff_day") -> pd.DataFrame:
    df = df.sort_values([group_col, order_col])
    out = pd.DataFrame(index=df.index)
    for c in cols:
        grp = df.groupby(group_col)[c]
        med = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).median())
        iqr = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=3)
            .apply(lambda w: np.nanpercentile(w, 75) - np.nanpercentile(w, 25), raw=True))
        out[f"dev__{c}"] = (df[c] - med) / (iqr + EPS)
    return out


def change_from_baseline(df: pd.DataFrame, cols: list[str], baseline_windows: int = 3,
                         group_col: str = "study_id",
                         order_col: str = "cutoff_day") -> pd.DataFrame:
    """Change relative to the participant's first `baseline_windows` observations."""
    df = df.sort_values([group_col, order_col])
    out = pd.DataFrame(index=df.index)
    base = df.groupby(group_col)[cols].transform(
        lambda s: s.head(baseline_windows).mean())
    rank = df.groupby(group_col).cumcount()
    for c in cols:
        delta = df[c] - base[c]
        # A window inside its own baseline period would use itself: suppress it.
        out[f"chg__{c}"] = delta.where(rank >= baseline_windows, np.nan)
    return out


def apply_personalization(df: pd.DataFrame, cols: list[str], group_col: str = "study_id",
                          order_col: str = "cutoff_day", min_history: int = 2,
                          hybrid: bool = True) -> pd.DataFrame:
    df = df.sort_values([group_col, order_col]).reset_index(drop=True)
    parts = [
        personal_zscore(df, cols, group_col, order_col, min_history),
        deviation_from_rolling_median(df, cols, 7, group_col, order_col),
        change_from_baseline(df, cols, 3, group_col, order_col),
    ]
    personal = pd.concat(parts, axis=1)
    return pd.concat([df, personal], axis=1) if hybrid else pd.concat(
        [df.drop(columns=cols), personal], axis=1)


def personalization_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("pz__", "dev__", "chg__"))]
