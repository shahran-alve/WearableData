"""Feature registry: names, domains, units, valid ranges, provenance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DOMAIN_PREFIXES = {
    "activity": ("activity_", "sedentary_", "active_bout"),
    "sleep": ("sleep_", "onset_", "wake_"),
    "mobility": ("radius_", "location_", "time_at_", "significant_"),
    "phone": ("unlock_", "screen_", "conversation_"),
    "ema": ("stress_", "affect_", "ema_"),
    "quality": ("cov__", "window_coverage", "valid_days"),
}
TRANSFORM_PREFIXES = {"pz__": "participant_zscore", "dev__": "participant_rolling_deviation",
                      "chg__": "participant_baseline_change"}
AGG_SUFFIXES = {"__mean": "window_mean", "__sd": "window_sd", "__slope": "window_slope"}


def _classify(col: str) -> dict[str, Any]:
    transform = "population_raw"
    base = col
    for pfx, name in TRANSFORM_PREFIXES.items():
        if col.startswith(pfx):
            transform, base = name, col[len(pfx):]
            break
    agg = "none"
    for sfx, name in AGG_SUFFIXES.items():
        if base.endswith(sfx):
            agg, base = name, base[: -len(sfx)]
            break
    domain = "other"
    for dom, pfxs in DOMAIN_PREFIXES.items():
        if base.startswith(pfxs) or col.startswith(pfxs):
            domain = dom
            break
    return {"feature": col, "base_signal": base, "domain": domain,
            "transform": transform, "aggregation": agg,
            "branch": "participant_relative" if transform != "population_raw" else "population"}


def build(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        rec = _classify(c)
        s = pd.to_numeric(df[c], errors="coerce")
        rec.update({
            "pct_missing": round(float(s.isna().mean()), 4),
            "min": None if s.dropna().empty else round(float(s.min()), 4),
            "p50": None if s.dropna().empty else round(float(s.median()), 4),
            "max": None if s.dropna().empty else round(float(s.max()), 4),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def save(registry: pd.DataFrame, csv_path: str | Path, json_path: str | Path) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(csv_path, index=False)
    Path(json_path).write_text(
        json.dumps(registry.to_dict(orient="records"), indent=2), encoding="utf-8")
