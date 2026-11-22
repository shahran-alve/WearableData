"""Depresjon reader.

Documented layout (datasets.simula.no/depresjon):

    depresjon/
      condition/condition_1.csv .. condition_23.csv
      control/control_1.csv .. control_32.csv
      scores.csv

Each participant CSV: timestamp, date, activity  (one-minute activity counts).
scores.csv: number, days, gender, age, afftype, melanch, inpatient, edu,
            marriage, work, madrs1, madrs2

Two traps this reader refuses to fall into:

  * MADRS exists only for the condition group. Filling controls with zero would
    create a perfectly separable feature and a meaningless classifier. MADRS is
    therefore returned as metadata, never as a predictor, and never imputed.
  * Five of the 23 condition participants were inpatients. The flag is carried
    through as metadata so a sensitivity analysis can exclude them; it is not a
    feature, because it encodes the outcome's clinical context.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..study_ids import study_id
from .contract import PanelContractError, validate_panel

EXPECTED_ACTIVITY_COLUMNS = {"timestamp", "date", "activity"}
EXPECTED_SCORE_COLUMNS = {"number", "days", "gender", "age", "afftype", "madrs1", "madrs2"}


def _daily_features(minutes: pd.DataFrame) -> pd.DataFrame:
    """Per-day circadian and activity summaries from minute-level counts."""
    m = minutes.copy()
    m["day"] = m["ts"].dt.floor("D")
    m["hour"] = m["ts"].dt.hour
    rows = []
    for day, blk in m.groupby("day"):
        a = blk["activity"].to_numpy(dtype=float)
        if len(a) < 600:          # < 10 h of recording; not a usable day
            continue
        rest = blk.loc[blk["hour"].isin([0, 1, 2, 3, 4]), "activity"].to_numpy(dtype=float)
        active = blk.loc[blk["hour"].between(10, 20), "activity"].to_numpy(dtype=float)
        nz = a[a > 0]
        hist = np.histogram(a, bins=20, range=(0, max(a.max(), 1)))[0].astype(float)
        pk = hist / max(hist.sum(), 1)
        rows.append({
            "day": day,
            "activity_total": float(a.sum()),
            "activity_mean": float(a.mean()),
            "activity_sd": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
            "sedentary_fraction": float((a == 0).mean()),
            "active_bout_count": int(((a[:-1] == 0) & (a[1:] > 0)).sum()),
            "activity_entropy": float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum()),
            # Intradaily variability: mean squared first difference over variance.
            "intradaily_variability": float(
                np.mean(np.diff(a) ** 2) / (np.var(a) + 1e-9)) if len(a) > 2 else np.nan,
            "rest_period_activity": float(rest.mean()) if len(rest) else np.nan,
            "peak_activity_mean": float(active.mean()) if len(active) else np.nan,
            "relative_amplitude": float(
                (active.mean() - rest.mean()) / (active.mean() + rest.mean() + 1e-9))
            if len(rest) and len(active) else np.nan,
            "nonzero_fraction": float(len(nz) / len(a)),
            "minutes_recorded": int(len(a)),
        })
    return pd.DataFrame(rows)


def load_daily_panel(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_panel, participant_metadata).

    The metadata frame carries MADRS, inpatient status, and demographics. It is
    deliberately a separate object so that nothing in it can be swept into the
    feature matrix by a wildcard column selection.
    """
    root = Path(raw_dir)
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    base = next((c for c in candidates
                 if (c / "condition").is_dir() and (c / "control").is_dir()), None)
    if base is None:
        raise PanelContractError(
            f"expected condition/ and control/ directories under {root}. Found: "
            f"{[p.name for p in root.iterdir()] if root.exists() else 'nothing'}. "
            "Unzip the official archive into data/raw/depresjon/ without renaming.")

    scores_path = base / "scores.csv"
    if not scores_path.exists():
        raise PanelContractError(f"missing {scores_path}; scores.csv ships with the dataset")
    scores = pd.read_csv(scores_path)
    if "number" in scores.columns and not scores["number"].astype(str).str.contains(
            "condition_|control_").any():
        raise PanelContractError(
            "scores.csv 'number' column does not contain condition_/control_ identifiers; "
            "the join key differs from the documented release. Inspect before proceeding.")
    missing = EXPECTED_SCORE_COLUMNS - set(scores.columns)
    if missing:
        raise PanelContractError(
            f"scores.csv is missing expected columns {sorted(missing)}. The dataset layout "
            "differs from the published description; inspect it before proceeding rather "
            "than adapting this reader silently.")

    panels, meta = [], []
    for group, label in (("condition", 1), ("control", 0)):
        for f in sorted((base / group).glob(f"{group}_*.csv")):
            df = pd.read_csv(f)
            if not EXPECTED_ACTIVITY_COLUMNS.issubset(df.columns):
                raise PanelContractError(
                    f"{f.name}: expected columns {sorted(EXPECTED_ACTIVITY_COLUMNS)}, "
                    f"found {sorted(df.columns)}")
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["ts"])
            daily = _daily_features(df[["ts", "activity"]])
            if daily.empty:
                warnings.warn(f"{f.stem}: no day met the 10-hour minimum; excluded", stacklevel=2)
                continue
            source_uid = f.stem                       # e.g. condition_7
            sid = study_id(source_uid, prefix="D")
            daily = daily.sort_values("day").reset_index(drop=True)
            daily["study_id"] = sid
            daily["day_index"] = np.arange(len(daily), dtype=int)
            daily["label"] = label
            panels.append(daily.drop(columns=["day"]))  # exact dates never leave the reader

            # `number` is the full "condition_7" string in the released scores.csv,
            # not an integer. Joining on int() silently produced an empty match and
            # a table of NaN metadata, which is why this compares the id directly.
            row = scores[scores["number"].astype(str) == source_uid]
            meta.append({
                "study_id": sid, "group": group, "n_days": len(daily),
                "madrs1": float(row["madrs1"].iloc[0]) if len(row) and group == "condition"
                          else np.nan,
                "madrs2": float(row["madrs2"].iloc[0]) if len(row) and group == "condition"
                          else np.nan,
                "inpatient": (float(row["inpatient"].iloc[0])
                              if len(row) and "inpatient" in row else np.nan),
                "gender": row["gender"].iloc[0] if len(row) else np.nan,
                "age_band": row["age"].iloc[0] if len(row) else np.nan,
            })

    if not panels:
        raise PanelContractError("no participant produced a usable day")
    panel = pd.concat(panels, ignore_index=True)
    metadata = pd.DataFrame(meta)

    n_cond = int((metadata["group"] == "condition").sum())
    n_ctrl = int((metadata["group"] == "control").sum())
    if (n_cond, n_ctrl) != (23, 32):
        warnings.warn(
            f"expected 23 condition / 32 control per the dataset paper, found "
            f"{n_cond}/{n_ctrl}. Note the discrepancy before "
            "using these counts in the manuscript.", stacklevel=2)
    if metadata.loc[metadata.group == "control", "madrs1"].notna().any():
        raise PanelContractError(
            "controls have MADRS values; absent MADRS must stay absent, not become zero")
    return validate_panel(panel, "depresjon"), metadata
