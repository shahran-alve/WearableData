"""The contract every reader must satisfy before its output enters the pipeline."""
from __future__ import annotations

import pandas as pd

from ...utils.redaction import scan_columns

REQUIRED = ("study_id", "day_index", "label")


class PanelContractError(RuntimeError):
    """Raised when a reader produces something the pipeline must not ingest."""


def validate_panel(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise PanelContractError(f"{source}: panel missing required columns {missing}")

    forbidden = scan_columns(list(df.columns))
    if forbidden:
        raise PanelContractError(
            f"{source}: panel carries forbidden columns {forbidden}. Coordinates, exact "
            "timestamps, and source identifiers must be consumed and dropped inside the "
            "reader; they may not reach the processed zone.")

    if df["study_id"].astype(str).str.match(r"^u\d{2}$").any():
        raise PanelContractError(
            f"{source}: study_id still contains raw source identifiers. Map them through "
            "preprocessing.study_ids.study_id() before returning.")

    if not pd.api.types.is_integer_dtype(df["day_index"]):
        raise PanelContractError(f"{source}: day_index must be an integer day offset")

    bad = df.groupby("study_id")["day_index"].apply(lambda s: s.duplicated().any())
    if bad.any():
        raise PanelContractError(
            f"{source}: duplicate (study_id, day_index) rows for {int(bad.sum())} "
            "participant(s); the panel must be one row per participant-day")

    n_lab = df["label"].dropna().nunique()
    if n_lab < 2:
        raise PanelContractError(
            f"{source}: label has {n_lab} distinct value(s); nothing can be learned")
    return df
