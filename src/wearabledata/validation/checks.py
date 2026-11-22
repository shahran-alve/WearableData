"""Data validation: schema, quality, leakage, and identifier scans."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.redaction import scan_columns, scan_text

# Column-name fragments that would leak the target into the features.
TARGET_LEAK_TOKENS = ("label", "target", "phq", "madrs", "diagnos", "depress",
                      "outcome", "y_true", "class", "group", "condition")


def schema_report(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "table": name,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": [
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "n_missing": int(df[c].isna().sum()),
                "pct_missing": round(float(df[c].isna().mean()), 4),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
    }


def quality_report(df: pd.DataFrame, group_col: str = "study_id",
                   label_col: str = "label") -> pd.DataFrame:
    rows = []
    for gid, sub in df.groupby(group_col):
        rec = {
            group_col: gid,
            "n_rows": len(sub),
            "pct_missing": round(float(sub.isna().mean().mean()), 4),
        }
        if label_col in sub.columns:
            rec["label_mean"] = round(float(sub[label_col].mean()), 4)
            rec["label_constant"] = bool(sub[label_col].nunique() == 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def leakage_scan(df: pd.DataFrame, feature_cols: list[str], label_col: str = "label",
                 corr_threshold: float = 0.995) -> dict[str, Any]:
    """Blocking scan. Any finding must be resolved before modeling."""
    findings: dict[str, Any] = {"suspicious_columns": [], "near_perfect_predictors": [],
                                "identifier_columns": [], "status": "PASS"}
    for c in feature_cols:
        low = c.lower()
        if any(tok in low for tok in TARGET_LEAK_TOKENS):
            findings["suspicious_columns"].append(c)
    findings["identifier_columns"] = scan_columns(feature_cols)

    if label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce")
        for c in feature_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() < 10 or x[ok].nunique() < 2:
                continue
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if abs(r) >= corr_threshold:
                findings["near_perfect_predictors"].append({"column": c, "pearson_r": round(r, 6)})

    if (findings["suspicious_columns"] or findings["near_perfect_predictors"]
            or findings["identifier_columns"]):
        findings["status"] = "FAIL"
    return findings


def assert_unique_columns(df: pd.DataFrame, context: str = "frame") -> None:
    """Duplicate column names double a feature's influence and break column scans."""
    dupes = sorted(set(df.columns[df.columns.duplicated()]))
    if dupes:
        raise ValueError(
            f"{context}: duplicate column names {dupes[:10]}"
            f"{' ...' if len(dupes) > 10 else ''}. Deduplicate the feature list before "
            "modelling; a repeated column silently doubles that feature's weight.")


def pii_scan_frame(df: pd.DataFrame, max_cells: int = 20000) -> dict[str, Any]:
    """Scan object columns for emails, phones, coordinates, timestamps, keys."""
    assert_unique_columns(df, "pii_scan_frame")
    hits: dict[str, Any] = {}
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    for c in obj_cols:
        sample = df[c].dropna().astype(str).head(max_cells)
        blob = "\n".join(sample.tolist())
        found = scan_text(blob)
        if found:
            hits[c] = found
    return {"status": "FAIL" if hits else "PASS", "columns_with_hits": hits,
            "object_columns_scanned": obj_cols}


def split_disjointness(splits: dict[str, list[str]]) -> dict[str, Any]:
    """Participant-level partitions must be pairwise disjoint."""
    names = list(splits)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sorted(set(splits[a]) & set(splits[b]))
            if inter:
                overlaps[f"{a}|{b}"] = inter
    return {"status": "FAIL" if overlaps else "PASS",
            "overlaps": overlaps,
            "sizes": {k: len(v) for k, v in splits.items()}}


def validate_processed_contract(df: pd.DataFrame) -> dict[str, Any]:
    """Processed zone: no coordinates, text, audio, exact timestamps, source IDs."""
    assert_unique_columns(df, "processed frame")
    bad_cols = scan_columns(list(df.columns))
    pii = pii_scan_frame(df)
    status = "FAIL" if (bad_cols or pii["status"] == "FAIL") else "PASS"
    return {"status": status, "forbidden_columns": bad_cols, "pii": pii}
