"""Participant-level splits, saved once and reused everywhere."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.hashing import sha256_text
from ..utils.seeds import rng


def grouped_stratified_split(df: pd.DataFrame, group_col: str = "study_id",
                             label_col: str = "label",
                             ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
                             seed_name: str = "split") -> dict[str, list[str]]:
    """Stratify on the participant-level majority label, split on participants."""
    per_person = (df.groupby(group_col)[label_col]
                  .agg(lambda s: int(round(float(s.mean())))).reset_index())
    g = rng(seed_name)
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for _lab, sub in per_person.groupby(label_col):
        ids = sub[group_col].to_numpy()
        ids = ids[g.permutation(len(ids))]
        n = len(ids)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        # Guarantee non-empty val/test strata when the stratum is large enough.
        if n >= 3:
            n_tr = min(max(n_tr, 1), n - 2)
            n_va = min(max(n_va, 1), n - n_tr - 1)
        splits["train"] += list(ids[:n_tr])
        splits["validation"] += list(ids[n_tr:n_tr + n_va])
        splits["test"] += list(ids[n_tr + n_va:])
    return {k: sorted(map(str, v)) for k, v in splits.items()}


def nested_grouped_cv(df: pd.DataFrame, group_col: str = "study_id", label_col: str = "label",
                      outer_folds: int = 5, seed_name: str = "cv") -> list[dict[str, list[str]]]:
    """Small-cohort fallback. Report fold-level results, not one point."""
    per_person = (df.groupby(group_col)[label_col]
                  .agg(lambda s: int(round(float(s.mean())))).reset_index())
    g = rng(seed_name)
    folds: list[list[str]] = [[] for _ in range(outer_folds)]
    for _lab, sub in per_person.groupby(label_col):
        ids = sub[group_col].to_numpy()
        ids = ids[g.permutation(len(ids))]
        for i, pid in enumerate(ids):
            folds[i % outer_folds].append(str(pid))
    out: list[dict[str, Any]] = []
    for k in range(outer_folds):
        test = sorted(folds[k])
        val = sorted(folds[(k + 1) % outer_folds])
        train = sorted(set(per_person[group_col].astype(str)) - set(test) - set(val))
        out.append({"fold": k, "train": train, "validation": val, "test": test})
    return out


def save_splits(splits: dict[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(splits, indent=2, sort_keys=True)
    digest = sha256_text(payload)
    p.write_text(payload, encoding="utf-8")
    (p.parent / (p.stem + ".sha256")).write_text(digest, encoding="utf-8")
    return digest


def load_splits(path: str | Path) -> tuple[dict[str, Any], str]:
    p = Path(path)
    payload = p.read_text(encoding="utf-8")
    return json.loads(payload), sha256_text(payload)


def mask_for(df: pd.DataFrame, ids: list[str], group_col: str = "study_id") -> np.ndarray:
    return df[group_col].astype(str).isin(set(ids)).to_numpy()
