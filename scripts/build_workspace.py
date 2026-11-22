#!/usr/bin/env python
"""Build the feature workspace for Depresjon, StudentLife, or the synthetic fixture.

Runs the preprocess and features stages and writes
<out>/interim/features.csv and <out>/interim/feature_columns.json, which the
permutation scripts read. Stages are skipped when their inputs are unchanged;
pass --force to recompute.

For Depresjon, the list of participants who were not inpatients is written to
<out>/results/cohorts/inpatients_excluded.json, for
participant_level_eval.py --subset-participants.

Examples:
    python scripts/build_workspace.py --source depresjon --out reports/real_depresjon
    python scripts/build_workspace.py --source studentlife --target phq9_post \\
        --out reports/real_sl_phq9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from wearabledata.workspace import Workspace, stage_features, stage_preprocess

ROOT = Path(__file__).resolve().parents[1]


def write_inpatient_cohort(ws: Workspace) -> None:
    meta_path = ws.path("validation", "participant_metadata.csv")
    meta = pd.read_csv(meta_path)
    feat = pd.read_csv(ws.path("interim", "features.csv"), usecols=["study_id"])
    inpatients = set(meta.loc[meta["inpatient"] == 1.0, "study_id"])
    keep = sorted(set(feat["study_id"]) - inpatients)
    out = ws.path("results", "cohorts", "inpatients_excluded.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(keep, indent=2))
    print(f"{len(inpatients)} inpatients excluded, {len(keep)} participants -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["depresjon", "studentlife", "fixture"],
                    required=True)
    ap.add_argument("--target", default=None, help="StudentLife outcome, e.g. phq9_post")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-root", default=None, help="default: data/raw")
    ap.add_argument("--participants", type=int, default=40, help="fixture only")
    ap.add_argument("--days", type=int, default=45, help="fixture only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    ws = Workspace(Path(a.out))
    stage_preprocess(ws, n_participants=a.participants, days=a.days, force=a.force,
                     source=a.source, target=a.target,
                     raw_root=Path(a.raw_root) if a.raw_root else None)
    feat, cols = stage_features(ws, force=a.force)
    print(f"{feat['study_id'].nunique()} participants, {len(feat)} windows, "
          f"{len(cols)} features -> {ws.path('interim', 'features.csv')}")
    if a.source == "depresjon":
        write_inpatient_cohort(ws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
