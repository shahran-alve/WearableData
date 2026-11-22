#!/usr/bin/env python
"""Inspect a downloaded archive before running anything on it.

Run this the moment a dataset lands in data/raw/. It unpacks nothing, changes
nothing, and reports whether the layout matches what the readers expect. Catching
a layout mismatch here costs seconds; catching it after a full pipeline run costs
an afternoon and produces numbers you then have to distrust.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from wearabledata.acquisition import sources

EXPECTATIONS = {
    "studentlife": {
        "required_dirs": ["sensing", "survey"],
        "required_globs": ["**/sensing/activity/activity_u*.csv",
                           "**/survey/*PHQ*"],
        "optional_globs": ["**/EMA/response/Stress/Stress_u*.json",
                           "**/sensing/gps/gps_u*.csv",
                           "**/sensing/conversation/conversation_u*.csv"],
        "expected_participants": 48,
    },
    "depresjon": {
        "required_dirs": ["condition", "control"],
        "required_globs": ["**/condition/condition_*.csv", "**/control/control_*.csv",
                           "**/scores.csv"],
        "optional_globs": [],
        "expected_participants": 55,
    },
}


def _listing(root: Path) -> list[str]:
    """Filenames on disk, plus the contents of any zip found (without extracting)."""
    names = [str(p.relative_to(root)) for p in root.rglob("*")
             if p.is_file() and p.name != ".gitkeep"]
    for z in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                names += [f"{z.name}!{n}" for n in zf.namelist()]
        except zipfile.BadZipFile:
            names.append(f"{z.name}!<UNREADABLE ZIP>")
    return names


def inspect(dataset: str) -> dict:
    spec = EXPECTATIONS[dataset]
    root = sources.raw_dir(dataset)
    if not root.exists():
        return {"dataset": dataset, "status": "MISSING", "path": str(root)}
    names = _listing(root)
    if not names:
        return {"dataset": dataset, "status": "EMPTY", "path": str(root),
                "action": sources.MANUAL_INSTRUCTIONS[dataset]}

    zipped = any("!" in n for n in names)
    found = {}
    for pattern in spec["required_globs"] + spec["optional_globs"]:
        hits = list(root.glob(pattern))
        found[pattern] = len(hits)
    missing = [p for p in spec["required_globs"] if found.get(p, 0) == 0]

    participants: set[str] = set()
    for p in root.rglob("*.csv"):
        stem = p.stem
        for pfx in ("activity_", "condition_", "control_"):
            if stem.startswith(pfx):
                participants.add(stem)

    status = "OK"
    notes = []
    if zipped and missing:
        status = "NEEDS_EXTRACTION"
        notes.append("A .zip is present but its contents are not on disk. Extract it "
                     "in place, keeping the archive's own directory names.")
    elif missing:
        status = "LAYOUT_MISMATCH"
        notes.append(f"required patterns not found: {missing}")
    if participants and len(participants) != spec["expected_participants"]:
        notes.append(f"found {len(participants)} participant files, the dataset paper "
                     f"reports {spec['expected_participants']}; record the discrepancy "
                     "before using either count")
    return {"dataset": dataset, "status": status, "path": str(root),
            "n_files_on_disk": sum(1 for n in names if "!" not in n),
            "zip_present": zipped, "pattern_hits": found,
            "participant_files": len(participants), "notes": notes,
            "next_command": (f"python scripts/acquire_data.py --dataset {dataset} "
                             "--register --verify") if status == "OK" else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*EXPECTATIONS, "all"], default="all")
    a = ap.parse_args()
    names = list(EXPECTATIONS) if a.dataset == "all" else [a.dataset]
    reports = [inspect(d) for d in names]
    print(json.dumps(reports, indent=2))
    return 0 if all(r["status"] == "OK" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
