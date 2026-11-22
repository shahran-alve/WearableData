"""Official dataset source registry and acquisition state machine.

Design rule from the build guide (Pages 8-9): stop at the permission boundary.
This module never falls back to a mirror, never generates replacement data, and
never marks a gate PASS on the basis of files it did not hash itself.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.config import load, repo_path
from ..utils.hashing import sha256_file


@dataclass
class AcquisitionStatus:
    dataset: str
    state: str            # PRESENT | BLOCKED_MANUAL_ACCESS | BLOCKED_NETWORK | MISSING
    reason: str
    landing_page: str
    files_found: int
    action_required: str


def raw_dir(dataset: str) -> Path:
    return repo_path("data", "raw", dataset)


def check(dataset: str) -> AcquisitionStatus:
    cfg = load("datasets")["datasets"][dataset]
    d = raw_dir(dataset)
    files = [p for p in d.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if files:
        return AcquisitionStatus(dataset, "PRESENT", f"{len(files)} raw files present",
                                 cfg["official_landing_page"], len(files), "none")
    mode = cfg.get("access_mode", "manual_form")
    action = MANUAL_INSTRUCTIONS.get(dataset, "See official landing page.")
    state = "BLOCKED_MANUAL_ACCESS" if mode in ("manual_form", "signed_eula") else "MISSING"
    return AcquisitionStatus(dataset, state,
                             f"no raw files under {d} (access_mode={mode})",
                             cfg["official_landing_page"], 0, action)


MANUAL_INSTRUCTIONS = {
    "studentlife": (
        "1. Open https://studentlife.cs.dartmouth.edu/dataset.html and read the terms of use.\n"
        "2. Download the full dataset archive from the official Dartmouth link only.\n"
        "3. Place the archive, unmodified, in data/raw/studentlife/.\n"
        "4. Run: python scripts/acquire_data.py --dataset studentlife --register\n"
        "   (this hashes every file and writes data/manifests/studentlife.json)\n"
        "Do NOT use Kaggle re-uploads, GitHub mirrors, or the R package cache as the "
        "source of record; they may be modified and their provenance is unverifiable."
    ),
    "depresjon": (
        "1. Open https://datasets.simula.no/depresjon/ and accept the stated terms.\n"
        "2. Download depresjon.zip (condition/, control/, scores.csv).\n"
        "3. Place it, unmodified, in data/raw/depresjon/.\n"
        "4. Run: python scripts/acquire_data.py --dataset depresjon --register"
    ),
    "daic_woz": (
        "FALLBACK ONLY. Requires a signed EULA from USC ICT "
        "(https://dcapswoz.ict.usc.edu/). Do not activate unless Depresjon acquisition "
        "has demonstrably failed; activating it changes the project scope and requires "
        "rewriting the title, methods, ethics, and limitations."
    ),
}


def register(dataset: str) -> Path:
    """Hash every raw file and write the source manifest. Requires files present."""
    cfg = load("datasets")["datasets"][dataset]
    d = raw_dir(dataset)
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name != ".gitkeep")
    if not files:
        raise FileNotFoundError(
            f"cannot register {dataset}: no files in {d}. "
            f"{MANUAL_INSTRUCTIONS.get(dataset, '')}"
        )
    entries = [
        {"path": str(p.relative_to(d)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in files
    ]
    manifest = {
        "dataset": dataset,
        "official_landing_page": cfg["official_landing_page"],
        "access_mode": cfg.get("access_mode"),
        "access_date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "license_text_verbatim": "see the official dataset page",
        "license_verified_by_human": False,
        "n_files": len(entries),
        "files": entries,
        "known_constraints": cfg.get("known_constraints", []),
    }
    out = repo_path("data", "manifests", f"{dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def verify(dataset: str) -> dict:
    """Re-hash files and compare against the recorded manifest."""
    man_path = repo_path("data", "manifests", f"{dataset}.json")
    if not man_path.exists():
        return {"dataset": dataset, "status": "NO_MANIFEST"}
    manifest = json.loads(man_path.read_text())
    d = raw_dir(dataset)
    mismatches, missing = [], []
    for entry in manifest["files"]:
        p = d / entry["path"]
        if not p.exists():
            missing.append(entry["path"])
        elif sha256_file(p) != entry["sha256"]:
            mismatches.append(entry["path"])
    return {
        "dataset": dataset,
        "status": "OK" if not (mismatches or missing) else "FAIL",
        "missing": missing,
        "hash_mismatch": mismatches,
        "license_verified_by_human": manifest.get("license_verified_by_human", False),
    }


def status_report() -> list[dict]:
    return [asdict(check(ds)) for ds in ("studentlife", "depresjon")]
