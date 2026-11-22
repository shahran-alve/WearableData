"""Immutable experiment manifests. Never overwrite a run."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .hashing import config_hash


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "UNVERSIONED"


def environment_record() -> dict[str, Any]:
    try:
        import numpy  # noqa: E401
        import scipy
        import sklearn

        pkg = {"numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
               "scipy": scipy.__version__}
    except Exception:
        pkg = {}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkg,
    }


def new_run_id(prefix: str) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}__{stamp}"


def write_manifest(out_dir: str | Path, run_id: str, *, config: dict[str, Any],
                   data_manifests: dict[str, str] | None = None,
                   seeds: dict[str, int] | None = None,
                   artifacts: dict[str, str] | None = None,
                   extra: dict[str, Any] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing run manifest: {path}")
    record = {
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_hash": config_hash(config),
        "config": config,
        "data_manifest_hashes": data_manifests or {},
        "seeds": seeds or {},
        "artifacts": artifacts or {},
        "environment": environment_record(),
        "extra": extra or {},
    }
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
