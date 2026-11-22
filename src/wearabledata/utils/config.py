"""Configuration loading. Configs are the single source of truth."""
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
REPO_ROOT = Path(__file__).resolve().parents[3]


@cache
def load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)
