"""Content hashing for manifests and reproducibility checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any) -> str:
    """Order-independent hash of a config dict."""
    return sha256_text(canonical_json(obj))


def hash_tree(root: str | Path, pattern: str = "*") -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    for p in sorted(root.rglob(pattern)):
        if p.is_file():
            out[str(p.relative_to(root))] = sha256_file(p)
    return out
