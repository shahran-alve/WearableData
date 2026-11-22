"""Source identifier -> generated study ID. Mapping never leaves the raw zone."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _salt() -> bytes:
    s = os.environ.get("WEARABLEDATA_ID_SALT")
    if not s:
        raise RuntimeError(
            "WEARABLEDATA_ID_SALT is not set. Generate one locally "
            "(python -c \"import secrets;print(secrets.token_hex(32))\"), export it, "
            "and never commit it. An unsalted hash of a small ID space is trivially "
            "reversible by brute force and is not de-identification."
        )
    return s.encode()


def study_id(source_uid: str, prefix: str = "P") -> str:
    digest = hashlib.blake2b(source_uid.encode(), key=_salt(), digest_size=6).hexdigest()
    return f"{prefix}{digest}"


def build_mapping(source_uids: list[str], out_path: str | Path | None = None) -> dict[str, str]:
    """Optionally persist the mapping OUTSIDE the repo (raw zone / local only)."""
    mapping = {u: study_id(u) for u in sorted(set(source_uids))}
    if out_path is not None:
        p = Path(out_path)
        if "data/raw" not in str(p).replace("\\", "/"):
            raise ValueError("ID mapping may only be written inside data/raw (git-ignored)")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        os.chmod(p, 0o600)
    return mapping
