"""Structured JSONL logging with mandatory redaction."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from .redaction import sanitize_record

_LOG_PATH: Path | None = None


def configure(path: str | Path) -> None:
    global _LOG_PATH
    _LOG_PATH = Path(path)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(event: str, level: str = "INFO", **fields: Any) -> None:
    record = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "level": level,
        "event": event,
    }
    record.update(sanitize_record(fields))
    line = json.dumps(record, default=str)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    print(line, file=sys.stderr)
