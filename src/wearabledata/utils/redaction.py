"""Redaction and privacy linting.

Used in two blocking places:
  1. structured logging (never write raw records or secrets)
  2. repository security scan (never commit restricted content)
"""
from __future__ import annotations

import re
from typing import Any

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
    "latlon": re.compile(r"([-+]?\d{1,3}\.\d{4,})\s*,\s*([-+]?\d{1,3}\.\d{4,})"),
    "iso_timestamp": re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?"),
    "api_key": re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{20,}|ghp_[A-Za-z0-9]{20,})"),
    "source_uid": re.compile(r"\bu\d{2}\b"),  # StudentLife source participant ids (u00..u59)
    "abs_path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}

# Field names that must never appear in a processed table or a prompt.
FORBIDDEN_FIELDS = {
    "lat", "lon", "latitude", "longitude", "gps", "bssid", "ssid",
    "timestamp", "time", "date", "uid", "source_uid", "participant_number",
    "audio", "transcript", "text", "response_text", "filename", "path",
}


def _plausible_latlon(match: tuple[str, str] | str) -> bool:
    """Reject adjacent decimals that cannot be a coordinate pair.

    A results CSV is full of pairs like `0.4915,0.1822`. Flagging those would
    make the scanner noisy enough that people start ignoring it, which is worse
    than a slightly narrower pattern. Both components inside (-1, 1) is treated
    as numeric data, since the only land near (0, 0) is the Gulf of Guinea.
    """
    if isinstance(match, str):
        return True
    try:
        a, b = float(match[0]), float(match[1])
    except ValueError:
        return True
    if abs(a) < 1.0 and abs(b) < 1.0:
        return False
    return abs(a) <= 90.0 and abs(b) <= 180.0


def scan_text(text: str,
              patterns: dict[str, re.Pattern[str]] | None = None) -> dict[str, list[str]]:
    """Return {pattern_name: [matches]} for anything found."""
    pats = patterns or PATTERNS
    hits: dict[str, list[str]] = {}
    for name, pat in pats.items():
        found = pat.findall(text)
        if name == "latlon":
            found = [m for m in found if _plausible_latlon(m)]
        if found:
            hits[name] = [m if isinstance(m, str) else ", ".join(m) for m in found][:5]
    return hits


def redact(text: str) -> str:
    out = text
    for name, pat in PATTERNS.items():
        out = pat.sub(f"<{name.upper()}_REDACTED>", out)
    return out


def scan_columns(columns: list[str]) -> list[str]:
    """Return column names that violate the processed-zone contract."""
    bad = []
    for c in columns:
        token = c.lower()
        if token in FORBIDDEN_FIELDS:
            bad.append(c)
    return bad


class PrivacyLintError(RuntimeError):
    """Raised when content that must never leave the machine is detected."""


def assert_clean(text: str, context: str = "payload") -> None:
    hits = scan_text(text)
    if hits:
        raise PrivacyLintError(
            f"privacy lint failed for {context}: {sorted(hits)} "
            "(this payload must not be logged, committed, or sent to an external API)"
        )


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: (redact(v) if isinstance(v, str) else v) for k, v in record.items()}
