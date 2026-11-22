#!/usr/bin/env python
"""Repository security scan: secrets, identifiers, coordinates, restricted data."""
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from wearabledata.utils.redaction import PATTERNS, scan_text

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "data"}
TEXT_EXT = {".py", ".yaml", ".yml", ".md", ".tex", ".bib", ".json", ".csv", ".txt", ".cff"}
# Tests deliberately contain example identifiers to prove the scanner works.
ALLOWLIST = {"tests/unit/test_redaction.py", "src/wearabledata/utils/redaction.py",
             "tests/unit/test_readers.py"}

# Numeric result tables are matrices of floats, where the phone-number and
# coordinate regexes match adjacent decimals constantly. Structured data is
# instead checked by validation.checks.validate_processed_contract, which
# inspects column names and object columns.
# For .csv/.json we keep only the patterns that indicate a genuine identifier.
# .bib joins the structured set for the same reason: DOIs, page ranges, and
# volume numbers are long digit strings that trip the phone-number pattern.
DATA_EXT = {".csv", ".json", ".bib"}
DATA_PATTERNS = {k: v for k, v in PATTERNS.items()
                 if k in ("email", "api_key", "source_uid")}


def main() -> int:
    findings = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_EXT:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts) or rel in ALLOWLIST:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = scan_text(text, DATA_PATTERNS if p.suffix in DATA_EXT else None)
        hits.pop("iso_timestamp", None)   # documentation dates are not sensitive
        hits.pop("abs_path", None)
        if hits:
            findings.append({"file": rel, "hits": sorted(hits)})
    if findings:
        for f in findings:
            print(f"FAIL {f['file']}: {f['hits']}", file=sys.stderr)
        print(f"\nsecurity scan FAILED with {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("security scan PASS: no secrets, identifiers, or coordinates in tracked text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
