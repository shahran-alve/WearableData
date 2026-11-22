"""Shared path bootstrap for scripts."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("WEARABLEDATA_ID_SALT", "SMOKE_ONLY_SALT_REPLACE_FOR_REAL_DATA")
