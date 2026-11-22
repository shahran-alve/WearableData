import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from wearabledata.utils.redaction import (
    PrivacyLintError,
    assert_clean,
    redact,
    scan_columns,
    scan_text,
)


@pytest.mark.parametrize("text,key", [
    ("contact a@b.edu", "email"),
    ("call 555-123-4567", "phone"),
    ("at 43.70421, -72.28869", "latlon"),
    ("2013-04-02T14:33:00", "iso_timestamp"),
    ("participant u07 logged in", "source_uid"),
    ("key sk-abcdefghijklmnopqrstuvwx", "api_key"),
])
def test_scan_detects(text, key):
    assert key in scan_text(text)


def test_redact_removes_content():
    assert "a@b.edu" not in redact("mail a@b.edu now")


def test_assert_clean_blocks():
    with pytest.raises(PrivacyLintError):
        assert_clean("lat 43.70421, -72.28869", context="prompt")
    assert_clean("sleep_duration__mean 7.1")  # must not raise


def test_forbidden_columns():
    assert set(scan_columns(["latitude", "sleep_mean", "uid"])) == {"latitude", "uid"}
