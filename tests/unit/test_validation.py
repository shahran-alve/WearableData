import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import pytest

from wearabledata.validation.checks import leakage_scan, validate_processed_contract


def test_leakage_scan_flags_target_derived_column():
    df = pd.DataFrame({"phq9_total": [1, 2, 3, 4], "x": [1, 0, 1, 0], "label": [0, 0, 1, 1]})
    r = leakage_scan(df, ["phq9_total", "x"])
    assert r["status"] == "FAIL" and "phq9_total" in r["suspicious_columns"]


def test_leakage_scan_flags_near_perfect_predictor():
    lab = [0] * 8 + [1] * 8
    df = pd.DataFrame({"sneaky": [v * 3.0 + 0.001 for v in lab], "label": lab})
    r = leakage_scan(df, ["sneaky"])
    assert r["near_perfect_predictors"] and r["status"] == "FAIL"


def test_leakage_scan_skips_correlation_on_too_few_rows():
    """A 4-row frame cannot support a correlation claim; the check abstains."""
    df = pd.DataFrame({"x": [0, 0, 1, 1], "label": [0, 0, 1, 1]})
    assert leakage_scan(df, ["x"])["near_perfect_predictors"] == []


def test_processed_contract_rejects_identifiers():
    df = pd.DataFrame({"latitude": [1.0], "sleep_mean": [7.0]})
    assert validate_processed_contract(df)["status"] == "FAIL"


def test_duplicate_columns_are_rejected_with_a_clear_message():
    """`pz__x__mean` also ends in `__mean`; naive selection produces this exact bug."""
    from wearabledata.validation.checks import assert_unique_columns

    df = pd.DataFrame(np.zeros((3, 2)), columns=["x__mean", "x__mean"])
    with pytest.raises(ValueError, match="duplicate column names"):
        assert_unique_columns(df, "test")
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_processed_contract(df)
