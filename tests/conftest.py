import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("WEARABLEDATA_ID_SALT", "test_salt_not_for_production_use")


@pytest.fixture(scope="session")
def panel():
    from wearabledata.acquisition.fixtures import make_daily_panel
    return make_daily_panel(n_participants=24, days=35, seed_name="test_panel")


@pytest.fixture(scope="session")
def windows(panel):
    from wearabledata.preprocessing.windows import WindowSpec, build_windows
    cols = [c for c in panel.columns if c not in ("study_id", "day_index", "label")]
    w, _ = build_windows(panel, cols, WindowSpec(history_days=7, min_coverage=0.4,
                                                 min_valid_days=4))
    return w
