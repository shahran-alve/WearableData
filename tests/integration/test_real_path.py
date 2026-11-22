"""End-to-end test of the real-data path, using synthetic files in the documented layouts.

reader -> contract -> windows -> personalization -> features, with nothing from the
fixture generator involved. It cannot validate the real schemas (only real files
can), but it shows that the stages are connected, so a first run on real data
fails for a data reason rather than a code reason.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
os.environ.setdefault("WEARABLEDATA_ID_SALT", "test_salt_not_for_production_use")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from test_readers import _make_depresjon, _make_studentlife  # noqa: E402

from wearabledata import workspace  # noqa: E402


def test_depresjon_builds_a_feature_workspace(tmp_path):
    ws = workspace.Workspace(tmp_path)
    _make_depresjon(tmp_path / "raw" / "depresjon", n_cond=8, n_ctrl=8, days=14)
    with pytest.warns(UserWarning, match="expected 23 condition"):
        workspace.stage_preprocess(ws, source="depresjon", raw_root=ws.root / "raw")
    feat, cols = workspace.stage_features(ws)

    assert ws.data_source == "REAL:depresjon"
    assert len(cols) > 10
    assert feat["study_id"].nunique() == 16
    assert ws.path("interim", "feature_columns.json").exists()


def test_studentlife_target_builds_a_feature_workspace(tmp_path):
    ws = workspace.Workspace(tmp_path)
    _make_studentlife(tmp_path / "raw" / "studentlife",
                      uids=tuple(f"u{i:02d}" for i in range(12)), days=20)
    workspace.stage_preprocess(ws, source="studentlife", target="ema_stress",
                               raw_root=ws.root / "raw")
    feat, _ = workspace.stage_features(ws)
    assert ws.data_source == "REAL:studentlife"
    assert feat["study_id"].nunique() >= 8


def test_studentlife_without_a_target_is_refused(tmp_path):
    from wearabledata.preprocessing.readers.studentlife import TargetNotChosen

    ws = workspace.Workspace(tmp_path)
    _make_studentlife(tmp_path / "raw" / "studentlife", days=20)
    with pytest.raises(TargetNotChosen):
        workspace.stage_preprocess(ws, source="studentlife", raw_root=ws.root / "raw")


def test_participant_metadata_never_reaches_the_feature_matrix(tmp_path):
    """MADRS and inpatient status are written beside the features, never into them."""
    ws = workspace.Workspace(tmp_path)
    _make_depresjon(tmp_path / "raw" / "depresjon", n_cond=8, n_ctrl=8, days=14)
    with pytest.warns(UserWarning):
        workspace.stage_preprocess(ws, source="depresjon", raw_root=ws.root / "raw")
    _, cols = workspace.stage_features(ws)

    meta_path = ws.path("validation", "participant_metadata.csv")
    assert meta_path.exists()
    meta_cols = set(pd.read_csv(meta_path).columns) - {"study_id"}
    # Exact names, not substrings: "window_coverage" contains "age".
    assert meta_cols & {"madrs1", "madrs2", "inpatient", "gender", "age_band"}
    assert not (meta_cols & set(cols)), (
        f"metadata leaked into the feature matrix: {sorted(meta_cols & set(cols))}")
