"""Reader tests against synthetic files laid out exactly as each dataset documents.

These cannot prove the readers work on the real archives -- only real files can do
that. What they do prove is that the readers implement the documented layout, that
they refuse the specific failure modes that would corrupt the study, and that
coordinates and exact dates are consumed rather than passed through.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ.setdefault("WEARABLEDATA_ID_SALT", "test_salt_not_for_production_use")

import numpy as np
import pandas as pd
import pytest

from wearabledata.preprocessing.readers.contract import PanelContractError, validate_panel
from wearabledata.preprocessing.readers.depresjon import load_daily_panel as load_depresjon
from wearabledata.preprocessing.readers.studentlife import TargetNotChosen
from wearabledata.preprocessing.readers.studentlife import load_daily_panel as load_sl

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ Depresjon
def _make_depresjon(root: Path, n_cond: int = 4, n_ctrl: int = 5, days: int = 6) -> None:
    # Group means overlap heavily and every participant carries a large personal
    # offset. An earlier version used well-separated means, and the leakage scan
    # correctly refused the resulting features as near-perfect predictors -- which
    # is exactly the behaviour we want, but it made the fixture useless.
    for group, n, base in (("condition", n_cond, 165.0), ("control", n_ctrl, 195.0)):
        d = root / group
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            ts = pd.date_range("2003-05-07 12:00", periods=days * 1440, freq="min")
            hour = ts.hour.to_numpy()
            circadian = np.where((hour >= 0) & (hour < 5), 0.05, 1.0)
            person = base * float(np.exp(RNG.normal(0, 0.35)))
            act = np.clip(RNG.poisson(person * circadian)
                          + RNG.normal(0, 45, size=len(ts)), 0, None)
            pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                          "date": ts.strftime("%Y-%m-%d"),
                          "activity": act}).to_csv(d / f"{group}_{i}.csv", index=False)
    rows = []
    for i in range(1, n_cond + 1):
        # The released scores.csv stores `number` as the full "condition_7" string,
        # not an integer. An earlier fixture used integers, the reader's int() join
        # matched nothing, and the metadata table came back silently all-NaN.
        rows.append({"number": f"condition_{i}", "days": days, "gender": 1,
                     "age": "35-39", "afftype": 1, "melanch": 1,
                     "inpatient": 1 if i == 1 else 2, "edu": "6-10",
                     "marriage": 1, "work": 1, "madrs1": 20 + i, "madrs2": 18 + i})
    for i in range(1, n_ctrl + 1):
        rows.append({"number": f"control_{i}", "days": days, "gender": 2,
                     "age": "30-34", "afftype": "", "melanch": "", "inpatient": "",
                     "edu": "", "marriage": "", "work": "", "madrs1": "", "madrs2": ""})
    pd.DataFrame(rows).to_csv(root / "scores.csv", index=False)


def test_depresjon_reader_produces_a_valid_panel(tmp_path):
    _make_depresjon(tmp_path / "depresjon")
    with pytest.warns(UserWarning, match="expected 23 condition"):
        panel, meta = load_depresjon(tmp_path)
    assert {"study_id", "day_index", "label"} <= set(panel.columns)
    assert panel["label"].nunique() == 2
    assert panel.groupby("study_id")["day_index"].apply(lambda s: s.is_monotonic_increasing).all()
    assert "intradaily_variability" in panel.columns


def test_depresjon_reader_never_emits_dates_or_source_ids(tmp_path):
    _make_depresjon(tmp_path / "depresjon")
    with pytest.warns(UserWarning):
        panel, _ = load_depresjon(tmp_path)
    assert not any(c.lower() in ("date", "timestamp", "day") for c in panel.columns)
    assert not panel["study_id"].astype(str).str.contains("condition_|control_").any()


def test_depresjon_keeps_control_madrs_absent(tmp_path):
    """Filling control MADRS with zero would create a perfectly separable feature."""
    _make_depresjon(tmp_path / "depresjon")
    with pytest.warns(UserWarning):
        _, meta = load_depresjon(tmp_path)
    assert meta.loc[meta.group == "control", "madrs1"].isna().all()
    assert meta.loc[meta.group == "condition", "madrs1"].notna().all()


def test_depresjon_carries_inpatient_flag_as_metadata_not_feature(tmp_path):
    _make_depresjon(tmp_path / "depresjon")
    with pytest.warns(UserWarning):
        panel, meta = load_depresjon(tmp_path)
    assert "inpatient" in meta.columns
    assert "inpatient" not in panel.columns


def test_depresjon_rejects_a_wrong_layout(tmp_path):
    (tmp_path / "depresjon" / "patients").mkdir(parents=True)
    with pytest.raises(PanelContractError, match="condition/ and control/"):
        load_depresjon(tmp_path)


# --------------------------------------------------------------- StudentLife
def _make_studentlife(root: Path, uids=("u00", "u01", "u02", "u03"), days: int = 8) -> None:
    ds = root / "dataset"
    for stream in ("activity", "conversation", "phonelock", "gps"):
        (ds / "sensing" / stream).mkdir(parents=True, exist_ok=True)
    for kind in ("Stress", "Mood"):
        (ds / "EMA" / "response" / kind).mkdir(parents=True, exist_ok=True)
    (ds / "survey").mkdir(parents=True, exist_ok=True)

    t0 = int(pd.Timestamp("2013-03-27 08:00").timestamp())
    for uid in uids:
        n = days * 96
        ts = t0 + np.arange(n) * 900
        pd.DataFrame({"timestamp": ts,
                      " activity inference": RNG.integers(0, 3, n)}).to_csv(
            ds / "sensing" / "activity" / f"activity_{uid}.csv", index=False)
        m = days * 12
        st = t0 + RNG.integers(0, days * 86400, m)
        pd.DataFrame({"start_timestamp": st,
                      " end_timestamp": st + RNG.integers(60, 1800, m)}).to_csv(
            ds / "sensing" / "conversation" / f"conversation_{uid}.csv", index=False)
        pd.DataFrame({"start": st, "end": st + RNG.integers(60, 3600, m)}).to_csv(
            ds / "sensing" / "phonelock" / f"phonelock_{uid}.csv", index=False)
        g = days * 40
        gt = t0 + RNG.integers(0, days * 86400, g)
        pd.DataFrame({"time": gt,
                      "latitude": 43.7044 + RNG.normal(0, 0.004, g),
                      "longitude": -72.2887 + RNG.normal(0, 0.004, g)}).to_csv(
            ds / "sensing" / "gps" / f"gps_{uid}.csv", index=False)
        (ds / "EMA" / "response" / "Stress" / f"Stress_{uid}.json").write_text(json.dumps(
            [{"resp_time": int(t0 + d * 86400 + 40000), "level": int(RNG.integers(1, 6))}
             for d in range(days)]))
        (ds / "EMA" / "response" / "Mood" / f"Mood_{uid}.json").write_text(json.dumps(
            [{"resp_time": int(t0 + d * 86400 + 50000), "happy": int(RNG.integers(1, 5))}
             for d in range(days)]))

    items = [f"Q{i}" for i in range(1, 10)]
    rows = []
    for k, uid in enumerate(uids):
        for wave in ("pre", "post"):
            rec = {"uid": uid, "type": wave}
            high = (k % 2 == 1) and wave == "post"
            for q in items:
                rec[q] = "Nearly every day" if high else "Not at all"
            rows.append(rec)
    pd.DataFrame(rows).to_csv(ds / "survey" / "PHQ-9.csv", index=False)


def test_studentlife_refuses_to_guess_a_target(tmp_path):
    _make_studentlife(tmp_path)
    with pytest.raises(TargetNotChosen, match="pre- and post-term only"):
        load_sl(tmp_path)


def test_studentlife_phq9_post_target(tmp_path):
    _make_studentlife(tmp_path)
    panel, meta = load_sl(tmp_path, target="phq9_post")
    assert panel["label"].nunique() == 2
    # One label per participant: constant within person, as a two-wave instrument implies.
    assert (panel.groupby("study_id")["label"].nunique() == 1).all()
    assert meta["phq9_post"].notna().all()


def test_studentlife_ema_target_is_strictly_forward_looking(tmp_path):
    _make_studentlife(tmp_path)
    panel, _ = load_sl(tmp_path, target="ema_stress")
    # A next-day label cannot exist for each participant's final day.
    per = panel.groupby("study_id")["day_index"].max()
    assert len(panel) > 0 and per.notna().all()
    assert panel["label"].nunique() == 2


def test_studentlife_consumes_coordinates_and_never_returns_them(tmp_path):
    _make_studentlife(tmp_path)
    panel, _ = load_sl(tmp_path, target="phq9_post")
    assert not any(c.lower() in ("latitude", "longitude", "lat", "lon") for c in panel.columns)
    assert "radius_of_gyration" in panel.columns
    assert "location_entropy" in panel.columns


def test_studentlife_warns_about_label_attrition(tmp_path):
    _make_studentlife(tmp_path)
    with pytest.warns(UserWarning, match="usable phq9_post label"):
        load_sl(tmp_path, target="phq9_post")


# ------------------------------------------------------------------ contract
def test_contract_rejects_raw_source_ids():
    df = pd.DataFrame({"study_id": ["u00", "u01"], "day_index": [0, 1], "label": [0, 1]})
    with pytest.raises(PanelContractError, match="raw source identifiers"):
        validate_panel(df, "test")


def test_contract_rejects_duplicate_participant_days():
    df = pd.DataFrame({"study_id": ["S1", "S1"], "day_index": [0, 0], "label": [0, 1]})
    with pytest.raises(PanelContractError, match="one row per participant-day"):
        validate_panel(df, "test")


def test_contract_rejects_a_constant_label():
    df = pd.DataFrame({"study_id": ["S1", "S2"], "day_index": [0, 0], "label": [1, 1]})
    with pytest.raises(PanelContractError, match="distinct value"):
        validate_panel(df, "test")


def test_depresjon_rejects_an_integer_join_key(tmp_path):
    """Regression: the released scores.csv keys on "condition_7", not 7.

    An integer key matched nothing and produced a table of all-NaN metadata
    rather than an error, which is exactly the kind of silent failure the
    contract checks exist to prevent.
    """
    root = tmp_path / "depresjon"
    _make_depresjon(root, n_cond=3, n_ctrl=3, days=6)
    scores = pd.read_csv(root / "scores.csv")
    scores["number"] = range(1, len(scores) + 1)      # the wrong format
    scores.to_csv(root / "scores.csv", index=False)
    with pytest.raises(PanelContractError, match="condition_/control_"):
        load_depresjon(tmp_path)
