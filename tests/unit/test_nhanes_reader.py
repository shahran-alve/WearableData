"""NHANES reader tests on small synthetic frames laid out like the CDC files.

They prove the exclusion rules, the feature definitions, and the identifier
handling. Only the last test touches a real file, and it is skipped when
data/raw/nhanes/ is absent.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("WEARABLEDATA_ID_SALT", "test_salt_not_for_production_use")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from wearabledata.preprocessing.readers import nhanes  # noqa: E402
from wearabledata.preprocessing.readers.contract import PanelContractError  # noqa: E402
from wearabledata.validation.checks import leakage_scan  # noqa: E402

RNG = np.random.default_rng(11)


def _person(seqn, cycle="G", age=40, ridstatr=2, items=(0,) * 9, in_dpq=True,
            paxsts=1, in_hd=True, wake=(900,) * 7, sex=1, stra=95, psu=1, hand=1):
    """`wake` lists PAXWWMD for days 1..len(wake); day 1 and the last are partial."""
    return dict(seqn=seqn, cycle=cycle, age=age, ridstatr=ridstatr, items=items,
                in_dpq=in_dpq, paxsts=paxsts, in_hd=in_hd, wake=wake, sex=sex,
                stra=stra, psu=psu, hand=hand)


def _frames(people):
    demo, dpq, hd, day = [], [], [], []
    for p in people:
        demo.append({"SEQN": float(p["seqn"]), "SDDSRVYR": 7.0 if p["cycle"] == "G" else 8.0,
                     "RIDSTATR": float(p["ridstatr"]), "RIAGENDR": float(p["sex"]),
                     "RIDAGEYR": float(p["age"]), "SDMVPSU": float(p["psu"]),
                     "SDMVSTRA": float(p["stra"]), "cycle": p["cycle"]})
        if p["in_dpq"]:
            dpq.append({"SEQN": float(p["seqn"]),
                        **{k: (np.nan if v is None else float(v))
                           for k, v in zip(nhanes.PHQ9_ITEMS, p["items"], strict=True)},
                        "cycle": p["cycle"]})
        if p["in_hd"]:
            hd.append({"SEQN": float(p["seqn"]), "PAXSTS": float(p["paxsts"]),
                       "PAXHAND": float(p["hand"]) if p["paxsts"] == 1 else np.nan,
                       "cycle": p["cycle"]})
        if p["in_hd"] and p["paxsts"] == 1:
            for d, w in enumerate(p["wake"], start=1):
                sleep = float(RNG.integers(300, 500))
                non = float(RNG.integers(0, 3)) * 10.0
                unk = float(RNG.integers(10, 60))
                vmd = min(1440.0, w + sleep + non + unk)
                day.append({"SEQN": float(p["seqn"]), "PAXDAYD": float(d), "PAXTMD": 1440.0,
                            "PAXAISMD": float(RNG.integers(1_000_000, 3_000_000)),
                            "PAXVMD": vmd, "PAXMTSD": float(RNG.normal(12000, 2000)),
                            "PAXWWMD": float(w), "PAXSWMD": sleep, "PAXNWMD": non,
                            "PAXUMD": unk, "PAXLXSD": float(RNG.normal(150000, 30000)),
                            "cycle": p["cycle"]})
    return {"DEMO": pd.DataFrame(demo), "DPQ": pd.DataFrame(dpq),
            "PAXHD": pd.DataFrame(hd), "PAXDAY": pd.DataFrame(day)}


POS = (2, 2, 2, 2, 1, 1, 0, 0, 0)       # total 10 -> positive
NEG = (1, 1, 1, 1, 1, 1, 1, 1, 1)       # total 9  -> negative


@pytest.fixture(scope="module")
def cohort():
    people = [
        _person(1, items=POS), _person(2, items=NEG),
        _person(3, cycle="H", items=POS, stra=110), _person(4, cycle="H", items=NEG, stra=110),
        _person(5, items=(7, 0, 0, 0, 0, 0, 0, 0, 0)),          # refused -> excluded
        _person(6, items=(0, 0, 0, 0, 0, 0, 0, 0, 9)),          # don't know -> excluded
        _person(7, items=(0, None, 0, 0, 0, 0, 0, 0, 0)),       # missing -> excluded
        _person(8, age=17),                                      # under 18 -> excluded
        _person(9, ridstatr=1),                                  # not examined -> excluded
        _person(10, in_dpq=False),                               # no DPQ -> excluded
        _person(11, paxsts=2),                                   # no PAM data -> excluded
        _person(12, in_hd=False),                                # no header -> excluded
        # 4 valid days only if the partial day 1 and final day counted -> excluded
        _person(13, wake=(900, 900, 900, 599, 599, 599, 900)),
        _person(14, items=POS, wake=(900, 600, 600, 600, 600, 599, 900)),  # exactly 4 -> kept
        _person(15, wake=(599,) * 7),                            # 0 valid days -> excluded
        _person(16, cycle="H", stra=110, hand=9),                # wrist unknown -> kept
        _person(17, wake=(900,)),                                # one (partial) day -> excluded
    ]
    return nhanes.build_cohort(_frames(people))


def _sid(seqn):
    from wearabledata.preprocessing.study_ids import study_id
    return study_id(f"SEQN{seqn}", prefix="N")


# -------------------------------------------------------------------- label
def test_phq9_scoring_and_exclusion_codes():
    dpq = pd.DataFrame([dict(zip(nhanes.PHQ9_ITEMS, row, strict=True)) for row in
                        (POS, NEG, (7,) + (0,) * 8, (9,) + (0,) * 8, (np.nan,) + (0,) * 8,
                         (3,) * 9)])
    dpq.insert(0, "SEQN", range(len(dpq)))
    s = nhanes.phq9(dpq)
    assert s["phq9_complete"].tolist() == [True, True, False, False, False, True]
    assert s["phq9_total"].iloc[[0, 1, 5]].tolist() == [10, 9, 27]
    assert s["label"].iloc[[0, 1, 5]].tolist() == [1, 0, 1]
    assert s["label"].iloc[2:5].isna().all()


def test_phq9_rejects_undocumented_code():
    dpq = pd.DataFrame([dict(zip(nhanes.PHQ9_ITEMS, (5,) + (0,) * 8, strict=True))])
    dpq.insert(0, "SEQN", [1])
    with pytest.raises(PanelContractError):
        nhanes.phq9(dpq)


# ---------------------------------------------------------------- exclusions
def test_final_cohort_membership(cohort):
    kept = set(cohort.metadata["study_id"])
    assert kept == {_sid(q) for q in (1, 2, 3, 4, 14, 16)}
    lab = dict(zip(cohort.metadata["study_id"], cohort.metadata["label"], strict=True))
    assert [lab[_sid(q)] for q in (1, 2, 3, 4, 14, 16)] == [1, 0, 1, 0, 1, 0]


def test_valid_day_boundary_is_600_minutes(cohort):
    day = pd.DataFrame({"PAXWWMD": [599, 600, 601]})
    assert nhanes.valid_day_mask(day).tolist() == [False, True, True]
    p14 = cohort.panel[cohort.panel["study_id"] == _sid(14)]
    assert len(p14) == 4 and (p14["wake_wear_minutes"] == 600).all()
    assert (p14["valid_days"] == 4).all()


def test_partial_day_mask():
    day = pd.DataFrame({"SEQN": [1, 1, 1, 1, 2, 2, 3], "PAXDAYD": [1, 2, 3, 4, 1, 2, 1]})
    assert nhanes.partial_day_mask(day).tolist() == [True, False, False, True,
                                                     True, True, True]


def test_first_and_final_days_never_become_windows(cohort):
    # every kept person wore the monitor 7 days (day_index 0..6); windows use 1..5 only
    assert cohort.panel["day_index"].between(1, 5).all()
    assert set(cohort.panel["day_index"]) == {1, 2, 3, 4, 5}
    assert cohort.notes["partial_days_dropped_final_cohort"] == 2 * 6


def test_paxhand_unknown_is_kept_and_counted(cohort):
    assert _sid(16) in set(cohort.metadata["study_id"])
    assert cohort.notes["paxhand_9_kept"] == 1
    assert cohort.notes["paxhand_counts_final"] == {"1": 5, "9": 1}


def test_flow_counts(cohort):
    f = cohort.flow.set_index("stage")
    assert f.loc["demo_records", "n"] == 17
    assert f.loc["mec_examined", "n"] == 16
    assert f.loc["adult", "n"] == 15
    assert f.loc["has_dpq", "n"] == 14
    assert f.loc["phq9_complete", "n"] == 11
    assert f.loc["has_paxhd", "n"] == 10
    assert f.loc["pam_data", "n"] == 9
    assert f.loc["has_day_rows", "n"] == 9
    assert f.loc["has_full_day", "n"] == 8                   # person 17 has no full day
    assert f.loc["any_valid_day", "n"] == 7                  # person 15
    assert f.loc["final", "n"] == 6 and f.loc["final", "n_positive"] == 3
    assert f.loc["day_rows", "n"] == 6 * 7                   # 6 people x 7 day rows
    assert f.loc["full_day_rows", "n"] == 6 * 5              # minus day 1 and day 7
    assert f.loc["windows", "n"] == len(cohort.panel) == 5 * 5 + 4   # person 14: 4 valid
    people = cohort.flow[cohort.flow["unit"] == "participants"]["n"].to_numpy()
    assert (np.diff(people) <= 0).all()
    assert f.loc["final", "n_2011_2012"] + f.loc["final", "n_2013_2014"] == 6


# --------------------------------------------------------------- identifiers
def test_identifiers_and_times_do_not_leave_the_reader(cohort):
    feats, cols = nhanes.build_features(cohort.panel)
    for frame in (cohort.panel, cohort.metadata, feats):
        assert not any(c.upper().startswith(("SEQN", "PAX", "DPQ", "RID")) for c in frame)
        assert frame["study_id"].str.match(r"^N[0-9a-f]{12}$").all()
    assert set(cohort.metadata.columns) == {"study_id", "cycle", "sdmvstra", "sdmvpsu",
                                            "age", "sex", "n_valid_days", "label"}
    assert set(cohort.metadata["cycle"]) == {"2011-2012", "2013-2014"}


def test_study_id_depends_on_salt(monkeypatch):
    a = _sid(62161)
    monkeypatch.setenv("WEARABLEDATA_ID_SALT", "a_different_salt")
    assert _sid(62161) != a


# ------------------------------------------------------------------ features
def test_feature_definitions(cohort):
    feats, cols = nhanes.build_features(cohort.panel)
    assert cols == list(nhanes.FEATURE_COLUMNS) and len(cols) == 35
    assert list(feats.columns[:3]) == ["study_id", "cutoff_day", "label"]
    for _, g in feats.groupby("study_id"):
        x = g["activity_mims_total"]
        np.testing.assert_allclose(g["pz__activity_mims_total"],
                                   (x - x.mean()) / x.std(ddof=1), atol=1e-12)
        assert abs(g["pz__activity_mims_total"].mean()) < 1e-12
        iqr = x.quantile(0.75) - x.quantile(0.25)
        np.testing.assert_allclose(g["dev__activity_mims_total"], (x - x.median()) / iqr,
                                   atol=1e-12)
        g = g.sort_values("cutoff_day")
        assert g["chg__activity_mims_total"].iloc[:3].isna().all()
        np.testing.assert_allclose(g["chg__activity_mims_total"].iloc[3:],
                                   x.loc[g.index].iloc[3:] - x.loc[g.index].iloc[:3].mean(),
                                   atol=1e-9)
    # person 14: every valid full day has wake = 600, so sd and IQR are 0 -> NaN, not 1e9
    p14 = feats[feats["study_id"] == _sid(14)]
    assert p14["pz__wake_wear_minutes"].isna().all()
    assert p14["dev__wake_wear_minutes"].isna().all()
    assert feats["window_coverage"].between(0, 1).all()
    assert feats["cov__wear_fraction"].between(0, 1).all()


def test_daily_feature_arithmetic():
    day = pd.DataFrame({"SEQN": [1.0], "PAXDAYD": [2.0], "PAXTMD": [1440.0],
                        "PAXAISMD": [3_456_000.0], "PAXVMD": [1200.0], "PAXMTSD": [12000.0],
                        "PAXWWMD": [800.0], "PAXSWMD": [300.0], "PAXNWMD": [60.0],
                        "PAXUMD": [40.0], "PAXLXSD": [240000.0]})
    f = nhanes.daily_features(day).iloc[0]
    assert f["day_index"] == 1
    assert f["activity_mims_per_min"] == pytest.approx(10.0)
    assert f["activity_idle_fraction"] == pytest.approx(0.5)
    assert f["light_lux_per_min"] == pytest.approx(200.0)
    assert f["window_coverage"] == pytest.approx(1200 / 1440)
    assert f["cov__wear_fraction"] == pytest.approx(1100 / 1200)


def test_candidate_branches_see_nhanes_columns(cohort):
    from selection_aware_permutation import branch_columns
    _, cols = nhanes.build_features(cohort.panel)
    pop = branch_columns(cols, "population")
    rel = branch_columns(cols, "relative")
    assert set(pop) == set(nhanes.BASE_FEATURES) | set(nhanes.QUALITY_FEATURES)
    assert len(rel) == 24 + 3 and set(nhanes.QUALITY_FEATURES) <= set(rel)
    assert branch_columns(cols, "all") == cols


def test_no_label_leakage(cohort):
    feats, cols = nhanes.build_features(cohort.panel)
    assert "label" not in cols
    assert not [c for c in cols if "dpq" in c.lower() or "phq" in c.lower()]
    scan = leakage_scan(feats, cols)
    assert not scan["suspicious_columns"] and not scan["identifier_columns"]


# ----------------------------------------------------------------------- I/O
def test_fix_xport_zeros():
    df = pd.DataFrame({"a": [5.397605346934028e-79, 1.0, 0.0, np.nan], "b": ["x"] * 4})
    out = nhanes.fix_xport_zeros(df)
    assert out["a"].iloc[0] == 0.0 and out["a"].iloc[1] == 1.0 and np.isnan(out["a"].iloc[3])
    assert out["b"].tolist() == ["x"] * 4


RAW_DPQ_G = ROOT / "data" / "raw" / "nhanes" / "DPQ_G.xpt"


@pytest.mark.skipif(not RAW_DPQ_G.exists(), reason="NHANES raw files not present")
def test_real_dpq_g_matches_codebook():
    df = nhanes.read_xpt(RAW_DPQ_G)
    assert len(df) == 5615
    assert int(df["DPQ010"].isna().sum()) == 664
    assert int((df["DPQ010"] == 0).sum()) == 3726
