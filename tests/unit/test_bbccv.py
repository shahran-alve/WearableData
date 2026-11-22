"""Participant-level BBC-CV.

Known answers on synthetic score matrices, deterministic seeding, and the out-of-fold
score matrix against the original analysis() in scripts/selection_aware_permutation.py
(imported, not copied) on a 30-participant synthetic workspace with a 4-candidate family.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import participant_level_eval as ple  # noqa: E402
import selection_aware_permutation as sap  # noqa: E402
from make_synthetic_workspace import generate  # noqa: E402

from wearabledata.evaluation.bbccv import (  # noqa: E402
    auroc_columns,
    bbc_cv,
    oof_matrix,
    stratified_resample,
)

SIGNALS = [f"sig{j}" for j in range(6)]
COLS = ([f"{s}__{a}" for s in SIGNALS for a in ("mean", "sd", "slope")]
        + [f"{p}__{s}__mean" for p in ("pz", "dev", "chg") for s in SIGNALS]
        + [f"cov__{s}" for s in SIGNALS] + ["window_coverage", "valid_days"])
SMALL_FAMILY = [
    {"name": "logreg_C0.1_all", "family": "logreg", "C": 0.1, "branch": "all"},
    {"name": "logreg_C1_relative", "family": "logreg", "C": 1.0, "branch": "relative"},
    {"name": "rf_small", "family": "rf", "min_samples_leaf": 5, "branch": "all"},
    {"name": "hgb_small", "family": "hgb", "learning_rate": 0.1, "max_leaf_nodes": 7,
     "max_iter": 15, "branch": "all"},
]
N_FOLDS = 5
B = 400


def _labels(n: int, n_pos: int, seed: int) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    y[np.random.default_rng(seed).permutation(n)[:n_pos]] = 1
    return y


def _signal(y: np.ndarray, shift: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=len(y)) + shift * y


# ------------------------------------------------------------- AUROC
def test_auroc_columns_equals_sklearn_with_ties_and_duplicates():
    y = _labels(300, 90, 1)
    g = np.random.default_rng(2)
    S = np.column_stack([np.round(g.normal(size=300) + 0.4 * y, 1),   # many ties
                         g.integers(0, 3, 300).astype(float),          # heavy ties
                         g.normal(size=300)])
    idx = stratified_resample(y, np.random.default_rng(3))             # duplicates
    for yy, SS in ((y, S), (y[idx], S[idx])):
        ours = auroc_columns(yy, SS)
        ref = [roc_auc_score(yy, SS[:, j]) for j in range(SS.shape[1])]
        assert np.max(np.abs(ours - ref)) <= 1e-12


def test_stratified_resample_keeps_class_counts():
    y = _labels(97, 31, 4)
    idx = stratified_resample(y, np.random.default_rng(5))
    assert len(idx) == len(y)
    assert int(y[idx].sum()) == int(y.sum())


# ------------------------------------------------------------- known answers
def test_identical_candidates_have_no_selection_optimism():
    """K copies of one score: the winner's out-of-bag AUROC is that score's, every time.

    For a fixed score vector the out-of-bag set is a uniform random subset within each
    class, so its expected AUROC is the full-sample AUROC exactly: the optimism is zero
    up to Monte Carlo error.
    """
    y = _labels(400, 120, 6)
    s = _signal(y, 0.6, 7)
    names = ["a", "b", "c", "d"]
    one = bbc_cv(s[:, None], y, names[:1], B=B, seed_name="unit_identical")
    many = bbc_cv(np.column_stack([s] * 4), y, names, B=B, seed_name="unit_identical")
    assert many["bbc_corrected"] == one["bbc_corrected"]                # bitwise
    assert many["ci"] == one["ci"]
    assert many["naive_max"] == one["naive_max"] == roc_auc_score(y, s)
    assert many["optimism"] == one["optimism"]
    assert many["winner_share"]["d"] == 1.0                             # tie rule: by name
    mc_se = one["sd_oob_winner"] / np.sqrt(one["n_valid_resamples"])
    assert abs(one["optimism"]) < 4 * mc_se, (one["optimism"], mc_se)


def test_pure_noise_candidates_never_help():
    """Adding noise candidates to a real one cannot raise the corrected estimate here.

    The resamples do not depend on K, so the signal candidate's out-of-bag mean in the
    mixed run is bitwise the signal-only estimate. Whenever a noise column wins in-bag,
    its out-of-bag AUROC replaces the signal's with one centred on 0.5.
    """
    y = _labels(200, 60, 8)
    g = np.random.default_rng(9)
    sig = _signal(y, 0.35, 10)
    S = np.column_stack([sig] + [g.normal(size=len(y)) for _ in range(5)])
    names = ["signal"] + [f"noise{j}" for j in range(5)]
    alone = bbc_cv(S[:, :1], y, names[:1], B=B, seed_name="unit_noise")
    mixed = bbc_cv(S, y, names, B=B, seed_name="unit_noise")
    assert mixed["oob_mean_per_candidate"]["signal"] == alone["bbc_corrected"]
    noise_share = 1.0 - mixed["winner_share"]["signal"]
    assert noise_share > 0                           # the noise does win some resamples
    assert mixed["bbc_corrected"] < alone["bbc_corrected"]
    for j in range(1, 6):
        assert mixed["oob_mean_per_candidate"][names[j]] < alone["bbc_corrected"]


def test_all_noise_corrected_near_half_naive_above():
    """K = 10 pure-noise candidates: the naive winner is optimistic, BBC-CV is not.

    Tolerance: 3 null SDs of one AUROC at these class sizes (Hanley-McNeil under H0).
    """
    n, n_pos, K = 1000, 250, 10
    y = _labels(n, n_pos, 11)
    g = np.random.default_rng(12)
    S = g.normal(size=(n, K))
    r = bbc_cv(S, y, [f"n{j}" for j in range(K)], B=B, seed_name="unit_all_noise")
    sd0 = np.sqrt((n + 1) / (12 * n_pos * (n - n_pos)))
    assert abs(r["bbc_corrected"] - 0.5) < 3 * sd0
    assert r["naive_max"] > 0.5
    assert r["optimism"] > 0
    assert r["ci"][0] < 0.5 < r["ci"][1]


def test_deterministic_seeding_and_single_candidate():
    y = _labels(150, 50, 13)
    g = np.random.default_rng(14)
    S = np.column_stack([_signal(y, 0.5, 15), g.normal(size=150), _signal(y, 0.3, 16)])
    names = ["x", "y", "z"]
    a = bbc_cv(S, y, names, B=200, seed_name="unit_det", single="z")
    b = bbc_cv(S, y, names, B=200, seed_name="unit_det", single="z")
    c = bbc_cv(S, y, names, B=200, seed_name="unit_det_other", single="z")
    assert json.dumps(a) == json.dumps(b)
    assert a["bbc_corrected"] != c["bbc_corrected"]
    z_alone = bbc_cv(S[:, 2:], y, ["z"], B=200, seed_name="unit_det")
    assert a["single"]["oob_mean"] == z_alone["bbc_corrected"]
    assert a["single"]["ci"] == z_alone["ci"]


def test_bbc_rejects_bad_input():
    y = _labels(40, 10, 17)
    with pytest.raises(ValueError):
        bbc_cv(np.full((40, 2), np.nan), y, ["a", "b"], B=10)
    with pytest.raises(ValueError):
        bbc_cv(np.zeros((40, 2)), y, ["a"], B=10)


# ------------------------------------------------------------- OOF matrix
@pytest.fixture(scope="module", autouse=True)
def one_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def workspaces():
    return {e: generate(COLS, n_participants=30, n_positive=12, n_windows=300,
                        min_windows=8, max_windows=14, effect=v, seed_name="unit_bbccv")
            for e, v in (("null", 0.0), ("effect", 1.0))}


@pytest.mark.parametrize("regime", ["null", "effect"])
def test_oof_matrix_equals_original_analysis(workspaces, monkeypatch, regime):
    monkeypatch.setattr(sap, "CANDIDATES", SMALL_FAMILY)
    monkeypatch.setattr(sap, "FIXED_PIPELINE", "hgb_small")
    feat = workspaces[regime]
    y = feat["label"].to_numpy()
    folds = sap.make_folds(feat, y, N_FOLDS, f"cv:unit_{regime}:primary")
    pre = sap.precompute(feat, COLS, folds)
    orig = sap.analysis(feat, COLS, y, folds, pre=pre)
    serial = oof_matrix(y, pre, SMALL_FAMILY, sap.build, ple.aggregate)
    par = oof_matrix(y, pre, SMALL_FAMILY, sap.build, ple.aggregate, n_jobs=2,
                     max_nbytes=1)                    # force joblib's memmapped arrays
    assert serial["per_candidate_outer"] == orig["per_candidate_outer"]      # bitwise
    assert par["per_candidate_outer"] == orig["per_candidate_outer"]
    assert np.array_equal(serial["S"], par["S"])
    assert serial["n_fits"] == len(SMALL_FAMILY) * N_FOLDS
    assert serial["names"] == [c["name"] for c in SMALL_FAMILY]
    assert len(serial["y"]) == feat["study_id"].nunique()
    via_ranks = auroc_columns(serial["y"], serial["S"])
    ref = np.array([orig["per_candidate_outer"][n] for n in serial["names"]])
    assert np.max(np.abs(via_ranks - ref)) <= 1e-12
