"""Null recycling: leave-one-out p-values over a permutation stream."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import null_recycling as nr  # noqa: E402

from wearabledata.evaluation.fastperm import write_checkpoint  # noqa: E402


def _stream(t_sel, t_fix, selected="x"):
    n = len(t_sel)
    return pd.DataFrame({"index": np.arange(n), "T_selection_aware": t_sel, "T_fixed": t_fix,
                         "selected": [selected] * n})


def _brute(s, s_idx, ref, ref_idx):
    out = []
    for sb, ib in zip(s, s_idx, strict=True):
        keep = [r for r, j in zip(ref, ref_idx, strict=True) if j != ib]
        out.append((1 + sum(r >= sb for r in keep)) / (len(keep) + 1))
    return np.array(out)


def test_identical_streams_give_alpha():
    t = np.random.default_rng(0).normal(size=1000)
    res, per_b = nr.recycle(_stream(t, t.copy()), alpha=0.05)
    assert np.array_equal(per_b["p_fixed"].to_numpy(), per_b["p_sel"].to_numpy())
    # p(b) = rank of b from the top / B, so exactly floor(0.05 B) labelings reject.
    assert res["typeI_fixed_null"]["n_reject"] == 50
    assert res["typeI_fixed_null"]["rate"] == pytest.approx(0.05)
    assert res["typeI_sel_null"]["rate"] == pytest.approx(0.05)
    lo, hi = res["typeI_fixed_null"]["ci"]
    assert lo < 0.05 < hi
    assert res["shift"]["sd_units"] == 0.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_brute_force_with_ties(seed):
    g = np.random.default_rng(seed)
    s = np.round(g.normal(0.55, 0.05, 120), 2)          # rounding forces ties
    ref_paired = np.round(g.normal(0.5, 0.05, 120), 2)
    idx = np.arange(120)
    assert np.array_equal(nr.loo_pvalues(s, idx, ref_paired, idx),
                          _brute(s, idx, ref_paired, idx))
    assert np.array_equal(nr.loo_pvalues(s, idx, s, idx), _brute(s, idx, s, idx))
    # Longer reference stream that holds most, but not all, of the pseudo-observed indices.
    ref_idx = np.r_[np.arange(10, 400)]
    ref = np.round(g.normal(0.5, 0.05, len(ref_idx)), 2)
    ref[:5] = s[10:15]                                   # own-index ties with s_b
    assert np.array_equal(nr.loo_pvalues(s, idx, ref, ref_idx), _brute(s, idx, ref, ref_idx))


def test_constant_shift():
    t = np.random.default_rng(4).normal(0.5, 0.02, 500)
    res, _ = nr.recycle(_stream(t + 0.03, t), alpha=0.05)
    assert res["shift"]["mean"] == pytest.approx(0.03)
    assert res["shift"]["sd_units"] == pytest.approx(0.03 / np.std(t, ddof=1))
    assert res["shift"]["sd_units_se"] == pytest.approx(0.0, abs=1e-12)
    assert res["typeI_fixed_null"]["rate"] > 0.3
    assert res["typeI_sel_null"]["rate"] <= 0.05


def test_checkpoint_round_trip_and_consistency(tmp_path):
    g = np.random.default_rng(5)
    t_fix = g.normal(0.5, 0.02, 300)
    t_sel = np.maximum(t_fix, g.normal(0.5, 0.02, 300))
    sel_path, fix_path = tmp_path / "sel.csv", tmp_path / "fix.csv"
    write_checkpoint(sel_path, {i: (t_sel[i], t_fix[i], "hgb_lr0.1_leaf31" if t_sel[i] == t_fix[i]
                                    else "rf_leaf5") for i in range(100)})
    fixed = {i: (float("nan"), t_fix[i], "none") for i in range(300)}
    fixed[7] = (float("nan"), np.nextafter(t_fix[7], 1.0), "none")   # one ulp off
    write_checkpoint(fix_path, fixed)
    sel, fix = nr.read_stream(sel_path), nr.read_stream(fix_path)
    assert np.array_equal(sel["T_fixed"].to_numpy(), t_fix[:100])       # bit-exact read
    res, per_b = nr.recycle(sel, fix, alpha=0.05)
    fo = res["fixed_only_checkpoint"]
    assert (fo["B"], fo["n_common_indices"], fo["n_T_fixed_bitwise_equal"]) == (300, 100, 99)
    assert res["B"] == 100 and res["indices_are_prefix"]
    share = np.mean(t_sel[:100] == t_fix[:100])
    assert res["share_selected_fixed_pipeline"] == pytest.approx(share)
    expected = _brute(t_sel[:100], np.arange(100), fix["T_fixed"].to_numpy(), np.arange(300))
    assert np.array_equal(per_b["p_fixed_full"].to_numpy(), expected)
