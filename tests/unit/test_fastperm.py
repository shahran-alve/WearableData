"""Exact-speedup engine: the lazy path, parallel runs and the sequential rule.

The reference is the original analysis() in scripts/selection_aware_permutation.py,
imported, not copied. A 4-candidate family (no random forest, whose 150 trees
dominate run time) and a 30-participant synthetic workspace keep this file fast;
the full 8-candidate check on the Depresjon-shaped workspace is
`scripts/fast_selection_permutation.py equality`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import fast_selection_permutation as fsp  # noqa: E402
import participant_level_eval as ple  # noqa: E402
import selection_aware_permutation as sap  # noqa: E402
from make_synthetic_workspace import generate  # noqa: E402

from wearabledata.evaluation.fastperm import (  # noqa: E402
    CHECKPOINT_COLUMNS,
    PermContext,
    besag_clifford,
    iter_permutations,
    lazy_analysis,
    permutation_statistic,
    read_checkpoint,
    write_checkpoint,
)
from wearabledata.utils.seeds import derive_seed  # noqa: E402

SIGNALS = [f"sig{j}" for j in range(6)]
COLS = ([f"{s}__{a}" for s in SIGNALS for a in ("mean", "sd", "slope")]
        + [f"{p}__{s}__mean" for p in ("pz", "dev", "chg") for s in SIGNALS]
        + [f"cov__{s}" for s in SIGNALS] + ["window_coverage", "valid_days"])
SMALL_FAMILY = [
    {"name": "logreg_C0.1_all", "family": "logreg", "C": 0.1, "branch": "all"},
    {"name": "logreg_C1_population", "family": "logreg", "C": 1.0, "branch": "population"},
    {"name": "logreg_C1_relative", "family": "logreg", "C": 1.0, "branch": "relative"},
    {"name": "hgb_small", "family": "hgb", "learning_rate": 0.1, "max_leaf_nodes": 7,
     "max_iter": 15, "branch": "all"},
]
SMALL_FIXED = "hgb_small"
N_FOLDS = 5
TOL = 1e-12


def _feat(effect: float) -> pd.DataFrame:
    return generate(COLS, n_participants=30, n_positive=12, n_windows=300,
                    min_windows=8, max_windows=14, effect=effect, seed_name="unit_fastperm")


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Thread count changes no result here, but default OpenMP threading makes small
    HGB fits very slow when other processes load the machine."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def workspaces() -> dict[str, pd.DataFrame]:
    return {"null": _feat(0.0), "effect": _feat(1.0)}


@pytest.fixture
def small_family(monkeypatch):
    monkeypatch.setattr(sap, "CANDIDATES", SMALL_FAMILY)
    monkeypatch.setattr(sap, "FIXED_PIPELINE", SMALL_FIXED)


def _ctx(feat: pd.DataFrame, name: str, threads: int | None = None) -> PermContext:
    return PermContext(feat=feat, cols=COLS, y=feat["label"].to_numpy(), n_folds=N_FOLDS,
                       fold_seed_name=f"cv:{name}:primary",
                       base_seed=derive_seed(f"selperm:{name}"),
                       candidates=SMALL_FAMILY, fixed_name=SMALL_FIXED, build=sap.build,
                       aggregate=ple.aggregate, permute=sap.permute,
                       make_folds=sap.make_folds, precompute=sap.precompute, threads=threads)


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= TOL


# ------------------------------------------------------------- lazy == original
@pytest.mark.parametrize("regime", ["null", "effect"])
def test_lazy_matches_original_analysis(workspaces, small_family, regime):
    feat = workspaces[regime]
    name = f"unit_{regime}"
    ctx = _ctx(feat, name)
    y = ctx.y
    for index in (-1, 0, 1):
        if index < 0:
            yb = y
        else:
            yb = sap.permute(feat, y, ctx.base_seed + index)
        folds = sap.make_folds(feat, yb, N_FOLDS, ctx.fold_seed_name)
        pre = sap.precompute(feat, COLS, folds)
        orig, n_orig = fsp.original_analysis_counted(feat, COLS, yb, folds, pre)
        lazy = lazy_analysis(yb, pre, SMALL_FAMILY, sap.build, ple.aggregate, SMALL_FIXED)
        assert lazy["selected_by_inner"] == orig["selected_by_inner"]
        assert _same(lazy["T_selection_aware"], orig["T_selection_aware"])
        assert _same(lazy["T_fixed"], orig["T_fixed"])
        assert set(lazy["inner_mean"]) == set(orig["inner_mean"])
        for k, v in lazy["inner_mean"].items():
            assert _same(v, orig["inner_mean"][k])
        for k, v in lazy["per_candidate_outer"].items():
            assert _same(v, orig["per_candidate_outer"][k])
        # Fit counts: original K x folds x 2; lazy K x folds + folds (+ folds for fixed).
        K = len(SMALL_FAMILY)
        assert n_orig == 2 * K * N_FOLDS
        assert lazy["n_fits_inner"] == K * N_FOLDS
        assert lazy["n_fits_outer_selected"] == N_FOLDS
        extra = 0 if lazy["selected_by_inner"] == SMALL_FIXED else N_FOLDS
        assert lazy["n_fits_outer_fixed"] == extra
        assert lazy["n_fits"] == K * N_FOLDS + N_FOLDS + extra
        if index >= 0:   # the driver's code path gives the same numbers
            r = permutation_statistic(ctx, index)
            assert r["selected"] == (orig["selected_by_inner"] or "none")
            assert _same(r["T_selection_aware"], orig["T_selection_aware"])
            assert _same(r["T_fixed"], orig["T_fixed"])


def test_lazy_without_fixed_skips_fixed_refit(workspaces):
    feat = workspaces["effect"]
    y = feat["label"].to_numpy()
    pre = sap.precompute(feat, COLS, sap.make_folds(feat, y, N_FOLDS, "cv:unit_effect:primary"))
    r = lazy_analysis(y, pre, SMALL_FAMILY, sap.build, ple.aggregate, SMALL_FIXED,
                      with_fixed=False)
    assert r["n_fits_outer_fixed"] == 0
    assert r["n_fits"] == len(SMALL_FAMILY) * N_FOLDS + N_FOLDS
    if r["selected_by_inner"] != SMALL_FIXED:
        assert np.isnan(r["T_fixed"])


# ------------------------------------------------------------- parallel == serial
def test_parallel_equals_serial(workspaces):
    ctx = _ctx(workspaces["effect"], "unit_effect")
    idx = [0, 1, 2, 3]
    serial = [r for c in iter_permutations(ctx, idx, n_jobs=1) for r in c]
    par_ctx = _ctx(workspaces["effect"], "unit_effect", threads=1)
    par = [r for c in iter_permutations(par_ctx, idx, n_jobs=2, chunk_size=3) for r in c]
    assert [r["index"] for r in par] == idx
    for s, p in zip(serial, par, strict=True):
        assert s["selected"] == p["selected"]
        assert _same(s["T_selection_aware"], p["T_selection_aware"])
        assert _same(s["T_fixed"], p["T_fixed"])
        assert s["n_fits"] == p["n_fits"]
    assert all(r["pid"] != os.getpid() for r in par)


# ------------------------------------------------------------- Besag-Clifford
def test_besag_clifford_hand_stream():
    stream = [0.6, 0.1, 0.7, 0.2, 0.9, 0.3, 0.8]
    r = besag_clifford(stream, observed=0.5, h=3)
    assert r["decided"] and r["reached_h"]
    assert r["L"] == 5 and r["n_ge"] == 3 and r["p_value"] == pytest.approx(3 / 5)
    tie = besag_clifford([0.5, 0.5], observed=0.5, h=2)      # ties count, as in mc_p
    assert tie["reached_h"] and tie["p_value"] == pytest.approx(1.0)


def test_besag_clifford_falls_back_to_fixed_B():
    stream = [0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.2, 0.3, 0.4, 0.1]
    r = besag_clifford(stream, observed=0.5, h=5)
    assert r["decided"] and not r["reached_h"]
    assert r["p_value"] == pytest.approx((1 + 2) / (10 + 1))
    capped = besag_clifford(stream, observed=0.5, h=5, B_max=4)
    assert capped["L"] == 4 and capped["p_value"] == pytest.approx((1 + 1) / (4 + 1))
    pending = besag_clifford(stream[:3], observed=0.5, h=5, B_max=10, complete=False)
    assert not pending["decided"] and pending["p_value"] is None
    with pytest.raises(ValueError):
        besag_clifford([0.1, float("nan")], observed=0.5)


def test_besag_clifford_null_is_exact_at_attainable_levels():
    """Under H0, P(p <= h/l) = h/l exactly; h=10 gives 0.05 at l=200 and 0.1 at l=100."""
    g = np.random.default_rng(derive_seed("unit_bc_null"))
    reps, B_max, h = 4000, 999, 10
    ps = np.array([besag_clifford(g.random(B_max), g.random(), h=h, B_max=B_max)["p_value"]
                   for _ in range(reps)])
    assert abs(np.mean(ps <= 0.05) - 0.05) < 0.015
    assert abs(np.mean(ps <= 0.10) - 0.10) < 0.02


def test_besag_clifford_known_p():
    """Observed at the 95th percentile of a U(0,1) null: true p = 0.05.
    (h - 1)/(L - 1) is the unbiased negative-binomial estimate of p."""
    g = np.random.default_rng(derive_seed("unit_bc_known_p"))
    h, B_max = 10, 999
    res = [besag_clifford(g.random(B_max), 0.95, h=h, B_max=B_max) for _ in range(2000)]
    assert all(r["reached_h"] for r in res)
    est = np.array([(h - 1) / (r["L"] - 1) for r in res])
    assert abs(est.mean() - 0.05) < 0.002
    assert abs(np.median([r["p_value"] for r in res]) - 0.05) < 0.01


# ------------------------------------------------------------- driver
def _write_ws(root: Path, name: str, feat: pd.DataFrame) -> Path:
    ws = root / name
    (ws / "interim").mkdir(parents=True)
    feat.to_csv(ws / "interim" / "features.csv", index=False)
    (ws / "interim" / "feature_columns.json").write_text(json.dumps(COLS))
    return ws


def _run_args(ws: Path, **kw) -> argparse.Namespace:
    base = dict(workspace=str(ws), permutations=3, folds=N_FOLDS, n_jobs=1,
                threads_per_task=1, chunk_size=None, time_budget=None, sequential=False,
                h=10, sequential_stat="selection_aware", checkpoint=None, out=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_checkpoint_schema_and_resume(tmp_path, workspaces, small_family):
    ws = _write_ws(tmp_path, "unit_resume", workspaces["null"])
    ckpt = ws / "results" / "selection_aware_null_checkpoint.csv"
    fsp.cmd_run(_run_args(ws, permutations=2))
    first = pd.read_csv(ckpt, float_precision="round_trip")
    assert list(first.columns) == CHECKPOINT_COLUMNS
    fsp.cmd_run(_run_args(ws, permutations=3))
    report = json.loads((ws / "results" / "selection_aware_permutation_fast.json").read_text())
    assert report["execution"]["new_permutations"] == 3 - len(first)
    second = read_checkpoint(ckpt)
    for _, r in first.iterrows():
        assert second[int(r["index"])][:2] == (r["T_selection_aware"], r["T_fixed"])
    # the permuted statistic in the checkpoint is the original loop body's statistic
    feat, cols, y = fsp.load_workspace(ws)
    o = fsp.original_permutation(ws, feat, cols, y, 2, N_FOLDS)
    assert _same(second[2][0], o["T_selection_aware"]) and _same(second[2][1], o["T_fixed"])
    assert second[2][2] == o["selected"]
    write_checkpoint(tmp_path / "rt.csv", second)
    assert read_checkpoint(tmp_path / "rt.csv") == second


def test_checkpoint_floats_round_trip_exactly(tmp_path):
    """AUROC-like values (k / (n_pos * n_neg)) must come back bit-identical."""
    vals = [k / (23 * 32) for k in range(23 * 32 + 1)]
    done = {i: (v, 1 - v, "x") for i, v in enumerate(vals)}
    write_checkpoint(tmp_path / "ck.csv", done)
    back = read_checkpoint(tmp_path / "ck.csv")
    assert all(back[i][0] == v and back[i][1] == 1 - v for i, v in enumerate(vals))


def test_sequential_stop_independent_of_n_jobs(tmp_path, workspaces, small_family):
    # On this seeded null workspace the observed fixed-pipeline AUROC is low, so the
    # rule on that statistic stops within the first permutations.
    ws = _write_ws(tmp_path, "unit_seq", workspaces["null"])
    out = {}
    for n_jobs in (1, 2):
        ck = tmp_path / f"ck{n_jobs}.csv"
        js = tmp_path / f"r{n_jobs}.json"
        fsp.cmd_run(_run_args(ws, permutations=6, n_jobs=n_jobs, chunk_size=2 if n_jobs > 1
                              else None, sequential=True, h=2, sequential_stat="fixed",
                              checkpoint=str(ck), out=str(js)))
        out[n_jobs] = json.loads(js.read_text())["sequential"]
    s1, s2 = out[1], out[2]
    assert s1["decided"] and s2["decided"]
    for key in ("reached_h", "L", "n_ge", "p_value", "stop_index"):
        assert s1[key] == s2[key]
    assert s1["reached_h"] and s1["stop_index"] < 5      # the early-stop path is exercised
    assert s1["p_value"] == pytest.approx(2 / s1["L"])


# ------------------------------------------------- fixed-only and within-block
from functools import partial  # noqa: E402

from wearabledata.evaluation.fastperm import (  # noqa: E402
    TYPEI_COLUMNS,
    permute_within_blocks,
    read_checkpoint_extras,
    typeI_procedures,
)


def _blocks(feat: pd.DataFrame, k: int = 3) -> dict:
    ids = sorted(feat["study_id"].unique())
    return {sid: 90 + (j % k) for j, sid in enumerate(ids)}


def _per_person(feat: pd.DataFrame, y: np.ndarray) -> pd.Series:
    return pd.DataFrame({"g": feat["study_id"], "y": y}).groupby("g")["y"].first()


def test_block_permutation_preserves_block_counts(workspaces):
    feat = workspaces["null"]
    y = feat["label"].to_numpy()
    block_of = _blocks(feat)
    obs = _per_person(feat, y)
    blk = obs.index.map(block_of)
    moved_any = False
    for seed in range(20):
        yb = permute_within_blocks(feat, y, seed, block_of)
        per = _per_person(feat, yb)
        # constant within person, and every block keeps its label counts
        assert (pd.DataFrame({"g": feat["study_id"], "y": yb}).groupby("g")["y"].nunique()
                == 1).all()
        assert (per.groupby(blk).sum() == obs.groupby(blk).sum()).all()
        assert (per.groupby(blk).size() == obs.groupby(blk).size()).all()
        moved_any |= bool((per != obs).any())
        np.testing.assert_array_equal(yb, permute_within_blocks(feat, y, seed, block_of))
    assert moved_any
    with pytest.raises(ValueError):
        permute_within_blocks(feat, y, 0, {})


def test_default_context_is_the_a1_context(tmp_path, workspaces):
    ws = _write_ws(tmp_path, "unit_default", workspaces["null"])
    feat, cols, y = fsp.load_workspace(ws)
    ctx = fsp.make_context(ws, feat, cols, y, N_FOLDS, 1)
    assert ctx.permute is sap.permute
    assert ctx.base_seed == derive_seed("selperm:unit_default")
    assert ctx.select and not ctx.typeI
    assert fsp.default_checkpoint(ws, False, None) == (
        ws / "results" / "selection_aware_null_checkpoint.csv")


def test_default_run_bitwise_unchanged(tmp_path, workspaces, small_family):
    """The original argument set (no fixed-only or block keys) still runs unchanged."""
    ws = _write_ws(tmp_path, "unit_default_run", workspaces["effect"])
    fsp.cmd_run(_run_args(ws, permutations=2))
    ck = ws / "results" / "selection_aware_null_checkpoint.csv"
    got = pd.read_csv(ck, float_precision="round_trip")
    assert list(got.columns) == CHECKPOINT_COLUMNS
    rep = json.loads((ws / "results" / "selection_aware_permutation_fast.json").read_text())
    assert not {"null", "mode", "real_typeI", "block"} & set(rep)
    feat, cols, y = fsp.load_workspace(ws)
    for i in (0, 1):
        o = fsp.original_permutation(ws, feat, cols, y, i, N_FOLDS)
        row = got[got["index"] == i].iloc[0]
        assert row["T_selection_aware"] == o["T_selection_aware"]
        assert row["T_fixed"] == o["T_fixed"]
        assert row["selected"] == o["selected"]


def test_fixed_only_matches_full_run_T_fixed(workspaces):
    feat = workspaces["effect"]
    full = _ctx(feat, "unit_effect", threads=1)
    fixed = PermContext(**{**full.__dict__, "select": False})
    for i in (0, 1, 2):
        a, b = permutation_statistic(full, i), permutation_statistic(fixed, i)
        assert a["T_fixed"] == b["T_fixed"]                     # bitwise
        assert np.isnan(b["T_selection_aware"]) and b["selected"] == "none"
        assert b["n_fits"] == N_FOLDS and b["n_fits_inner"] == 0


def test_block_parallel_equals_serial(workspaces):
    feat = workspaces["effect"]
    base = _ctx(feat, "unit_effect", threads=1)
    ctx = PermContext(**{**base.__dict__,
                         "permute": partial(permute_within_blocks, block_of=_blocks(feat))})
    idx = [0, 1, 2, 3, 4]
    serial = [r for c in iter_permutations(ctx, idx, n_jobs=1) for r in c]
    chunks = list(iter_permutations(ctx, idx, n_jobs=2, chunk_size=2))
    assert [len(c) for c in chunks] == [2, 2, 1]
    par = [r for c in chunks for r in c]
    assert [r["index"] for r in par] == idx
    for s, p in zip(serial, par, strict=True):
        assert s["selected"] == p["selected"]
        assert s["T_selection_aware"] == p["T_selection_aware"]
        assert s["T_fixed"] == p["T_fixed"]


def test_typeI_procedures_known_cases():
    g = np.repeat(np.arange(40), 5).astype(str)
    y = np.repeat(np.r_[np.ones(15), np.zeros(25)], 5).astype(int)
    agg = ple.aggregate
    score = y + 0.01 * np.arange(len(y)) / len(y)
    sep = typeI_procedures(y, score, g, agg, "unit:sep")
    assert sep["window_naive_reject"] == 1 and sep["window_naive_p"] < 1e-10
    assert sep["participant_boot_reject"] == 1 and sep["window_boot_reject"] == 1
    assert sep["participant_boot_lo"] > 0.5
    rev = typeI_procedures(y, 1.0 - score, g, agg, "unit:rev")   # wrong direction
    assert rev["window_naive_p"] > 0.99 and rev["window_naive_reject"] == 0
    assert rev["participant_boot_reject"] == 1 and rev["participant_boot_hi"] < 0.5
    assert typeI_procedures(y, score, g, agg, "unit:sep") == sep   # fixed seed per name
    assert set(sep) == set(TYPEI_COLUMNS)


def test_typeI_columns_are_additive(tmp_path, workspaces, small_family):
    ws = _write_ws(tmp_path, "unit_typeI", workspaces["effect"])
    plain = tmp_path / "plain.csv"
    typed = tmp_path / "typed.csv"
    fsp.cmd_run(_run_args(ws, permutations=4, fixed_only=True, checkpoint=str(plain),
                          out=str(tmp_path / "plain.json")))
    fsp.cmd_run(_run_args(ws, permutations=4, fixed_only=True, typeI=True,
                          checkpoint=str(typed), out=str(tmp_path / "typed.json")))
    a = pd.read_csv(plain, float_precision="round_trip")
    b = pd.read_csv(typed, float_precision="round_trip")
    assert list(a.columns) == CHECKPOINT_COLUMNS
    assert list(b.columns) == CHECKPOINT_COLUMNS + TYPEI_COLUMNS
    assert (a["T_fixed"].to_numpy() == b["T_fixed"].to_numpy()).all()       # bitwise
    assert a["T_selection_aware"].isna().all() and b["T_selection_aware"].isna().all()
    assert b["window_naive_p"].between(0, 1).all()
    assert (b["participant_boot_lo"] <= b["participant_boot_hi"]).all()
    assert (b["window_naive_reject"] == (b["window_naive_p"] <= 0.05)).all()
    excl = (b["participant_boot_lo"] > 0.5) | (b["participant_boot_hi"] < 0.5)
    assert (b["participant_boot_reject"] == excl.astype(int)).all()
    # the columns match a direct call with the per-index seed name
    feat, cols, y = fsp.load_workspace(ws)
    ctx = fsp.make_context(ws, feat, cols, y, N_FOLDS, 1, select=False, typeI=True)
    r0 = permutation_statistic(ctx, 0)
    for c in TYPEI_COLUMNS:
        assert r0[c] == b.loc[b["index"] == 0, c].iloc[0]
    rep = json.loads((tmp_path / "typed.json").read_text())
    assert rep["mode"] == "fixed_only" and rep["real_typeI"]["n_permutations"] == 4
    assert "selection_aware_null" not in rep and "fixed_pipeline_null" in rep
    assert len(read_checkpoint_extras(typed)) == 4
    # resuming keeps the extras of earlier rows
    fsp.cmd_run(_run_args(ws, permutations=5, fixed_only=True, typeI=True,
                          checkpoint=str(typed), out=str(tmp_path / "typed.json")))
    c = pd.read_csv(typed, float_precision="round_trip")
    assert len(c) == 5 and c[TYPEI_COLUMNS].notna().all().all()
    pd.testing.assert_frame_equal(c.iloc[:4], b)


def test_block_run_through_driver(tmp_path, workspaces, small_family):
    feat = workspaces["effect"]
    ws = _write_ws(tmp_path, "unit_block", feat)
    block_of = _blocks(feat)
    (ws / "validation").mkdir()
    pd.DataFrame({"study_id": list(block_of), "sdmvstra": list(block_of.values())}).to_csv(
        ws / "validation" / "participant_metadata.csv", index=False)
    fsp.cmd_run(_run_args(ws, permutations=3, fixed_only=True, block_col="sdmvstra"))
    ck = ws / "results" / "fixed_null_within_sdmvstra_checkpoint.csv"
    rep = json.loads((ws / "results" / "fixed_null_within_sdmvstra_fast.json").read_text())
    assert rep["null"] == "within_sdmvstra" and rep["block"]["n_blocks"] == 3
    assert rep["block"]["perm0_block_positive_counts_preserved"]
    got = read_checkpoint(ck)
    # index 0 by hand: within-block labels with the within seed, then the fixed pipeline
    y = feat["label"].to_numpy()
    yb = permute_within_blocks(feat, y, derive_seed("selperm:unit_block:within:sdmvstra"),
                               block_of)
    folds = sap.make_folds(feat, yb, N_FOLDS, "cv:unit_block:primary")
    r = lazy_analysis(yb, sap.precompute(feat, COLS, folds), SMALL_FAMILY, sap.build,
                      ple.aggregate, SMALL_FIXED, select=False)
    assert got[0][1] == r["T_fixed"]


def test_deadline_in_the_past_computes_nothing(tmp_path, workspaces, small_family):
    ws = _write_ws(tmp_path, "unit_deadline", workspaces["null"])
    out = tmp_path / "r.json"
    fsp.cmd_run(_run_args(ws, permutations=3, fixed_only=True, deadline="2000-01-01T00:00",
                          out=str(out)))
    rep = json.loads(out.read_text())
    assert rep["stopped_on_deadline"] and rep["B_completed"] == 0
    assert rep["execution"]["new_permutations"] == 0


def test_compare_is_bitwise(tmp_path):
    done = {i: (0.5 + i / 100, 0.4 + i / 100, "x") for i in range(5)}
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    write_checkpoint(a, done)
    write_checkpoint(b, done)
    near = dict(done)
    near[3] = (np.nextafter(done[3][0], 1.0), done[3][1], "x")
    write_checkpoint(c, near)

    def ns(other, out):
        return argparse.Namespace(a=str(a), b=str(other), first=0, last=4, label="t",
                                  append=False, out=str(out))
    assert fsp.cmd_compare(ns(b, tmp_path / "eq.csv")) == 0
    assert fsp.cmd_compare(ns(c, tmp_path / "ne.csv")) == 1
    ne = pd.read_csv(tmp_path / "ne.csv")
    assert (~ne["bitwise_equal"]).sum() == 1
