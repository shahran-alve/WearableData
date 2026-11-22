"""Per-fold nested selection, analysis A'.

References are imported, not copied: the original analysis() in
scripts/selection_aware_permutation.py and lazy_analysis / permutation_statistic in
fastperm.py. A 30-participant synthetic workspace and a 4-candidate family keep the
file fast.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import fast_selection_permutation as fsp  # noqa: E402
import participant_level_eval as ple  # noqa: E402
import perfold_selection_permutation as pfs  # noqa: E402
import selection_aware_permutation as sap  # noqa: E402
from make_synthetic_workspace import generate  # noqa: E402

from wearabledata.evaluation.fastperm import (  # noqa: E402
    lazy_analysis,
    permutation_statistic,
    read_checkpoint,
)
from wearabledata.evaluation.perfold import (  # noqa: E402
    PERFOLD_COLUMNS,
    SKIP,
    iter_perfold,
    perfold_analysis,
    perfold_permutation_statistic,
    read_perfold_checkpoint,
    recycled_typeI,
    write_perfold_checkpoint,
)

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
K1_FAMILY = [c for c in SMALL_FAMILY if c["name"] == SMALL_FIXED]
N_FOLDS = 5


def _feat(effect: float) -> pd.DataFrame:
    return generate(COLS, n_participants=30, n_positive=12, n_windows=300,
                    min_windows=8, max_windows=14, effect=effect, seed_name="unit_perfold")


@pytest.fixture(scope="module", autouse=True)
def one_thread():
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


def _design(feat: pd.DataFrame, name: str, index: int):
    y = feat["label"].to_numpy()
    yb = y if index < 0 else sap.permute(feat, y, fsp.perm_base_seed(Path(name)) + index)
    folds = sap.make_folds(feat, yb, N_FOLDS, f"cv:{name}:primary")
    return yb, folds, sap.precompute(feat, COLS, folds)


def _pooled_auc(y, oof, groups) -> float:
    ok = ~np.isnan(oof)
    pa, ya, _ = ple.aggregate(oof[ok], groups[ok], y[ok])
    return float(roc_auc_score(ya, pa))


# ------------------------------------------------------------- K = 1
@pytest.mark.parametrize("regime", ["null", "effect"])
def test_k1_perfold_equals_global_equals_fixed(workspaces, monkeypatch, regime):
    monkeypatch.setattr(sap, "CANDIDATES", K1_FAMILY)
    monkeypatch.setattr(sap, "FIXED_PIPELINE", SMALL_FIXED)
    feat = workspaces[regime]
    for index in (-1, 0, 1, 2):
        yb, folds, pre = _design(feat, f"unit_{regime}", index)
        r = perfold_analysis(yb, pre, K1_FAMILY, sap.build, ple.aggregate, SMALL_FIXED,
                             with_global=True)
        lz = lazy_analysis(yb, pre, K1_FAMILY, sap.build, ple.aggregate, SMALL_FIXED)
        orig = sap.analysis(feat, COLS, yb, folds, pre=pre)
        values = [r["T_perfold"], r["T_global"], r["T_fixed"], lz["T_selection_aware"],
                  lz["T_fixed"], orig["T_selection_aware"], orig["T_fixed"]]
        assert not np.isnan(values[0])
        assert all(v == values[0] for v in values), values     # bitwise
        fm = [r["T_perfold_foldmean"], r["T_global_foldmean"], r["T_fixed_foldmean"]]
        assert not np.isnan(fm[0]) and fm[0] == fm[1] == fm[2]
        assert r["fold_test_auc"]["perfold"] == r["fold_test_auc"]["fixed"]
        assert r["selected_global"] == lz["selected_by_inner"] == SMALL_FIXED
        assert all(p == SMALL_FIXED for p in r["perfold_selected"])
        # one outer fit per fold, reused by the fixed and global statistics
        assert r["n_fits_outer_perfold"] == N_FOLDS
        assert r["n_fits_outer_fixed_extra"] == 0 and r["n_fits_outer_global_extra"] == 0


# ------------------------------------------------------------- K = 4
def _reference(yb, pre, family):
    """Per-fold selection written out directly from the definition in the brief."""
    groups = pre["groups"]
    by = {c["name"]: c for c in family}
    picks, fold_scores, fold_auc = [], [], []
    oof = np.full(len(yb), np.nan)
    for fm in pre["masks"]:
        tr, va, te = fm["train"], fm["validation"], fm["test"]
        if len(set(yb[tr])) < 2 or len(set(yb[tr | va])) < 2:
            picks.append(SKIP)
            fold_scores.append(None)
            continue
        scores = {}
        for spec in family:
            X = pre["branches"][spec["branch"]]
            p = sap.build(spec, 0).fit(X[tr], yb[tr]).predict_proba(X[va])[:, 1]
            pa, ya, _ = ple.aggregate(p, groups[va], yb[va])
            scores[spec["name"]] = float(roc_auc_score(ya, pa))
        best = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))[-1][0]
        picks.append(best)
        fold_scores.append(scores)
        X = pre["branches"][by[best]["branch"]]
        oof[te] = sap.build(by[best], 0).fit(X[tr | va], yb[tr | va]).predict_proba(X[te])[:, 1]
        pa, ya, _ = ple.aggregate(oof[te], groups[te], yb[te])
        fold_auc.append(float(roc_auc_score(ya, pa)))
    return picks, fold_scores, _pooled_auc(yb, oof, groups), fold_auc


@pytest.mark.parametrize("regime", ["null", "effect"])
def test_small_family_matches_lazy_and_reference(workspaces, regime):
    feat = workspaces[regime]
    n_mixed = 0
    for index in (-1, 0, 1, 2):
        yb, folds, pre = _design(feat, f"unit_{regime}", index)
        r = perfold_analysis(yb, pre, SMALL_FAMILY, sap.build, ple.aggregate, SMALL_FIXED,
                             with_global=True)
        lz = lazy_analysis(yb, pre, SMALL_FAMILY, sap.build, ple.aggregate, SMALL_FIXED)
        # the global side is the lazy path, bitwise
        assert r["selected_global"] == lz["selected_by_inner"]
        assert r["inner_mean"] == lz["inner_mean"]
        assert r["T_global"] == lz["T_selection_aware"]
        assert r["T_fixed"] == lz["T_fixed"]
        # the per-fold side is the definition, bitwise
        picks, scores, t_ref, fold_auc = _reference(yb, pre, SMALL_FAMILY)
        assert r["perfold_selected"] == picks
        assert r["fold_inner"] == scores
        assert r["T_perfold"] == t_ref
        assert r["fold_test_auc"]["perfold"] == fold_auc
        assert r["T_perfold_foldmean"] == float(np.mean(fold_auc))
        assert r["n_fallback_folds"] == 0
        # fit accounting
        valid = [k for k, p in enumerate(picks) if p != SKIP]
        assert r["n_fits_inner"] == len(SMALL_FAMILY) * len(valid)
        assert r["n_fits_outer_perfold"] == len(valid)
        assert r["n_fits_outer_fixed_extra"] == sum(picks[k] != SMALL_FIXED for k in valid)
        assert r["n_fits_outer_global_extra"] == sum(
            r["selected_global"] not in (picks[k], SMALL_FIXED) for k in valid)
        n_mixed += len({picks[k] for k in valid}) > 1
    assert n_mixed > 0, "no case with different picks across folds; test too weak"


def test_fallback_and_skip(workspaces):
    feat = workspaces["null"]
    y = feat["label"].to_numpy()
    lab = feat.groupby("study_id")["label"].first()
    folds = sap.make_folds(feat, y, N_FOLDS, "cv:unit_null:primary")
    # fold 0: validation keeps only negatives (its positives move to train) -> fallback
    pos0 = [i for i in folds[0]["validation"] if lab[i] == 1]
    folds[0]["validation"] = [i for i in folds[0]["validation"] if lab[i] == 0]
    folds[0]["train"] = sorted(folds[0]["train"] + pos0)
    # fold 2: train and validation keep only negatives -> skipped
    for part in ("train", "validation"):
        folds[2][part] = [i for i in folds[2][part] if lab[i] == 0]
    pre = sap.precompute(feat, COLS, folds)
    r = perfold_analysis(y, pre, SMALL_FAMILY, sap.build, ple.aggregate, SMALL_FIXED)
    assert r["fold_inner"][0] == {}
    assert r["fold_inner"][2] is None
    assert r["n_fallback_folds"] == 1
    assert r["perfold_selected"][0] == r["selected_global"]
    assert r["perfold_selected"][2] == SKIP
    assert r["n_fits_outer_perfold"] == N_FOLDS - 1
    assert not np.isnan(r["T_perfold"])


# ------------------------------------------------------------- driver
def test_permutation_labeling_is_fastperm(workspaces, small_family):
    feat = workspaces["null"]
    ctx = fsp.make_context(Path("unit_null"), feat, COLS, feat["label"].to_numpy(),
                           N_FOLDS, None)
    for i in (0, 1, 2):
        pr = perfold_permutation_statistic(ctx, i, with_global=True)
        fr = permutation_statistic(ctx, i)
        assert pr["T_fixed"] == fr["T_fixed"]
        assert pr["T_global"] == fr["T_selection_aware"]
        assert pr["selected_global"] == fr["selected"]
        assert pr["index"] == i


def test_parallel_equals_serial(workspaces, small_family):
    feat = workspaces["effect"]
    ctx = fsp.make_context(Path("unit_effect"), feat, COLS, feat["label"].to_numpy(),
                           N_FOLDS, 1)
    idx = [0, 1, 2, 3]
    serial = [r for c in iter_perfold(ctx, idx, n_jobs=1) for r in c]
    par = [r for c in iter_perfold(ctx, idx, n_jobs=2, chunk_size=3) for r in c]
    assert [r["index"] for r in par] == idx
    for s, p in zip(serial, par, strict=True):
        for c in ("T_perfold", "T_fixed", "T_perfold_foldmean", "T_fixed_foldmean",
                  "selected_global", "perfold_selected", "n_fits"):
            assert s[c] == p[c]
    assert all(r["pid"] != os.getpid() for r in par)


def _write_ws(root: Path, name: str, feat: pd.DataFrame) -> Path:
    ws = root / name
    (ws / "interim").mkdir(parents=True)
    feat.to_csv(ws / "interim" / "features.csv", index=False)
    (ws / "interim" / "feature_columns.json").write_text(json.dumps(COLS))
    return ws


def _run_args(ws: Path, **kw) -> argparse.Namespace:
    base = dict(workspace=str(ws), permutations=2, folds=N_FOLDS, n_jobs=1, chunk_size=None,
                threads_per_task=1, deadline=None, checkpoint=None, perm_log=None, out=None,
                with_global=False, observed_from=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_resume_observed_and_report(tmp_path, workspaces, small_family):
    ws = _write_ws(tmp_path, "unit_run", workspaces["null"])
    ckpt = ws / "results" / "perfold_null_checkpoint.csv"
    log = tmp_path / "permlog.csv"
    pfs.cmd_run(_run_args(ws, permutations=2, perm_log=str(log)))
    first = pd.read_csv(ckpt, float_precision="round_trip")
    assert list(first.columns) == PERFOLD_COLUMNS
    pfs.cmd_run(_run_args(ws, permutations=3, perm_log=str(log)))
    rep = json.loads((ws / "results" / "perfold_null.json").read_text())
    assert rep["execution"]["new_permutations"] == 1 and rep["B_completed"] == 3
    done = read_perfold_checkpoint(ckpt)
    for _, r in first.iterrows():
        assert done[int(r["index"])]["T_perfold"] == r["T_perfold"]
    feat, cols, y = fsp.load_workspace(ws)
    direct = perfold_permutation_statistic(fsp.make_context(ws, feat, cols, y, N_FOLDS, 1), 2)
    assert done[2]["T_perfold"] == direct["T_perfold"]
    assert done[2]["perfold_selected"] == direct["perfold_selected"]
    assert len(pd.read_csv(log)) == 3
    write_perfold_checkpoint(tmp_path / "rt.csv", done)
    back = read_perfold_checkpoint(tmp_path / "rt.csv")
    assert back.keys() == done.keys()
    for i in done:
        for c, v in done[i].items():
            assert (back[i][c] == v) or (isinstance(v, float) and np.isnan(v)
                                         and np.isnan(back[i][c]))

    # the global-selection driver on the same workspace: the global checkpoint the report joins on
    gck = tmp_path / "global.csv"
    fsp.cmd_run(argparse.Namespace(
        workspace=str(ws), permutations=3, folds=N_FOLDS, n_jobs=1, threads_per_task=1,
        chunk_size=None, time_budget=None, sequential=False, h=10,
        sequential_stat="selection_aware", checkpoint=str(gck), out=str(tmp_path / "g.json")))
    obs = tmp_path / "obs.json"
    pfs.cmd_observed(argparse.Namespace(workspace=str(ws), folds=N_FOLDS, threads_per_task=1,
                                        compare_observed=None, out=str(obs)))
    ob = json.loads(obs.read_text())
    folds = sap.make_folds(feat, y, N_FOLDS, fsp.fold_seed_name(ws))
    orig = sap.analysis(feat, cols, y, folds, pre=sap.precompute(feat, cols, folds))
    assert ob["T_global"] == orig["T_selection_aware"]
    assert ob["T_fixed"] == orig["T_fixed"]
    assert ob["selected_global"] == orig["selected_by_inner"]
    out = tmp_path / "report.json"
    pfs.cmd_report(argparse.Namespace(checkpoint=str(ckpt), observed=str(obs),
                                      global_checkpoint=str(gck), fixed_checkpoint=None,
                                      out=str(out)))
    r = json.loads(out.read_text())
    c = r["consistency_with_global_run"]
    assert c["T_fixed_bitwise_equal_to_global_checkpoint"] == {
        "source": str(gck), "n_equal": 3, "n": 3}
    assert c["selected_global_equal_to_global_checkpoint"] == {"n_equal": 3, "n": 3}
    g = read_checkpoint(gck)
    fam = r["families"]["pooled"]
    assert fam["global_source"] == str(gck) and "foldmean" not in r["families"]
    assert fam["tests"]["global"]["n_ge"] == sum(g[i][0] >= ob["T_global"] for i in g)

    # with --with-global the checkpoint carries T_global (equal to the global-selection driver's,
    # bitwise) and the fold-mean family
    ck2 = tmp_path / "with_global.csv"
    pfs.cmd_run(_run_args(ws, permutations=3, checkpoint=str(ck2), with_global=True,
                          out=str(tmp_path / "wg.json")))
    pfs.cmd_report(argparse.Namespace(checkpoint=str(ck2), observed=str(obs),
                                      global_checkpoint=str(gck), fixed_checkpoint=None,
                                      out=str(out)))
    r = json.loads(out.read_text())
    assert r["families"]["pooled"]["global_source"] == "this checkpoint"
    assert r["consistency_with_global_run"]["T_global_bitwise_equal_to_global_checkpoint"] == {
        "n_equal": 3, "n": 3}
    fm = r["families"]["foldmean"]["tests"]
    done2 = read_perfold_checkpoint(ck2)
    assert fm["perfold"]["n_ge"] == sum(v["T_perfold_foldmean"] >= ob["T_perfold_foldmean"]
                                        for v in done2.values())
    assert all(done2[i]["T_perfold"] == done[i]["T_perfold"] for i in done)


def test_recycled_typeI_known_answer():
    ref = np.random.default_rng(0).permutation(1000) / 1000.0
    same = recycled_typeI(ref, ref)
    assert same["rejections"] == 50 and same["rate"] == 0.05      # exchangeable: alpha
    lo, hi = same["ci_95_clopper_pearson"]
    assert lo < 0.05 < hi
    assert recycled_typeI(ref + 2.0, ref)["rate"] == 1.0            # always above the null
    assert recycled_typeI(ref - 2.0, ref)["rejections"] == 0


# ------------------------------------------------------------------ --block-col
def test_block_col_matches_global_within_labeling(tmp_path, workspaces, small_family):
    """With --block-col, permutation i is the labeling of the global-selection within-block run:
    T_fixed, T_global and the global pick equal fastperm.permutation_statistic on the
    fsp within-block context, bitwise. Without it the run is the unrestricted one."""
    from wearabledata.evaluation.fastperm import permute_within_blocks
    from wearabledata.utils.seeds import derive_seed

    feat = workspaces["effect"]
    ws = _write_ws(tmp_path, "unit_pf_block", feat)
    ids = sorted(feat["study_id"].unique())
    block_of = {sid: 90 + (j % 3) for j, sid in enumerate(ids)}
    (ws / "validation").mkdir()
    pd.DataFrame({"study_id": list(block_of), "sdmvstra": list(block_of.values())}).to_csv(
        ws / "validation" / "participant_metadata.csv", index=False)
    pfs.cmd_run(_run_args(ws, permutations=3, with_global=True, block_col="sdmvstra"))
    ck = ws / "results" / "perfold_null_within_sdmvstra_checkpoint.csv"
    rep = json.loads((ws / "results" / "perfold_null_within_sdmvstra.json").read_text())
    assert rep["null"] == "within_sdmvstra"
    assert rep["permutation_seed_name"] == "selperm:unit_pf_block:within:sdmvstra"
    assert rep["block"]["n_blocks"] == 3
    assert rep["block"]["perm0_block_positive_counts_preserved"]
    got = read_perfold_checkpoint(ck)
    assert sorted(got) == [0, 1, 2]

    y = feat["label"].to_numpy()
    fctx = fsp.make_context(ws, feat, COLS, y, N_FOLDS, 1,
                            block_of=fsp.load_blocks(ws, "sdmvstra", None),
                            block_col="sdmvstra")
    base = derive_seed("selperm:unit_pf_block:within:sdmvstra")
    assert fctx.base_seed == base
    per = pd.DataFrame({"g": feat["study_id"], "y": y}).groupby("g")["y"].first()
    blk = per.index.map(block_of)
    moved = False
    for i in range(3):
        fr = permutation_statistic(fctx, i)
        assert got[i]["T_fixed"] == fr["T_fixed"]                  # bitwise
        assert got[i]["T_global"] == fr["T_selection_aware"]
        assert got[i]["selected_global"] == fr["selected"]
        yb = permute_within_blocks(feat, y, base + i, block_of)
        pb = pd.DataFrame({"g": feat["study_id"], "y": yb}).groupby("g")["y"].first()
        assert (per.groupby(blk).sum() == pb.groupby(blk).sum()).all()
        moved |= bool((pb != per).any())
        unrestricted = sap.permute(feat, y, fsp.perm_base_seed(ws) + i)
        assert not np.array_equal(yb, unrestricted)
    assert moved

    # default path: no block column gives the unrestricted labeling and names
    pfs.cmd_run(_run_args(ws, permutations=2, with_global=True))
    rep0 = json.loads((ws / "results" / "perfold_null.json").read_text())
    assert rep0["null"] == "unrestricted" and "block" not in rep0
    d0 = read_perfold_checkpoint(ws / "results" / "perfold_null_checkpoint.csv")
    uctx = fsp.make_context(ws, feat, COLS, y, N_FOLDS, 1)
    for i in (0, 1):
        assert d0[i]["T_fixed"] == permutation_statistic(uctx, i)["T_fixed"]
