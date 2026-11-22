"""More records per participant on real Depresjon features (Table VI).

For each k, only the first k windows of every participant are kept. The fixed
pipeline of `participant_level_eval.py` (histogram gradient boosting, participant
score = mean window probability) is evaluated with grouped stratified five-fold
cross-validation, and the same analysis is rerun on permuted participant labels,
with folds rebuilt from each permuted labeling.

Reported for each k:
  * participant-level out-of-fold AUROC on the observed labels,
  * the window-naive Mann-Whitney p-value on the observed labels,
  * the permutation p-value, (1 + #{permuted AUROC >= observed}) / (B + 1),
  * Type-I error over the B permuted labelings for the window-naive test, the
    window bootstrap, and the participant bootstrap of stored out-of-fold
    predictions (each at a nominal 5%).

Paper settings: k in {1, 3, 5, 9}, B = 100.

Example:
    python scripts/records_sweep_depresjon.py --workspace reports/real_depresjon \\
        --windows 1 3 5 9 --permutations 100
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from participant_level_eval import aggregate, make_folds, run_cv
from scipy import stats
from simulation_study import _boot_ci
from sklearn.metrics import roc_auc_score

from wearabledata.utils.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05


def first_k_windows(feat: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the first k windows of each participant, in recording order."""
    order = "cutoff_day" if "cutoff_day" in feat.columns else "window_index"
    f = feat.sort_values(["study_id", order], kind="mergesort")
    f = f.groupby("study_id", sort=False).head(k)
    counts = f.groupby("study_id").size()
    keep = counts[counts == k].index
    return f[f["study_id"].isin(keep)].reset_index(drop=True)


def analyse(feat: pd.DataFrame, cols: list[str], y: np.ndarray, seed_name: str,
            boot_seed: int) -> dict:
    """One run of the analysis on labels y: AUROC and three shortcut decisions."""
    folds = make_folds(feat, y, 5, seed_name)
    oof = run_cv(feat, cols, y, folds)
    ok = ~np.isnan(oof)
    pa, ya, _ = aggregate(oof, feat["study_id"].to_numpy(), y)
    pos, neg = oof[ok][y[ok] == 1], oof[ok][y[ok] == 0]
    naive_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    lo_w, hi_w = _boot_ci(y[ok], oof[ok], np.random.default_rng(boot_seed))
    lo_p, hi_p = _boot_ci(ya, pa, np.random.default_rng(boot_seed + 1))
    return {
        "participant_auroc": float(roc_auc_score(ya, pa)),
        "window_naive_p": naive_p,
        "reject_window_naive": naive_p <= ALPHA,
        "reject_window_bootstrap": bool(np.isfinite(lo_w) and (lo_w > 0.5 or hi_w < 0.5)),
        "reject_participant_bootstrap": bool(np.isfinite(lo_p)
                                             and (lo_p > 0.5 or hi_p < 0.5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workspace", default=str(ROOT / "reports" / "real_depresjon"))
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 9])
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--time-budget", type=float, default=None,
                    help="stop after this many seconds; rerun to resume")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ws = Path(a.workspace)
    out = Path(a.out) if a.out else ws / "results" / "records_sweep"
    out.mkdir(parents=True, exist_ok=True)
    feat_all = pd.read_csv(ws / "interim" / "features.csv")
    cols = [c for c in json.loads((ws / "interim" / "feature_columns.json").read_text())
            if c in feat_all.columns]

    start = time.perf_counter()
    summary = []
    for k in a.windows:
        feat = first_k_windows(feat_all, k)
        y = feat["label"].to_numpy().astype(int)
        obs = analyse(feat, cols, y, f"cv:records:{ws.name}:k{k}", derive_seed(f"rboot:k{k}"))

        ckpt = out / f"records_k{k}_permutations.csv"
        rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
        done = {int(r["index"]) for r in rows}
        per = feat.groupby("study_id")["label"].first()
        ids, labs = per.index.to_numpy(), per.to_numpy()
        base = derive_seed(f"records:{ws.name}:k{k}")
        stopped = False
        for i in range(a.permutations):
            if i in done:
                continue
            if a.time_budget and time.perf_counter() - start > a.time_budget:
                stopped = True
                break
            g = np.random.default_rng(base + i)
            mapping = dict(zip(ids, g.permutation(labs), strict=True))
            yp = feat["study_id"].map(mapping).to_numpy().astype(int)
            r = analyse(feat, cols, yp, f"cv:records:{ws.name}:k{k}",
                        derive_seed(f"rboot:k{k}:{i}"))
            rows.append({"index": i, **r})
            pd.DataFrame(rows).to_csv(ckpt, index=False)
        null = pd.DataFrame(rows)
        rec = {"windows_per_participant": k,
               "participants": int(len(ids)), "windows": int(len(feat)),
               "participant_auroc": round(obs["participant_auroc"], 3),
               "window_naive_p_observed": float(f"{obs['window_naive_p']:.2g}"),
               "permutations": int(len(null))}
        if len(null):
            n_ge = int((null["participant_auroc"] >= obs["participant_auroc"]).sum())
            rec["permutation_p"] = round((1 + n_ge) / (len(null) + 1), 4)
            for proc in ("window_naive", "window_bootstrap", "participant_bootstrap"):
                rec[f"typeI_{proc}"] = round(float(null[f"reject_{proc}"].mean()), 3)
        summary.append(rec)
        if stopped:
            print("stopped on time budget; rerun to continue")
            break

    table = pd.DataFrame(summary)
    table.to_csv(out / "records_sweep_summary.csv", index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
