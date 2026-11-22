"""More records per participant at a fixed cohort size (Table III and Fig. 4).

The cohort size is held at 40 participants and every participant contributes the
same number of windows k. Each replicate uses the simulation design of
`simulation_study.py` and applies the same four procedures: window-naive
Mann-Whitney test, window bootstrap, participant bootstrap of stored out-of-fold
predictions, and the participant-label permutation test.

Paper settings:
    delta = 0    k in {10, 30, 100, 300, 1000}, 200 replicates (100 at k = 1000)
    delta = 0.3  k in {10, 30, 100, 300}, 100 replicates

Replicates are checkpointed; rerunning resumes where the last run stopped.

Example:
    python scripts/records_simulation.py --delta 0 --windows 10 30 100 300 \\
        --replicates 200
    python scripts/records_simulation.py --delta 0 --windows 1000 --replicates 100
    python scripts/records_simulation.py --delta 0.3 --windows 10 30 100 300 \\
        --replicates 100
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from scipy import stats
from simulation_study import one_replicate

ROOT = Path(__file__).resolve().parents[1]
PROCEDURES = ("window_naive", "window_bootstrap", "participant_bootstrap", "permutation")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, k), g in df.groupby(["delta", "windows_per_participant"]):
        rec = {"delta": float(d), "windows_per_participant": int(k),
               "windows_in_dataset": int(g["n_windows"].iloc[0]),
               "n_replicates": int(len(g)),
               "mean_participant_auroc": round(float(g["participant_auroc"].mean()), 3)}
        for proc in PROCEDURES:
            col = f"reject_{proc}"
            hits, n = int(g[col].sum()), len(g)
            lo, hi = stats.binomtest(hits, n).proportion_ci(confidence_level=0.95)
            rec[f"{proc}_rate"] = round(hits / n, 3)
            rec[f"{proc}_ci_lo"] = round(float(lo), 3)
            rec[f"{proc}_ci_hi"] = round(float(hi), 3)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["delta", "windows_per_participant"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--participants", type=int, default=40)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--windows", type=int, nargs="+", default=[10, 30, 100, 300])
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--time-budget", type=float, default=None,
                    help="stop after this many seconds; rerun to resume")
    ap.add_argument("--out", default=str(ROOT / "reports" / "records_simulation"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "records_replicates.csv"
    rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
    done = {(float(r["delta"]), int(r["windows_per_participant"]), int(r["seed"]))
            for r in rows}

    start = time.perf_counter()
    for k in a.windows:
        for s in range(a.replicates):
            seed = 100_000 * k + 1_000 * s + int(round(a.delta * 100)) + a.participants
            if (a.delta, k, seed) in done:
                continue
            if a.time_budget and time.perf_counter() - start > a.time_budget:
                print("stopped on time budget; rerun to continue")
                break
            r = one_replicate(a.participants, a.delta, seed, a.permutations,
                              windows_lo=k, windows_hi=k)
            if not r:
                continue
            r["windows_per_participant"] = k
            rows.append(r)
            done.add((a.delta, k, seed))
            pd.DataFrame(rows).to_csv(ckpt, index=False)

    if not rows:
        print("no replicates yet")
        return 0
    summary = summarize(pd.DataFrame(rows))
    summary.to_csv(out / "records_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
