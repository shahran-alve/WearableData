"""Block files and stratified subsamples for the NHANES analyses.

blocks      Writes <workspace>/validation/blocks.csv with one row per participant:
              sdmvstra     masked variance stratum (29 blocks)
              strata_psu   stratum by primary sampling unit (61 blocks)
              sex_age      sex by age band, 18-39, 40-59, 60-80 (6 blocks)
            Use with fast_selection_permutation.py --block-file and --block-col.

subsamples  Draws one subsample per size, stratified on the label so that the
            prevalence matches the full cohort, and writes each as a workspace
            reports/<name>_n<size>/ that the permutation scripts read directly.
            Paper sizes: 100, 250, 500, 1000, 2000, 4000.

The age bands of the sex-by-age blocks are not stated in the paper; the bands
here are a choice made for this repository.

Examples:
    python scripts/prepare_nhanes_analyses.py blocks --workspace reports/real_nhanes
    python scripts/prepare_nhanes_analyses.py subsamples --workspace reports/real_nhanes
    (default sizes are the paper sizes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from wearabledata.utils.seeds import derive_seed

ROOT = Path(__file__).resolve().parents[1]
AGE_BANDS = [(18, 39), (40, 59), (60, 80)]


def age_band(age: float) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    raise ValueError(f"age {age} outside {AGE_BANDS}")


def cmd_blocks(a: argparse.Namespace) -> int:
    ws = Path(a.workspace)
    meta = pd.read_csv(ws / "validation" / "participant_metadata.csv")
    blocks = pd.DataFrame({
        "study_id": meta["study_id"],
        "sdmvstra": meta["sdmvstra"],
        "strata_psu": meta["sdmvstra"].astype(str) + "_" + meta["sdmvpsu"].astype(str),
        "sex_age": meta["sex"].astype(str) + "_" + meta["age"].map(age_band),
    })
    out = ws / "validation" / "blocks.csv"
    blocks.to_csv(out, index=False)
    for col in ("sdmvstra", "strata_psu", "sex_age"):
        print(f"{col:11s} {blocks[col].nunique():3d} blocks")
    print(f"wrote {out}")
    return 0


def cmd_subsamples(a: argparse.Namespace) -> int:
    ws = Path(a.workspace)
    feat = pd.read_csv(ws / "interim" / "features.csv")
    cols_file = ws / "interim" / "feature_columns.json"
    meta = pd.read_csv(ws / "validation" / "participant_metadata.csv")
    per = feat.groupby("study_id")["label"].first()
    pos, neg = per[per == 1].index.to_numpy(), per[per == 0].index.to_numpy()
    prevalence = len(pos) / len(per)

    for n in a.sizes:
        if n >= len(per):
            raise SystemExit(f"size {n} is not smaller than the cohort ({len(per)})")
        g = np.random.default_rng(derive_seed(f"subsample:{ws.name}:{n}"))
        n_pos = int(round(prevalence * n))
        keep = set(g.choice(pos, n_pos, replace=False)) | set(
            g.choice(neg, n - n_pos, replace=False))
        sub = ws.parent / f"{ws.name}_n{n}"
        (sub / "interim").mkdir(parents=True, exist_ok=True)
        (sub / "validation").mkdir(parents=True, exist_ok=True)
        feat[feat["study_id"].isin(keep)].to_csv(sub / "interim" / "features.csv",
                                                  index=False)
        (sub / "interim" / "feature_columns.json").write_text(cols_file.read_text())
        meta[meta["study_id"].isin(keep)].to_csv(
            sub / "validation" / "participant_metadata.csv", index=False)
        (sub / "subsample.json").write_text(json.dumps({
            "source_workspace": str(ws), "participants": n, "positives": n_pos,
            "seed_name": f"subsample:{ws.name}:{n}"}, indent=2))
        print(f"n={n:5d}  positives={n_pos:4d}  -> {sub}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blocks", help="write validation/blocks.csv")
    b.add_argument("--workspace", default=str(ROOT / "reports" / "real_nhanes"))
    b.set_defaults(func=cmd_blocks)
    s = sub.add_parser("subsamples", help="write stratified subsample workspaces")
    s.add_argument("--workspace", default=str(ROOT / "reports" / "real_nhanes"))
    s.add_argument("--sizes", type=int, nargs="+",
                   default=[100, 250, 500, 1000, 2000, 4000])
    s.set_defaults(func=cmd_subsamples)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
