#!/usr/bin/env python
"""Null recycling: the real-data Type-I error of the fixed-pipeline null.

Each permuted labeling b is an exact draw from the null. Treat it as a pseudo-observed
dataset whose reported statistic is the selection-aware one, s_b = T_selection_aware_b,
and give it a leave-one-out Monte Carlo p-value against the other labelings:

    p(b) = (1 + #{j in I, j != b : R_j >= s_b}) / (|I without b| + 1)

  p_fixed       R = T_fixed of the same checkpoint (denominator B). The brief's formula.
  p_sel         R = T_selection_aware of the same checkpoint (denominator B).
  p_fixed_full  R = T_fixed of a fixed-only checkpoint (e.g. B = 999), when given.

The rejection rate of p_fixed at alpha is the Type-I error of testing a selected model
against the fixed-pipeline null. No model is refitted.

Subcommands
  recycle   one selection checkpoint (+ optional fixed-only checkpoint) -> JSON
  checks    combine recycle JSONs into reports/bigdata/null_recycling_checks.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import selection_aware_permutation as sap
from scipy import stats

from wearabledata.evaluation.fastperm import read_checkpoint

ROOT = Path(__file__).resolve().parents[1]
BIGDATA = ROOT / "reports" / "bigdata"
CI_NOTE = ("Clopper-Pearson 95% interval treating the B rejections as independent. They "
           "share one reference set, so they are exchangeable but not independent, and the "
           "interval is approximate.")


# ------------------------------------------------------------------ core
def loo_pvalues(s: np.ndarray, s_index: np.ndarray, ref: np.ndarray,
                ref_index: np.ndarray) -> np.ndarray:
    """Leave-one-out Monte Carlo p-value of each s_b against the reference stream.

    The reference entry with the same index as b (the same labeling) is left out.
    Ties count as exceedances (>=), as in selection_aware_permutation.mc_p.
    """
    s, ref = np.asarray(s, dtype=float), np.asarray(ref, dtype=float)
    s_index, ref_index = np.asarray(s_index), np.asarray(ref_index)
    if np.isnan(s).any() or np.isnan(ref).any():
        raise ValueError("NaN statistic; drop invalid rows first")
    if len(np.unique(ref_index)) != len(ref_index):
        raise ValueError("duplicate reference index")
    order = np.sort(ref)
    n_ge = len(order) - np.searchsorted(order, s, side="left")
    pos = pd.Index(ref_index).get_indexer(s_index)
    own = pos >= 0
    own_ge = np.zeros(len(s), dtype=bool)
    own_ge[own] = ref[pos[own]] >= s[own]
    n_ref = len(ref) - own.astype(int)
    return (1 + n_ge - own_ge.astype(int)) / (n_ref + 1)


def rejection(p: np.ndarray, alpha: float) -> dict:
    k, n = int((p <= alpha).sum()), int(len(p))
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return {"rate": k / n, "n_reject": k, "B": n,
            "ci": [float(ci.low), float(ci.high)]}


def summary(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
            "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


def recycle(sel: pd.DataFrame, fixed: pd.DataFrame | None = None,
            alpha: float = 0.05) -> tuple[dict, pd.DataFrame]:
    """sel: index, T_selection_aware, T_fixed, selected. fixed: index, T_fixed."""
    sel = sel.dropna(subset=["T_selection_aware", "T_fixed"]).sort_values("index")
    idx = sel["index"].to_numpy()
    t_sel = sel["T_selection_aware"].to_numpy(float)
    t_fix = sel["T_fixed"].to_numpy(float)
    B = len(sel)
    if B < 2:
        raise ValueError("need at least 2 permutations")
    per_b = pd.DataFrame({"index": idx, "T_selection_aware": t_sel, "T_fixed": t_fix,
                          "selected": sel["selected"].to_numpy(),
                          "p_fixed": loo_pvalues(t_sel, idx, t_fix, idx),
                          "p_sel": loo_pvalues(t_sel, idx, t_sel, idx)})
    s_fix, s_sel = summary(t_fix), summary(t_sel)
    diff = t_sel - t_fix
    out: dict = {
        "B": B,
        "indices_are_prefix": bool(np.array_equal(idx, np.arange(B))),
        "alpha": alpha,
        "typeI_fixed_null": rejection(per_b["p_fixed"].to_numpy(), alpha),
        "typeI_sel_null": rejection(per_b["p_sel"].to_numpy(), alpha),
        "ci_note": CI_NOTE,
        "null_fixed": s_fix,
        "null_selection_aware": s_sel,
        "shift": {
            "mean": s_sel["mean"] - s_fix["mean"],
            "p95": s_sel["p95"] - s_fix["p95"],
            "sd_units": (s_sel["mean"] - s_fix["mean"]) / s_fix["sd"],
            "sd_units_se": float(np.std(diff, ddof=1) / math.sqrt(B) / s_fix["sd"]),
            "p95_sd_units": (s_sel["p95"] - s_fix["p95"]) / s_fix["sd"],
            "definition": ("(mean T_selection_aware - mean T_fixed) / SD(T_fixed) over the "
                           "same B labelings; SE from the paired differences, SD(T_fixed) "
                           "treated as known"),
        },
        "share_selected_fixed_pipeline": float(np.mean(per_b["selected"] == sap.FIXED_PIPELINE)),
        "p_value_definition": ("p(b) = (1 + #{j != b : R_j >= T_selection_aware_b}) / "
                               "(#{j != b} + 1); reject if p <= alpha"),
    }
    if fixed is not None:
        fixed = fixed.dropna(subset=["T_fixed"]).sort_values("index")
        f_idx, f_val = fixed["index"].to_numpy(), fixed["T_fixed"].to_numpy(float)
        per_b["p_fixed_full"] = loo_pvalues(t_sel, idx, f_val, f_idx)
        common = pd.merge(sel[["index", "T_fixed"]], fixed[["index", "T_fixed"]],
                          on="index", suffixes=("_sel", "_fix"))
        a = common["T_fixed_sel"].to_numpy(float)
        b = common["T_fixed_fix"].to_numpy(float)
        out["fixed_only_checkpoint"] = {
            "B": int(len(fixed)),
            "typeI_fixed_null": rejection(per_b["p_fixed_full"].to_numpy(), alpha),
            "null_fixed": summary(f_val),
            "shift_sd_units_vs_this_null": (s_sel["mean"] - float(np.mean(f_val)))
                                           / float(np.std(f_val, ddof=1)),
            "n_common_indices": int(len(common)),
            "n_T_fixed_bitwise_equal": int(sum(x.hex() == y.hex() for x, y in
                                               zip(a.tolist(), b.tolist(), strict=True))),
            "max_abs_diff_T_fixed": float(np.max(np.abs(a - b))) if len(common) else None,
            "reading": ("p_fixed_full uses every fixed-only labeling except b itself, as an "
                        "analyst with a B = 999 fixed-pipeline null would"),
        }
    return out, per_b


def read_stream(path: Path) -> pd.DataFrame:
    """A checkpoint as a frame, floats read round-trip exact (fastperm.read_checkpoint)."""
    done = read_checkpoint(Path(path))
    if not done:
        raise SystemExit(f"{path}: missing or empty")
    return pd.DataFrame([(i, *v) for i, v in sorted(done.items())],
                        columns=["index", "T_selection_aware", "T_fixed", "selected"])


# ------------------------------------------------------------------ cli
def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def cmd_recycle(a: argparse.Namespace) -> int:
    sel = read_stream(Path(a.selection))
    fixed = read_stream(Path(a.fixed)) if a.fixed else None
    res, per_b = recycle(sel, fixed, a.alpha)
    report = {"label": a.label, "selection_checkpoint": _rel(a.selection),
              "fixed_checkpoint": _rel(a.fixed) if a.fixed else None,
              "data_source": a.data_source, **res}
    fo = res.get("fixed_only_checkpoint")
    if fo and fo["n_T_fixed_bitwise_equal"] != fo["n_common_indices"]:
        n_diff = fo["n_common_indices"] - fo["n_T_fixed_bitwise_equal"]
        print(f"WARNING: T_fixed differs on {n_diff} of {fo['n_common_indices']} common indices")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if a.per_b:
        per_b.to_csv(a.per_b, index=False)
    print(json.dumps({k: report[k] for k in ("label", "B", "typeI_fixed_null",
                                             "typeI_sel_null", "shift")}, indent=2))
    return 0


def _checks_entry(path: Path) -> dict:
    r = json.loads(Path(path).read_text())
    return {"B": r["B"], "typeI_fixed_null": r["typeI_fixed_null"]["rate"],
            "ci": r["typeI_fixed_null"]["ci"], "typeI_sel_null": r["typeI_sel_null"]["rate"],
            "shift_sd_units": r["shift"]["sd_units"]}


def cmd_checks(a: argparse.Namespace) -> int:
    checks = {"nhanes": _checks_entry(a.nhanes), "synthetic_n40": _checks_entry(a.synthetic_n40)}
    for item in a.extra or []:
        name, path = item.split("=", 1)
        checks[name] = _checks_entry(Path(path))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recycle", help="Type-I of the fixed null from one selection checkpoint")
    r.add_argument("--selection", required=True,
                   help="checkpoint with T_selection_aware and T_fixed per index")
    r.add_argument("--fixed", default=None, help="optional fixed-only checkpoint, same null")
    r.add_argument("--label", required=True)
    r.add_argument("--data-source", default="")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--out", required=True)
    r.add_argument("--per-b", default=None, help="optional CSV of the per-labeling p-values")
    r.set_defaults(func=cmd_recycle)
    c = sub.add_parser("checks", help="write reports/bigdata/null_recycling_checks.json")
    c.add_argument("--nhanes", required=True)
    c.add_argument("--synthetic-n40", required=True)
    c.add_argument("--extra", nargs="*", help="name=path of further recycle JSONs")
    c.add_argument("--out", default=str(BIGDATA / "null_recycling_checks.json"))
    c.set_defaults(func=cmd_checks)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
