"""NHANES 2011-2014 reader: wrist ActiGraph day summaries and the PHQ-9.

Documented layout (CDC NHANES public-use files, one SAS transport file per
component and cycle; G = 2011-2012, H = 2013-2014):

    data/raw/nhanes/
      DEMO_G.xpt   DEMO_H.xpt     demographics, exam status, survey design
      DPQ_G.xpt    DPQ_H.xpt      PHQ-9 depression screener (MEC interview, 18+)
      PAXHD_G.xpt  PAXHD_H.xpt    physical activity monitor (PAM) header
      PAXDAY_G.xpt PAXDAY_H.xpt   PAM day summaries, one row per wear day (1-9)

Variable meanings are taken from the CDC codebook pages (same URL, .htm):

  * DPQ010-DPQ090 are the nine PHQ-9 items, 0-3 each; 7 = Refused,
    9 = Don't know, missing = not answered. DPQ100 (impairment) is not scored.
  * RIDSTATR 2 = interviewed and MEC examined. RIDAGEYR is topcoded at 80.
  * PAXSTS 1 = the person has PAM data in the summary files. PAXHAND is
    "PAM worn on non-dominant hand": 1 = yes, 2 = no, 9 = don't know.
  * PAXWWMD / PAXSWMD / PAXNWMD / PAXUMD are valid wake-wear / sleep-wear /
    non-wear / unknown minutes in the day; they sum to PAXVMD (valid minutes,
    no QC flag). PAXMTSD is the day sum of MIMS over valid minutes, PAXLXSD the
    day sum of per-minute mean lux over valid minutes, PAXAISMD the count of
    80 Hz samples logged in idle sleep mode, PAXTMD the minutes with data.
  * PAXDAYD runs 1..last for each person. Day 1 starts at the MEC visit and the
    last day ends when recording stops, so both are partial days (in the
    2011-2014 files every day 1 and every final day has PAXTMD < 1440, and
    every other day has PAXTMD = 1440). Both are dropped before any rule is
    applied.

Estimand: association, not prediction. The PHQ-9 is taken at the MEC
visit and the monitor is worn in the days after, so no past-only windowing rule
applies. Participant-relative features are computed over all of the person's
valid days, which the causal transforms in `features.personalization` would
not do.

Identifiers and clock times (SEQN, PAXSENID, PAXMSTD, PAXFTIME, PAXETLDY) and
the day of the week never leave this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ...features.personalization import change_from_baseline
from ..study_ids import study_id
from .contract import PanelContractError, validate_panel

CYCLES = {"G": "2011-2012", "H": "2013-2014"}
COMPONENTS = ("DEMO", "DPQ", "PAXHD", "PAXDAY")

PHQ9_ITEMS = tuple(f"DPQ0{i}0" for i in range(1, 10))
PHQ9_VALID_CODES = (0, 1, 2, 3)
PHQ9_INVALID_CODES = (7, 9)            # 7 = Refused, 9 = Don't know
PHQ9_CUTOFF = 10                       # positive means total >= 10

# Fixed before any outcome was inspected. Do not tune.
MIN_AGE = 18
MIN_WAKE_WEAR_MINUTES = 600            # a valid day has >= 10 h of valid wake wear
MIN_VALID_DAYS = 4                     # a participant needs >= 4 valid days
PAXHAND_UNKNOWN = 9                    # wrist unknown: kept, and counted

MINUTES_PER_DAY = 1440
SAMPLES_PER_MINUTE = 60 * 80           # 80 Hz

EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "DEMO": ("SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "SDMVPSU",
             "SDMVSTRA"),
    "DPQ": ("SEQN", *PHQ9_ITEMS),
    "PAXHD": ("SEQN", "PAXSTS", "PAXHAND"),
    "PAXDAY": ("SEQN", "PAXDAYD", "PAXTMD", "PAXAISMD", "PAXVMD", "PAXMTSD",
               "PAXWWMD", "PAXSWMD", "PAXNWMD", "PAXUMD", "PAXLXSD"),
}

# One window = one valid day. Base features form the population branch.
BASE_FEATURES = (
    "activity_mims_total",       # PAXMTSD
    "activity_mims_per_min",     # PAXMTSD / PAXVMD
    "activity_idle_fraction",    # PAXAISMD / (PAXTMD * 60 * 80)
    "wake_wear_minutes",         # PAXWWMD
    "sleep_wear_minutes",        # PAXSWMD
    "nonwear_minutes",           # PAXNWMD
    "unknown_minutes",           # PAXUMD
    "light_lux_per_min",         # PAXLXSD / PAXVMD
)
# Quality columns; the `relative` candidate branch also includes these names.
QUALITY_FEATURES = ("window_coverage", "valid_days", "cov__wear_fraction")
RELATIVE_PREFIXES = ("pz__", "dev__", "chg__")
CHANGE_BASELINE_DAYS = 3
FEATURE_COLUMNS = (list(BASE_FEATURES) + list(QUALITY_FEATURES)
                   + [f"{p}{c}" for p in RELATIVE_PREFIXES for c in BASE_FEATURES])


@dataclass
class NhanesCohort:
    panel: pd.DataFrame        # study_id, day_index, BASE + QUALITY features, label
    metadata: pd.DataFrame     # one row per participant; never merged into features
    flow: pd.DataFrame         # counts at every exclusion step
    notes: dict = field(default_factory=dict)   # non-exclusion counts (PAXHAND, partial days)


# --------------------------------------------------------------------- I/O
def fix_xport_zeros(df: pd.DataFrame, tol: float = 1e-70) -> pd.DataFrame:
    """`pandas.read_sas` returns 5.397605e-79 where NHANES XPT files store 0.

    Nothing in these files is that small, so any |x| < tol is set to exactly 0.
    """
    out = df.copy()
    num = out.select_dtypes("number").columns
    for c in num:
        x = out[c]
        out[c] = x.mask(x.abs().lt(tol) & x.ne(0), 0.0)
    return out


def read_xpt(path: str | Path) -> pd.DataFrame:
    """Read one SAS transport file. pyreadstat when available, else pandas."""
    try:
        import pyreadstat
    except ImportError:  # pragma: no cover - the project venv has pyreadstat
        pyreadstat = None
    if pyreadstat is not None:
        df, _meta = pyreadstat.read_xport(str(path))
        return df
    df = pd.read_sas(path, format="xport")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return fix_xport_zeros(df)


def load_frames(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the eight public-use files; one frame per component with a `cycle` column.

    Only the documented columns are kept, all as numbers.
    """
    root = Path(raw_dir)
    frames: dict[str, pd.DataFrame] = {}
    for comp in COMPONENTS:
        parts = []
        for cyc in CYCLES:
            p = root / f"{comp}_{cyc}.xpt"
            if not p.exists():
                raise PanelContractError(
                    f"missing {p}. Download {comp}_{cyc}.xpt from the CDC NHANES site "
                    "(scripts/build_nhanes.py --download) into data/raw/nhanes/ without "
                    "renaming.")
            df = read_xpt(p)
            missing = [c for c in EXPECTED_COLUMNS[comp] if c not in df.columns]
            if missing:
                raise PanelContractError(
                    f"{p.name}: documented columns {missing} are absent; the release "
                    "differs from the codebook. Inspect it before adapting this reader.")
            df = df[list(EXPECTED_COLUMNS[comp])].apply(pd.to_numeric, errors="coerce")
            df["cycle"] = cyc
            parts.append(df)
        frames[comp] = pd.concat(parts, ignore_index=True)
    return frames


# ------------------------------------------------------------------- label
def phq9(dpq: pd.DataFrame) -> pd.DataFrame:
    """PHQ-9 total and label. Incomplete if any item is 7, 9, or missing."""
    items = dpq[list(PHQ9_ITEMS)]
    known = items.isna() | items.isin(PHQ9_VALID_CODES + PHQ9_INVALID_CODES)
    if not known.all().all():
        bad = sorted(set(items.where(~known).stack().tolist()))
        raise PanelContractError(f"DPQ items carry undocumented codes {bad[:10]}")
    invalid = items.isna() | items.isin(PHQ9_INVALID_CODES)
    complete = ~invalid.any(axis=1)
    total = items.where(~invalid).sum(axis=1, min_count=len(PHQ9_ITEMS))
    out = pd.DataFrame({"SEQN": dpq["SEQN"].to_numpy(),
                        "phq9_complete": complete.to_numpy(),
                        "phq9_total": total.to_numpy()})
    out["label"] = np.where(out["phq9_complete"],
                            (out["phq9_total"] >= PHQ9_CUTOFF).astype(float), np.nan)
    return out


# -------------------------------------------------------------- day level
def valid_day_mask(day: pd.DataFrame) -> pd.Series:
    return day["PAXWWMD"] >= MIN_WAKE_WEAR_MINUTES


def partial_day_mask(day: pd.DataFrame) -> pd.Series:
    """True for day 1 and for each person's final recorded day (both partial)."""
    last = day.groupby("SEQN")["PAXDAYD"].transform("max")
    return (day["PAXDAYD"] == 1) | (day["PAXDAYD"] == last)


def daily_features(day: pd.DataFrame) -> pd.DataFrame:
    """Per-day features from PAXDAY. Rows are returned in input order."""
    vmd = day["PAXVMD"].where(day["PAXVMD"] > 0)
    tmd = day["PAXTMD"].where(day["PAXTMD"] > 0)
    return pd.DataFrame({
        "SEQN": day["SEQN"].to_numpy(),
        "day_index": (day["PAXDAYD"] - 1).astype(int).to_numpy(),
        "activity_mims_total": day["PAXMTSD"].to_numpy(dtype=float),
        "activity_mims_per_min": (day["PAXMTSD"] / vmd).to_numpy(dtype=float),
        "activity_idle_fraction": (day["PAXAISMD"] / (tmd * SAMPLES_PER_MINUTE)
                                   ).to_numpy(dtype=float),
        "wake_wear_minutes": day["PAXWWMD"].to_numpy(dtype=float),
        "sleep_wear_minutes": day["PAXSWMD"].to_numpy(dtype=float),
        "nonwear_minutes": day["PAXNWMD"].to_numpy(dtype=float),
        "unknown_minutes": day["PAXUMD"].to_numpy(dtype=float),
        "light_lux_per_min": (day["PAXLXSD"] / vmd).to_numpy(dtype=float),
        "window_coverage": (day["PAXVMD"] / MINUTES_PER_DAY).to_numpy(dtype=float),
        "cov__wear_fraction": ((day["PAXWWMD"] + day["PAXSWMD"]) / vmd
                               ).to_numpy(dtype=float),
    })


# ------------------------------------------------------------------ cohort
def _flow_row(step: int, stage: str, description: str, seqn: pd.Series,
              cycle: pd.Series, label: pd.Series | None = None) -> dict:
    row = {"step": step, "stage": stage, "unit": "participants",
           "description": description, "n": int(len(seqn)),
           "n_2011_2012": int((cycle == "G").sum()),
           "n_2013_2014": int((cycle == "H").sum()),
           "n_positive": int(label.sum()) if label is not None else np.nan}
    return row


def build_cohort(frames: dict[str, pd.DataFrame]) -> NhanesCohort:
    """Apply the exclusion steps in order and return panel, metadata, and flow."""
    demo, dpq, hd, day = (frames[c] for c in COMPONENTS)
    for name, df in (("DEMO", demo), ("DPQ", dpq), ("PAXHD", hd)):
        if df["SEQN"].duplicated().any():
            raise PanelContractError(f"{name}: duplicate SEQN rows")
    if day.duplicated(["SEQN", "PAXDAYD"]).any():
        raise PanelContractError("PAXDAY: duplicate (SEQN, PAXDAYD) rows")

    rows = []
    s = demo[["SEQN", "cycle", "RIDSTATR", "RIDAGEYR"]]
    rows.append(_flow_row(0, "demo_records", "DEMO_G + DEMO_H records", s.SEQN, s.cycle))
    s = s[s["RIDSTATR"] == 2]
    rows.append(_flow_row(1, "mec_examined", "interviewed and MEC examined (RIDSTATR = 2)",
                          s.SEQN, s.cycle))
    s = s[s["RIDAGEYR"] >= MIN_AGE]
    rows.append(_flow_row(2, "adult", f"age >= {MIN_AGE} years (RIDAGEYR)",
                          s.SEQN, s.cycle))
    s = s[s["SEQN"].isin(dpq["SEQN"])]
    rows.append(_flow_row(3, "has_dpq", "has a DPQ (PHQ-9) record", s.SEQN, s.cycle))

    scored = phq9(dpq)
    s = s.merge(scored[scored["phq9_complete"]][["SEQN", "phq9_total", "label"]],
                on="SEQN", how="inner")
    rows.append(_flow_row(4, "phq9_complete",
                          "all nine PHQ-9 items answered 0-3 (no 7, 9, or missing)",
                          s.SEQN, s.cycle, s.label))
    s = s[s["SEQN"].isin(hd["SEQN"])]
    rows.append(_flow_row(5, "has_paxhd", "has a PAM header record (PAXHD)",
                          s.SEQN, s.cycle, s.label))
    s = s[s["SEQN"].isin(hd.loc[hd["PAXSTS"] == 1, "SEQN"])]
    rows.append(_flow_row(6, "pam_data", "PAM data available (PAXSTS = 1)",
                          s.SEQN, s.cycle, s.label))

    d = day[day["SEQN"].isin(s["SEQN"])]
    has_day = s["SEQN"].isin(d["SEQN"])
    rows.append(_flow_row(7, "has_day_rows", "has at least one PAXDAY row",
                          s.SEQN[has_day], s.cycle[has_day], s.label[has_day]))
    partial = partial_day_mask(d)
    df_ = d[~partial]
    has_full = s["SEQN"].isin(df_["SEQN"])
    rows.append(_flow_row(8, "has_full_day",
                          "has at least one full day (partial day 1 and final day dropped)",
                          s.SEQN[has_full], s.cycle[has_full], s.label[has_full]))
    dv = df_[valid_day_mask(df_)]
    n_valid = s["SEQN"].map(dv.groupby("SEQN").size()).fillna(0).astype(int)
    k = n_valid >= 1
    rows.append(_flow_row(9, "any_valid_day",
                          f"at least one valid full day (>= {MIN_WAKE_WEAR_MINUTES} valid "
                          "wake-wear minutes, PAXWWMD)",
                          s.SEQN[k], s.cycle[k], s.label[k]))
    k = n_valid >= MIN_VALID_DAYS
    s = s[k].assign(n_valid_days=n_valid[k].to_numpy())
    rows.append(_flow_row(10, "final",
                          f"at least {MIN_VALID_DAYS} valid full days (final cohort)",
                          s.SEQN, s.cycle, s.label))

    dv = dv[dv["SEQN"].isin(s["SEQN"])]
    final_days = d[d["SEQN"].isin(s["SEQN"])]
    for step, stage, desc, dd in (
            (11, "day_rows", "PAXDAY rows of the final cohort", final_days),
            (12, "full_day_rows", "full days of the final cohort (day 1 and final day "
             "dropped)", final_days[~partial_day_mask(final_days)]),
            (13, "windows", "valid full days of the final cohort (one window each)", dv)):
        rows.append({**_flow_row(step, stage, desc, dd.SEQN, dd.cycle), "unit": "days"})
    flow = pd.DataFrame(rows)
    flow["n_excluded"] = np.r_[0, -np.diff(flow["n"].to_numpy())]
    flow.loc[flow["unit"] == "days", "n_excluded"] = np.nan

    hand = s["SEQN"].map(hd.set_index("SEQN")["PAXHAND"])
    notes = {
        "paxhand_9_kept": int((hand == PAXHAND_UNKNOWN).sum()),
        "paxhand_counts_final": {("missing" if pd.isna(k_) else str(int(k_))): int(v)
                                 for k_, v in hand.value_counts(dropna=False)
                                 .sort_index().items()},
        "partial_days_dropped_all_candidates": int(partial.sum()),
        "partial_days_dropped_final_cohort": int(partial_day_mask(final_days).sum()),
    }

    # ---- identifiers: SEQN -> salted study_id, then SEQN is dropped ---------
    sid = {q: study_id(f"SEQN{int(q)}", prefix="N") for q in s["SEQN"]}
    if len(set(sid.values())) != len(sid):
        raise PanelContractError("salted study_id collision")

    feats = daily_features(dv)
    feats = feats.merge(s[["SEQN", "label", "n_valid_days"]], on="SEQN", how="inner")
    feats["valid_days"] = feats.pop("n_valid_days").astype(float)
    feats["label"] = feats["label"].astype(int)
    feats.insert(0, "study_id", feats["SEQN"].map(sid))
    panel = (feats.drop(columns=["SEQN"])
             .sort_values(["study_id", "day_index"]).reset_index(drop=True))
    panel = panel[["study_id", "day_index", *BASE_FEATURES, *QUALITY_FEATURES, "label"]]

    demo_meta = demo.set_index("SEQN")
    meta = pd.DataFrame({
        "study_id": s["SEQN"].map(sid).to_numpy(),
        "cycle": s["cycle"].map(CYCLES).to_numpy(),
        "sdmvstra": s["SEQN"].map(demo_meta["SDMVSTRA"]).astype(int).to_numpy(),
        "sdmvpsu": s["SEQN"].map(demo_meta["SDMVPSU"]).astype(int).to_numpy(),
        "age": s["SEQN"].map(demo_meta["RIDAGEYR"]).astype(int).to_numpy(),
        "sex": s["SEQN"].map(demo_meta["RIAGENDR"]).map({1: "male", 2: "female"}).to_numpy(),
        "n_valid_days": s["n_valid_days"].to_numpy(),
        "label": s["label"].astype(int).to_numpy(),
    }).sort_values("study_id").reset_index(drop=True)

    if len(meta) != panel["study_id"].nunique():
        raise PanelContractError("metadata and panel disagree on the participant count")
    return NhanesCohort(validate_panel(panel, "nhanes"), meta, flow, notes)


def load_cohort(raw_dir: str | Path) -> NhanesCohort:
    return build_cohort(load_frames(raw_dir))


def load_daily_panel(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same (panel, metadata) signature as the other readers."""
    c = load_cohort(raw_dir)
    return c.panel, c.metadata


# ---------------------------------------------------------------- features
def participant_relative(panel: pd.DataFrame, cols: list[str] | tuple[str, ...],
                         group_col: str = "study_id",
                         order_col: str = "day_index") -> pd.DataFrame:
    """pz__, dev__, chg__ over each person's own valid days (all days, D4).

    pz__  (x - person mean) / person sd (ddof 1); NaN when sd = 0.
    dev__ (x - person median) / person IQR; NaN when IQR = 0. The causal
          version adds 1e-9 instead, which turns a zero-IQR minute count into
          values of order 1e9.
    chg__ x - mean of the first 3 valid days, NaN on those days
          (`features.personalization.change_from_baseline`, unchanged).
    """
    df = panel.sort_values([group_col, order_col]).reset_index(drop=True)
    g = df.groupby(group_col, sort=False)
    parts: dict[str, pd.Series] = {}
    for c in cols:
        x = df[c]
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        parts[f"pz__{c}"] = ((x - mu) / sd).where(sd > 0)
    for c in cols:
        x = df[c]
        med = g[c].transform("median")
        iqr = df[group_col].map(g[c].quantile(0.75) - g[c].quantile(0.25))
        parts[f"dev__{c}"] = ((x - med) / iqr).where(iqr > 0)
    chg = change_from_baseline(df, list(cols), CHANGE_BASELINE_DAYS, group_col, order_col)
    rel = pd.DataFrame(parts, index=df.index)
    return pd.concat([df, rel, chg[[f"chg__{c}" for c in cols]]], axis=1)


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Workspace table: study_id, cutoff_day, label, then FEATURE_COLUMNS."""
    df = participant_relative(panel, BASE_FEATURES)
    df = df.rename(columns={"day_index": "cutoff_day"})
    cols = list(FEATURE_COLUMNS)
    out = df[["study_id", "cutoff_day", "label", *cols]]
    if out.columns.duplicated().any():
        raise PanelContractError("duplicate feature columns")
    return out, cols
