"""StudentLife reader.

Documented layout (studentlife.cs.dartmouth.edu):

    dataset/
      sensing/<stream>/<stream>_u##.csv       activity, conversation, dark,
                                              phonelock, phonecharge, gps, wifi, audio
      EMA/response/<Type>/<Type>_u##.json     Stress, Mood, PAM, Sleep, Social, ...
      survey/PHQ-9.csv, PerceivedStressScale.csv, panas.csv, ...
      education/                              deadlines, grades

**The target problem, stated in code rather than buried in a docstring.**
PHQ-9 exists in exactly two waves, pre-term and post-term. There is no daily or
weekly depression trajectory in this dataset. `target` must therefore be chosen
explicitly, and each choice changes what the paper is entitled to say:

  "phq9_post"    one label per participant, ~38 labelled after post-survey
                 attrition. Title must say post-term depression severity.
  "phq9_change"  post - pre above a threshold. Same n, plus regression to the
                 mean on a two-wave difference.
  "ema_stress"   next-day elevated EMA stress. Thousands of windows and real
                 statistical power, but the paper is then about momentary stress
                 and the title must say so.

There is no default. Passing nothing raises, because the most likely way this
study goes wrong is a daily-depression label quietly materialising from a
twice-measured instrument.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..study_ids import study_id
from .contract import PanelContractError, validate_panel

VALID_TARGETS = ("phq9_post", "phq9_change", "ema_stress")
PHQ9_CUTOFF = 10          # conventional moderate-or-above threshold
UID_RE = r"u(\d{2})"

SENSING_STREAMS = ("activity", "conversation", "dark", "phonelock", "phonecharge")


class TargetNotChosen(RuntimeError):
    pass


def _uid_from(path: Path) -> str | None:
    import re

    m = re.search(UID_RE, path.stem)
    return f"u{m.group(1)}" if m else None


def _find_dataset_root(raw_dir: Path) -> Path:
    for cand in [raw_dir, *[p for p in raw_dir.rglob("*") if p.is_dir()]]:
        if (cand / "sensing").is_dir() and (cand / "survey").is_dir():
            return cand
    raise PanelContractError(
        f"could not locate a directory containing sensing/ and survey/ under {raw_dir}. "
        "Unzip the official archive without renaming its directories.")


def _daily_activity(path: Path) -> pd.DataFrame:
    """activity_u##.csv: timestamp, activity inference (0 still,1 walk,2 run,3 unknown)."""
    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    inf_col = next((c for c in df.columns if "inference" in c.lower()), None)
    if ts_col is None or inf_col is None:
        raise PanelContractError(f"{path.name}: expected timestamp and inference columns, "
                                 f"found {list(df.columns)}")
    ts = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    d = pd.DataFrame({"day": ts.dt.floor("D"), "inf": pd.to_numeric(df[inf_col],
                                                                    errors="coerce")})
    d = d.dropna()
    g = d.groupby("day")["inf"]
    out = pd.DataFrame({
        "activity_samples": g.size(),
        "sedentary_fraction": g.apply(lambda s: float((s == 0).mean())),
        "walking_fraction": g.apply(lambda s: float((s == 1).mean())),
        "running_fraction": g.apply(lambda s: float((s == 2).mean())),
    })
    out["activity_total"] = out["activity_samples"] * (
        out["walking_fraction"] + 2 * out["running_fraction"])
    return out.reset_index()


def _daily_duration(path: Path, prefix: str) -> pd.DataFrame:
    """Streams stored as start/end epoch pairs (conversation, dark, phonelock, charge)."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if "start" in c.lower() or "end" in c.lower()]
    if len(cols) < 2:
        raise PanelContractError(f"{path.name}: expected start/end columns, "
                                 f"found {list(df.columns)}")
    start = pd.to_datetime(pd.to_numeric(df[cols[0]], errors="coerce"), unit="s",
                           errors="coerce")
    end = pd.to_datetime(pd.to_numeric(df[cols[1]], errors="coerce"), unit="s",
                         errors="coerce")
    ok = start.notna() & end.notna()
    d = pd.DataFrame({"day": start[ok].dt.floor("D"),
                      "minutes": (end[ok] - start[ok]).dt.total_seconds() / 60.0})
    d = d[(d["minutes"] >= 0) & (d["minutes"] < 24 * 60)]
    g = d.groupby("day")["minutes"]
    return pd.DataFrame({f"{prefix}_count": g.size(),
                         f"{prefix}_minutes": g.sum(),
                         f"{prefix}_mean_minutes": g.mean()}).reset_index()


def _daily_mobility(path: Path) -> pd.DataFrame:
    """GPS -> mobility summaries. Coordinates are consumed here and never returned."""
    df = pd.read_csv(path)
    lat = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lon = next((c for c in df.columns if c.lower() in ("longitude", "lon", "lng")), None)
    ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    if not (lat and lon and ts_col):
        return pd.DataFrame(columns=["day"])
    ts = pd.to_datetime(pd.to_numeric(df[ts_col], errors="coerce"), unit="s", errors="coerce")
    d = pd.DataFrame({"day": ts.dt.floor("D"),
                      "lat": pd.to_numeric(df[lat], errors="coerce"),
                      "lon": pd.to_numeric(df[lon], errors="coerce")}).dropna()
    rows = []
    for day, blk in d.groupby("day"):
        # Local equirectangular projection in km, relative to the day's own centroid.
        lat0 = blk["lat"].mean()
        x = (blk["lon"] - blk["lon"].mean()) * 111.32 * np.cos(np.radians(lat0))
        y = (blk["lat"] - lat0) * 110.57
        rg = float(np.sqrt((x ** 2 + y ** 2).mean()))
        cell = (np.round(x / 0.1).astype(int).astype(str) + "_"
                + np.round(y / 0.1).astype(int).astype(str))
        counts = cell.value_counts(normalize=True)
        rows.append({
            "day": day,
            "radius_of_gyration": rg,
            "location_entropy": float(-(counts * np.log(counts)).sum()),
            "significant_places": int((cell.value_counts() >= 5).sum()),
            "time_at_primary_place": float(counts.iloc[0]) if len(counts) else np.nan,
        })
    return pd.DataFrame(rows)


def _ema_daily(ema_dir: Path, uid: str, kind: str, field_hint: str) -> pd.DataFrame:
    f = ema_dir / kind / f"{kind}_{uid}.json"
    if not f.exists():
        return pd.DataFrame(columns=["day"])
    try:
        records = json.loads(f.read_text())
    except json.JSONDecodeError:
        warnings.warn(f"{f.name}: malformed JSON, skipped", stacklevel=2)
        return pd.DataFrame(columns=["day"])
    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        ts = r.get("resp_time") or r.get("location_time")
        val = next((v for k, v in r.items()
                    if field_hint in k.lower() and isinstance(v, (int, float, str))), None)
        if ts is None or val is None:
            continue
        try:
            rows.append({"day": pd.to_datetime(int(ts), unit="s").floor("D"),
                         "value": float(val)})
        except (ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame(columns=["day"])
    d = pd.DataFrame(rows)
    g = d.groupby("day")["value"]
    return pd.DataFrame({f"{kind.lower()}_mean": g.mean(),
                         f"{kind.lower()}_max": g.max(),
                         f"{kind.lower()}_count": g.size()}).reset_index()


PHQ9_FREQUENCY = {"Not at all": 0, "Several days": 1,
                  "More than half the days": 2, "Nearly every day": 3}
# The tenth PHQ-9 question asks about functional impairment on a different scale and
# is NOT part of the 0-27 severity score. Its answer set is how we exclude it.
PHQ9_IMPAIRMENT = {"Not difficult at all", "Somewhat difficult",
                   "Very difficult", "Extremely difficult"}


def _phq9_labels(survey_dir: Path) -> pd.DataFrame:
    """Score PHQ-9 from the released survey file.

    The release names each column with the full question text rather than Q1..Q9,
    and appends the impairment item, so item columns are identified by their answer
    vocabulary instead of by name. That is stable against rewording and refuses to
    silently score the wrong ten columns.
    """
    f = next((p for p in survey_dir.glob("*.csv") if "phq" in p.name.lower()), None)
    if f is None:
        raise PanelContractError(f"no PHQ-9 survey file found in {survey_dir}")
    df = pd.read_csv(f)
    uid_col = next((c for c in df.columns if c.lower() in ("uid", "u_id")), df.columns[0])
    type_col = next((c for c in df.columns if c.lower() == "type"), None)
    if type_col is None:
        raise PanelContractError(f"{f.name}: no 'type' column distinguishing pre from post")

    item_cols, impairment_cols = [], []
    for c in df.columns:
        if c in (uid_col, type_col):
            continue
        vals = set(df[c].dropna().astype(str).unique())
        if vals and vals <= set(PHQ9_FREQUENCY):
            item_cols.append(c)
        elif vals and vals <= PHQ9_IMPAIRMENT:
            impairment_cols.append(c)

    if len(item_cols) != 9:
        raise PanelContractError(
            f"{f.name}: found {len(item_cols)} scorable PHQ-9 item columns, expected 9. "
            f"Impairment-scale columns detected: {impairment_cols}. Inspect the file "
            "before scoring; a wrong item set silently changes every label.")

    scored = df[item_cols].apply(lambda s: s.map(PHQ9_FREQUENCY))
    # Allow at most one skipped item, otherwise the total is not comparable.
    total = scored.sum(axis=1, min_count=8)
    out = pd.DataFrame({"uid": df[uid_col].astype(str),
                        "wave": df[type_col].astype(str).str.lower(),
                        "phq9": total})
    wide = out.pivot_table(index="uid", columns="wave", values="phq9", aggfunc="first")
    for w in ("pre", "post"):
        if w not in wide.columns:
            wide[w] = np.nan
    return wide.reset_index()


def load_daily_panel(raw_dir: str | Path, target: str | None = None,
                     phq9_cutoff: int = PHQ9_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_panel, participant_metadata). `target` must be chosen explicitly."""
    if target is None:
        raise TargetNotChosen(
            "StudentLife measures PHQ-9 pre- and post-term only; there is no daily "
            f"depression label to default to. Choose one of {VALID_TARGETS} and record "
            "the choice, because it determines what the "
            "title is allowed to say.")
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    base = _find_dataset_root(Path(raw_dir))
    sensing, survey = base / "sensing", base / "survey"
    ema = base / "EMA" / "response"
    phq = _phq9_labels(survey)

    uids = sorted({u for p in (sensing / "activity").glob("activity_u*.csv")
                   if (u := _uid_from(p))})
    if not uids:
        raise PanelContractError(f"no activity_u##.csv files under {sensing / 'activity'}")

    panels, meta = [], []
    for uid in uids:
        parts: list[pd.DataFrame] = []
        act = sensing / "activity" / f"activity_{uid}.csv"
        if act.exists():
            parts.append(_daily_activity(act))
        for stream, prefix in (("conversation", "conversation"), ("dark", "dark"),
                               ("phonelock", "unlock"), ("phonecharge", "charge")):
            p = sensing / stream / f"{stream}_{uid}.csv"
            if p.exists():
                parts.append(_daily_duration(p, prefix))
        gps = sensing / "gps" / f"gps_{uid}.csv"
        if gps.exists():
            parts.append(_daily_mobility(gps))
        for kind, hint in (("Stress", "level"), ("Mood", "happy"), ("Sleep", "hour"),
                           ("PAM", "picture")):
            e = _ema_daily(ema, uid, kind, hint)
            if not e.empty:
                parts.append(e)

        parts = [p for p in parts if not p.empty and "day" in p.columns]
        if not parts:
            continue
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on="day", how="outer")
        merged = merged.sort_values("day").reset_index(drop=True)

        sid = study_id(uid, prefix="S")
        merged["study_id"] = sid
        merged["day_index"] = np.arange(len(merged), dtype=int)

        row = phq[phq["uid"].astype(str).str.contains(uid, na=False)]
        pre = float(row["pre"].iloc[0]) if len(row) else np.nan
        post = float(row["post"].iloc[0]) if len(row) else np.nan

        if target == "phq9_post":
            label = np.nan if np.isnan(post) else float(post >= phq9_cutoff)
        elif target == "phq9_change":
            label = np.nan if (np.isnan(post) or np.isnan(pre)) else float(post - pre >= 5)
        else:  # ema_stress -- next-day elevated stress, strictly forward-looking
            col = next((c for c in merged.columns if c.startswith("stress_mean")), None)
            if col is None:
                continue
            thresh = merged[col].median()
            label = (merged[col].shift(-1) > thresh).astype(float)
            label[merged[col].shift(-1).isna()] = np.nan

        merged["label"] = label
        merged = merged.drop(columns=["day"])          # exact dates stay in the reader
        merged = merged.dropna(subset=["label"])
        if merged.empty:
            continue
        merged["label"] = merged["label"].astype(int)
        panels.append(merged)
        meta.append({"study_id": sid, "n_days": len(merged), "phq9_pre": pre,
                     "phq9_post": post, "target": target})

    if not panels:
        raise PanelContractError(
            f"no participant produced a labelled panel for target={target!r}. For the "
            "PHQ-9 targets this usually means post-survey attrition: check how many "
            "participants actually have a post-term score before proceeding.")

    panel = pd.concat(panels, ignore_index=True)
    metadata = pd.DataFrame(meta)
    n_lab = metadata["study_id"].nunique()
    if target.startswith("phq9") and n_lab < 48:
        warnings.warn(
            f"only {n_lab} of {len(uids)} participants carry a usable {target} label. "
            "Report this attrition before interpreting any result.", stacklevel=2)
    return validate_panel(panel, f"studentlife[{target}]"), metadata
