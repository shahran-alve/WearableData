"""Workspace construction for Depresjon, StudentLife, and the synthetic fixture.

Two stages, each skipped when its inputs are unchanged:

  preprocess  daily panel -> participant-safe windows, with validation reports
  features    personalization transforms, feature registry, leakage and
              contract checks -> interim/features.csv and feature_columns.json

The permutation scripts read interim/features.csv and
interim/feature_columns.json from the workspace.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .acquisition.fixtures import FIXTURE_STAMP, make_daily_panel
from .features.personalization import apply_personalization, personalization_columns
from .features.registry import build as build_registry
from .features.registry import save as save_registry
from .preprocessing.windows import WindowSpec, build_windows
from .utils.config import load, repo_path
from .utils.hashing import config_hash, sha256_file
from .utils.logging_utils import log
from .validation.checks import (
    leakage_scan,
    quality_report,
    schema_report,
    validate_processed_contract,
)


def _dedupe(names: list[str]) -> list[str]:
    """Order-preserving deduplication. A duplicated feature name silently doubles
    that feature's weight in any linear model and breaks column-wise scans."""
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


@dataclass
class Workspace:
    root: Path
    data_source: str = FIXTURE_STAMP

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for sub in ("interim", "results", "validation"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    # ---- resume bookkeeping ---------------------------------------------
    def state_path(self, stage: str) -> Path:
        return self.root / "interim" / f"{stage}.state.json"

    def is_current(self, stage: str, key: str) -> bool:
        p = self.state_path(stage)
        if not p.exists():
            return False
        try:
            return json.loads(p.read_text()).get("key") == key
        except (json.JSONDecodeError, OSError):
            return False

    def mark(self, stage: str, key: str, **fields: Any) -> None:
        self.state_path(stage).write_text(json.dumps(
            {"stage": stage, "key": key, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                       time.gmtime()),
             **fields}, indent=2, default=str))


def load_source_panel(source: str, target: str | None = None,
                      raw_root: Path | None = None,
                      n_participants: int = 40, days: int = 45
                      ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (daily_panel, participant_metadata) for the requested source.

    `fixture` generates synthetic data and marks every downstream artifact as
    synthetic. The real sources go through the readers, which check the
    documented layout and raise an error on an unexpected schema.
    """
    if source == "fixture":
        return make_daily_panel(n_participants=n_participants, days=days,
                                seed_name="pipeline_panel"), None

    root = Path(raw_root) if raw_root is not None else repo_path("data", "raw")
    if source == "studentlife":
        from .preprocessing.readers import studentlife

        return studentlife.load_daily_panel(root / "studentlife", target=target)
    if source == "depresjon":
        from .preprocessing.readers import depresjon

        return depresjon.load_daily_panel(root / "depresjon")
    raise ValueError(f"unknown source {source!r}; expected fixture|studentlife|depresjon")


def stage_preprocess(ws: Workspace, n_participants: int = 40, days: int = 45,
                     force: bool = False, source: str = "fixture",
                     target: str | None = None,
                     raw_root: Path | None = None) -> pd.DataFrame:
    """Daily panel -> participant-safe windows, with validation reports."""
    out = ws.path("interim", "windows.csv")
    window_cfg = load("features")["windowing"].get(
        source if source in ("studentlife", "depresjon") else "studentlife")
    key = config_hash({"source": source, "target": target, "n": n_participants,
                       "d": days, "spec": window_cfg})
    if out.exists() and ws.is_current("preprocess", key) and not force:
        log("stage_skipped", stage="preprocess", reason="inputs unchanged")
        return pd.read_csv(out)

    panel, metadata = load_source_panel(source, target, raw_root, n_participants, days)
    ws.data_source = FIXTURE_STAMP if source == "fixture" else f"REAL:{source}"

    ws.path("validation", "panel_schema.json").write_text(
        json.dumps(schema_report(panel, f"{source}_daily_panel"), indent=2))
    quality_report(panel).to_csv(ws.path("validation", "panel_quality.csv"), index=False)
    if metadata is not None:
        # Metadata (MADRS, inpatient status, demographics) stays out of the feature
        # matrix by construction: it is written to a separate file and never merged.
        metadata.to_csv(ws.path("validation", "participant_metadata.csv"), index=False)

    daily = [c for c in panel.columns if c not in ("study_id", "day_index", "label")]
    overrides = {"min_coverage": 0.4} if source == "fixture" else {}
    spec = WindowSpec(**{**window_cfg, **overrides})
    windows, report = build_windows(panel, daily, spec)
    if windows.empty:
        raise RuntimeError(
            f"{source}: no window survived the coverage thresholds "
            f"({spec}). Loosen min_coverage/min_valid_days in configs/features.yaml "
            "deliberately and record the change, or accept that this cohort cannot "
            "support this window length.")
    report["source"] = source
    report["target"] = target
    ws.path("validation", "window_report.json").write_text(json.dumps(report, indent=2))
    windows.to_csv(out, index=False)
    ws.mark("preprocess", key, source=source, target=target, n_windows=len(windows),
            n_participants=int(windows["study_id"].nunique()),
            output_sha256=sha256_file(out))
    log("stage_complete", stage="preprocess", source=source,
        **{k: report[k] for k in ("n_windows", "n_participants")})
    return windows


def stage_features(ws: Workspace, force: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Personalization transforms, feature registry, leakage and contract checks."""
    src = ws.path("interim", "windows.csv")
    if not src.exists():
        raise FileNotFoundError("run the preprocess stage first")
    key = sha256_file(src) + config_hash(load("features")["personalization"])
    out = ws.path("interim", "features.csv")
    cols_path = ws.path("interim", "feature_columns.json")
    if out.exists() and cols_path.exists() and ws.is_current("features", key) and not force:
        log("stage_skipped", stage="features", reason="inputs unchanged")
        return pd.read_csv(out), json.loads(cols_path.read_text())

    windows = pd.read_csv(src)
    # Base features are selected from the PRE-personalization frame. Selecting them
    # afterwards would also match `pz__x__mean`, which ends in `__mean` too, and the
    # duplicate column would surface much later as an unrelated pandas error.
    base = [c for c in windows.columns if c.endswith(("__mean", "__sd", "__slope"))]
    feat = apply_personalization(windows, [c for c in windows.columns if c.endswith("__mean")])
    cols = _dedupe(base
                   + personalization_columns(feat)
                   + [c for c in feat.columns if c.startswith("cov__")]
                   + ["window_coverage", "valid_days"])

    contract = validate_processed_contract(feat[cols])
    leak = leakage_scan(feat, cols)
    ws.path("validation", "leakage_scan.json").write_text(
        json.dumps({"contract": contract, "leakage": leak}, indent=2))
    if leak["status"] == "FAIL" or contract["status"] == "FAIL":
        raise RuntimeError(f"leakage or contract check failed: leakage={leak} contract={contract}")

    save_registry(build_registry(feat, cols), ws.path("results", "feature_registry.csv"),
                  ws.path("results", "feature_registry.json"))
    feat.to_csv(out, index=False)
    cols_path.write_text(json.dumps(cols, indent=2))
    ws.mark("features", key, n_features=len(cols), output_sha256=sha256_file(out))
    log("stage_complete", stage="features", n_features=len(cols))
    return feat, cols
