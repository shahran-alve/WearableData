"""Workspace builder: end-to-end run, determinism, and resume behaviour."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARGS = ["--source", "fixture", "--participants", "24", "--days", "35"]


def _build(out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "build_workspace.py"),
                           *ARGS, "--out", str(out), *extra],
                          capture_output=True, text=True, cwd=ROOT)


def test_fixture_workspace_builds_end_to_end(tmp_path):
    r = _build(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    for rel in ("interim/features.csv", "interim/feature_columns.json",
                "validation/leakage_scan.json", "results/feature_registry.csv"):
        assert (tmp_path / rel).exists(), f"missing {rel}"
    cols = json.loads((tmp_path / "interim" / "feature_columns.json").read_text())
    assert len(cols) > 10


def test_two_builds_are_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _build(a).returncode == 0
    assert _build(b).returncode == 0
    for rel in ("interim/features.csv", "interim/feature_columns.json"):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_rerunning_is_a_no_op(tmp_path):
    assert _build(tmp_path).returncode == 0
    second = _build(tmp_path)
    assert second.returncode == 0
    assert second.stderr.count("stage_skipped") == 2, "an unchanged stage was recomputed"
