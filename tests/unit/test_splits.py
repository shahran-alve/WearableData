import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wearabledata.features.splits import (
    grouped_stratified_split,
    load_splits,
    nested_grouped_cv,
    save_splits,
)
from wearabledata.validation.checks import split_disjointness


def test_participants_never_cross_partitions(windows):
    s = grouped_stratified_split(windows)
    assert split_disjointness(s)["status"] == "PASS"
    assert sum(len(v) for v in s.values()) == windows["study_id"].nunique()


def test_split_is_deterministic(windows):
    assert grouped_stratified_split(windows) == grouped_stratified_split(windows)


def test_split_hash_roundtrip(tmp_path, windows):
    s = grouped_stratified_split(windows)
    h = save_splits(s, tmp_path / "splits.json")
    loaded, h2 = load_splits(tmp_path / "splits.json")
    assert loaded == s and h == h2


def test_nested_cv_folds_are_disjoint(windows):
    folds = nested_grouped_cv(windows, outer_folds=4)
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert not set(f["validation"]) & set(f["test"])
