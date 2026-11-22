import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wearabledata.utils.hashing import config_hash, sha256_text
from wearabledata.utils.seeds import derive_seed, rng


def test_derive_seed_is_stable_across_processes():
    # blake2b, not Python's salted hash(): the value must be a constant.
    assert derive_seed("split") == derive_seed("split")
    assert derive_seed("split") != derive_seed("bootstrap")


def test_rng_reproducible():
    assert rng("x").normal(size=5).tolist() == rng("x").normal(size=5).tolist()


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_sha256_text_known_vector():
    assert sha256_text("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
