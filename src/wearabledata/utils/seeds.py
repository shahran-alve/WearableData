"""Deterministic seed derivation. Prohibits unseeded randomness."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np

GLOBAL_SEED = 42


def derive_seed(name: str, global_seed: int = GLOBAL_SEED) -> int:
    """Derive a stable component seed from a name and the global seed.

    Uses blake2b rather than hash() because Python's hash() is salted per
    process and would silently break reproducibility.
    """
    payload = f"{global_seed}:{name}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def seed_everything(name: str = "global", global_seed: int = GLOBAL_SEED) -> int:
    seed = derive_seed(name, global_seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def rng(name: str, global_seed: int = GLOBAL_SEED) -> np.random.Generator:
    return np.random.default_rng(derive_seed(name, global_seed))
