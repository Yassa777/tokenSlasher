"""MinHash utilities for TokenSlasher."""
from __future__ import annotations

import os
import time
from typing import Iterable, List

try:
    import xxhash  # type: ignore
except ImportError:  # pragma: no cover
    xxhash = None  # noqa: N816

from datasketch import MinHash

# -----------------------------------------------------------
# xxHash helpers
# -----------------------------------------------------------


def _hash_xx64(value: str) -> int:
    if xxhash is None:
        return hash(value) & 0xFFFFFFFFFFFFFFFF  # fallback Python hash masked to 64-bit
    return xxhash.xxh64_intdigest(value)


def batch_xxhash64(strings: List[str]) -> List[int]:
    """Vectorised 64-bit hashes with a fallback path when *xxhash* SIMD is unavailable."""
    if xxhash is None or not getattr(xxhash, "has_numpy", False):
        # Naïve Python loop fallback.
        return [_hash_xx64(s) for s in strings]
    # Fast NumPy path.
    import numpy as np  # local import to avoid hard dependency when not needed

    hashes = xxhash.xxh64_np(strings)  # type: ignore[attr-defined]
    return hashes.astype(np.uint64).tolist()


def batch_xxhash32(strings: List[str]) -> List[int]:
    """Return lower 32 bits of xxhash64 for each string."""
    return [h & 0xFFFFFFFF for h in batch_xxhash64(strings)]


# -----------------------------------------------------------
# MinHash helpers
# -----------------------------------------------------------

_NUM_PERMUTATIONS = 128


def compute_minhash(tokens: Iterable[str], num_perm: int = _NUM_PERMUTATIONS) -> MinHash:
    """Compute a ``datasketch.MinHash`` from an iterable of tokens."""
    mh = MinHash(num_perm=num_perm)
    for h in batch_xxhash64(list(tokens)):
        mh.update(h.to_bytes(8, byteorder="little"))
    return mh


def compute_minhash_from_hashes(hashes: Iterable[int], num_perm: int = _NUM_PERMUTATIONS) -> MinHash:
    """Compute a MinHash sketch directly from an iterable of 64-bit integer hashes.

    This variant avoids an additional pass through *xxHash* when the caller
    already supplies pre-hashed 64-bit values (e.g. output of
    :pyfunc:`detector.ingest.hashed_ngrams`).
    """
    mh = MinHash(num_perm=num_perm)
    for h in hashes:
        mh.update(h.to_bytes(8, byteorder="little"))
    return mh 