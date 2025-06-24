"""Wrapper around datasketch.MinHashLSH with default parameters for TokenSlasher."""
from __future__ import annotations

from typing import Dict, Iterable, List, Set

from datasketch import MinHash, MinHashLSH


class LSHIndex:
    """Light wrapper with sensible defaults (≈0.8 Jaccard)."""

    def __init__(
        self,
        *,
        threshold: float = 0.8,
        num_perm: int = 128,
        bands: int = 8,
        rows: int = 16,
    ) -> None:
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._stored: Dict[str, MinHash] = {}
        self.bands = bands
        self.rows = rows

    # --------------------------------------------------
    # Mutation
    # --------------------------------------------------

    def add(self, key: str, mh: MinHash) -> None:
        """Add *mh* under *key* to the index."""
        self.lsh.insert(key, mh)
        self._stored[key] = mh

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------

    def get_candidates(self, mh: MinHash) -> List[str]:
        """Return candidate duplicate keys for *mh* using LSH."""
        return list(self.lsh.query(mh))

    def get_minhash(self, key: str) -> MinHash:
        return self._stored[key]

    # --------------------------------------------------
    # Bulk utilities
    # --------------------------------------------------

    def items(self):  # noqa: D401 – simple method
        """Iterate *(key, MinHash)* pairs in insertion order."""
        return self._stored.items() 