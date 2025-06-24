"""Similarity utilities: SimHash and cosine embeddings."""
from __future__ import annotations

from typing import Iterable, List

import math
import logging

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# SimHash
# -----------------------------------------------------------


def compute_simhash(hashes: Iterable[int], bits: int = 64) -> int:
    """Compute *bits*-wide SimHash signature from an iterable of 64-bit token hashes."""
    v = [0] * bits
    mask = 1 << (bits - 1)
    for h in hashes:
        for i in range(bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming_similarity(a: int, b: int, bits: int = 64) -> float:
    """Return similarity in [0,1] as (bits - distance) / bits."""
    dist = (a ^ b).bit_count()
    return (bits - dist) / bits

# -----------------------------------------------------------
# Sentence-BERT embeddings
# -----------------------------------------------------------

_model_cache: SentenceTransformer | None = None


def _get_model(name: str = "paraphrase-MiniLM-L6-v2") -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Install to use cosine metric.")
        _model_cache = SentenceTransformer(name)
    return _model_cache


def embed_text(text: str) -> "np.ndarray":  # type: ignore[name-defined]
    model = _get_model()
    return model.encode(text, convert_to_tensor=False)  # returns 1-D np.ndarray


def cosine_similarity(vec1: "np.ndarray", vec2: "np.ndarray") -> float:  # type: ignore[name-defined]
    if np is None:
        raise ImportError("NumPy not installed for cosine similarity")
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-9
    return float(np.dot(vec1, vec2) / denom) 