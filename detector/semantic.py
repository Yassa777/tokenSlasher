"""Lightweight wrapper for sentence-transformer embeddings."""
from __future__ import annotations

from functools import lru_cache
from typing import List
import os

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# Prevent Transformers from importing TensorFlow / Keras (avoids Keras 3 conflict)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

MODEL_NAME = os.getenv("TOKSLASHER_SBER_MODEL", "all-mpnet-base-v2")


@lru_cache(maxsize=1)
def _get_model():
    """Load SentenceTransformer model (memoised). Override via env var TOKSLASHER_SBER_MODEL."""
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers not installed. Please install to use semantic similarity."
        )
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:  # type: ignore[name-defined]
    """Return 384-dim embedding for *text*."""
    model = _get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.astype("float32")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[name-defined]
    if a.shape != b.shape:
        raise ValueError("Embedding shapes differ")
    return float(np.dot(a, b))


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:  # type: ignore[name-defined]
    """Vectorise a list of texts to embeddings (numpy array)."""
    model = _get_model()
    arr = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return arr.astype("float32") 