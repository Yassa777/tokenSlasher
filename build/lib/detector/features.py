"""Feature extraction for duplicate detection classifier."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .ingest import tokenize, char_ngrams, token_skipgrams
from .minhash import batch_xxhash64
from .semantic import embed_text, cosine_similarity


def _hashed_shingles(text: str) -> set[int]:
    toks = tokenize(text)
    shingles = set(char_ngrams(text, 5)).union(token_skipgrams(toks, skip=1))
    return set(batch_xxhash64(list(shingles)))


def extract_pair_features(q1: str, q2: str) -> Tuple[np.ndarray, float]:
    """Return feature vector *x* and label placeholder (caller supplies)."""
    s1 = _hashed_shingles(q1)
    s2 = _hashed_shingles(q2)

    len1 = len(q1)
    len2 = len(q2)
    tlen1 = len(tokenize(q1))
    tlen2 = len(tokenize(q2))

    # basic ratios
    lex_jacc = len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0
    len_ratio = min(len1, len2) / max(len1, len2 or 1)
    tok_ratio = min(tlen1, tlen2) / max(tlen1, tlen2 or 1)

    # token overlap
    toks1 = set(tokenize(q1))
    toks2 = set(tokenize(q2))
    tok_overlap = len(toks1 & toks2) / len(toks1 | toks2) if toks1 and toks2 else 0.0

    # semantic cosine
    emb1 = embed_text(q1)
    emb2 = embed_text(q2)
    sem_cos = cosine_similarity(emb1, emb2)

    feats = np.array([
        lex_jacc,
        sem_cos,
        len_ratio,
        tok_ratio,
        tok_overlap,
        abs(len1 - len2),
        abs(tlen1 - tlen2),
    ], dtype="float32")
    return feats 