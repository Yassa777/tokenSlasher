"""Property-based similarity tests (Hypothesis)."""
from __future__ import annotations

import math
import string
from typing import List

import pytest

hyp = pytest.importorskip("hypothesis")

import hypothesis.strategies as st  # type: ignore
from hypothesis import given, assume  # type: ignore

from detector.ingest import ngrams
from detector.minhash import compute_minhash
from detector.similarity import hamming_similarity, cosine_similarity

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

BITS = 64

# ---------------------------------------------------------------------------
# Jaccard via MinHash
# ---------------------------------------------------------------------------


@st.composite
def _token_lists(draw) -> List[str]:
    size = draw(st.integers(min_value=5, max_value=40))
    toks = draw(
        st.lists(
            st.text(string.ascii_lowercase, min_size=1, max_size=6),
            min_size=size,
            max_size=size,
        )
    )
    return toks


@given(a=_token_lists(), b=_token_lists(), t1=st.floats(0.0, 1.0), t2=st.floats(0.0, 1.0))
def test_jaccard_threshold_monotone(a: List[str], b: List[str], t1: float, t2: float) -> None:
    if t2 <= t1:
        t1, t2 = t2, t1  # ensure t2 > t1
    s = _jaccard(a, b)
    assert (s >= t2) <= (s >= t1)


@given(tokens=_token_lists())
def test_jaccard_reflexive(tokens: List[str]) -> None:
    assert _jaccard(tokens, tokens) >= 0.9


def _jaccard(a: List[str], b: List[str]) -> float:
    mh1 = compute_minhash(ngrams(a, 6))
    mh2 = compute_minhash(ngrams(b, 6))
    return mh1.jaccard(mh2)

# ---------------------------------------------------------------------------
# SimHash (Hamming)
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=(1 << BITS) - 1))
def test_simhash_self_similarity(full_int: int) -> None:
    assert math.isclose(hamming_similarity(full_int, full_int, bits=BITS), 1.0)


@given(original=st.integers(min_value=0, max_value=(1 << BITS) - 1), flips=st.integers(0, BITS))
def test_hamming_similarity_formula(original: int, flips: int) -> None:
    mask = sum(1 << i for i in range(flips))  # lowest *flips* bits
    modified = original ^ mask
    expected = (BITS - flips) / BITS
    sim = hamming_similarity(original, modified, bits=BITS)
    assert math.isclose(sim, expected)

# ---------------------------------------------------------------------------
# Cosine similarity (Sentence-BERT vectors test only when numpy present)
# ---------------------------------------------------------------------------


@given(vec=st.lists(st.floats(-1, 1, allow_nan=False, allow_infinity=False), min_size=8, max_size=64))
def test_cosine_self_similarity(vec):  # type: ignore[no-any-unbound]
    assume(np is not None)
    v = np.asarray(vec, dtype=float)
    assume(v.any())
    assert math.isclose(cosine_similarity(v, v), 1.0, abs_tol=1e-6)


@given(v1=st.lists(st.floats(-1, 1, allow_nan=False, allow_infinity=False), min_size=16, max_size=64), noise=st.floats(0.1, 5))
def test_cosine_similarity_decreases_with_noise(v1, noise):  # type: ignore[no-any-unbound]
    assume(np is not None)
    a = np.asarray(v1, dtype=float)
    assume(a.any())
    b = a + np.random.normal(scale=noise, size=a.shape)
    assert cosine_similarity(a, b) <= cosine_similarity(a, a) + 1e-6 