"""Basic sanity tests for TokenSlasher."""
from __future__ import annotations

from detector.ingest import is_junk, ngrams
from detector.minhash import compute_minhash


def test_junk_filter_short_doc() -> None:
    assert is_junk(["word"] * 10)  # fewer than 20 tokens


def test_junk_filter_low_alpha() -> None:
    toks = ["!!!"] * 100 + ["alpha"] * 10  # 10% alpha ratio
    assert is_junk(toks)


def test_ngrams_no_cross_doc() -> None:
    toks = ["hello", "world", "<doc>", "foo", "bar", "baz", "qux"]
    grams = list(ngrams(toks, n=6))
    # Any n-gram containing SENTINEL should not span both sides meaning its length should be <6 due sentinel circumvent – here we simply ensure it's excluded.
    assert all("<doc>" not in g or g.startswith("<doc>") or g.endswith("<doc>") for g in grams)


def test_minhash_collision() -> None:
    toks1 = [f"tok{i}" for i in range(100)]
    toks2 = [f"tok{i+1000}" for i in range(100)]
    mh1 = compute_minhash(toks1)
    mh2 = compute_minhash(toks2)
    similarity = mh1.jaccard(mh2)
    assert similarity < 0.2  # low collision rate 