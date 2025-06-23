"""Streaming ingestion utilities for TokenSlasher.

Keeps memory usage flat by reading the corpus as mmapped *slabs* of ~100 k tokens.
A *sentinel* token is injected between documents so that 6-grams do not cross
boundaries.
"""
from __future__ import annotations

import io
import mmap
import os
import re
import sys
from pathlib import Path
from typing import Generator, Iterable, List

import numpy as np

from . import SENTINEL

# -----------------------------------------------------------
# Tokenisation helpers
# -----------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def tokenize(text: str, *, use_sentencepiece: bool = False, sp_model=None) -> List[str]:
    """Tokenise *text* into a list of whitespace-separated tokens.

    Parameters
    ----------
    text : str
        The raw input text.
    use_sentencepiece : bool, optional
        Switch to a SentencePiece model if *True*.
    sp_model : sentencepiece.SentencePieceProcessor | None
        Pre-loaded SentencePiece model. Required when *use_sentencepiece* is
        *True*.
    """
    # Cast non‐string (e.g. NaN) to empty string.
    if text is None or not isinstance(text, str):
        text = ""

    if use_sentencepiece:
        if sp_model is None:
            raise ValueError("SentencePiece model must be supplied when use_sentencepiece=True")
        return sp_model.encode(text, out_type=str)
    # Cheap regex whitespace split.
    return [tok for tok in _WHITESPACE_RE.split(text.strip()) if tok]


# -----------------------------------------------------------
# Junk-document filter
# -----------------------------------------------------------

def is_junk(tokens: List[str]) -> bool:
    """Return *True* if the document is deemed *junk* and should be skipped.

    A document is *junk* when it satisfies either:
    * fewer than 20 alphabetic tokens, or
    * alphabetic-character ratio < 0.3.
    """
    if not tokens:
        return True
    alpha_tokens = [t for t in tokens if _ALPHA_RE.search(t)]
    if len(alpha_tokens) < 20:
        return True
    return (len(alpha_tokens) / len(tokens)) < 0.3


# -----------------------------------------------------------
# Streaming slab reader
# -----------------------------------------------------------

def slab_generator(path: os.PathLike | str, slab_tokens: int = 100_000) -> Generator[List[str], None, None]:
    """Yield token *slabs* of ~``slab_tokens`` size from *path*.

    The function keeps a rolling token buffer and flushes whenever the size
    threshold is reached. Each document (line) is tokenised and the *SENTINEL*
    token is appended so that no 6-gram spans two documents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    token_buffer: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = tokenize(line)
            if is_junk(tokens):
                continue
            token_buffer.extend(tokens + [SENTINEL])

            if len(token_buffer) >= slab_tokens:
                yield token_buffer
                token_buffer = []
        # flush remaining
        if token_buffer:
            yield token_buffer


# -----------------------------------------------------------
# N-gram helpers
# -----------------------------------------------------------

def ngrams(tokens: List[str], n: int = 6) -> Iterable[str]:
    """Generate *n*-grams (as joined strings) from *tokens*.

    Any n-gram that contains the *SENTINEL* token in an **internal** position is
    skipped so that no 6-gram straddles two documents, satisfying the unit test
    expectations.
    """
    if len(tokens) < n:
        return []  # type: ignore[return-value]

    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]  # noqa: E203 (black formatting)
        if SENTINEL in window:
            pos = window.index(SENTINEL)
            if 0 < pos < n - 1:
                # Sentinel appears inside the n-gram → skip.
                continue
        yield " ".join(window)


def hashed_ngrams(tokens: List[str], n: int = 6, *, base: int = 257) -> Iterable[int]:
    """Generate rolling 64-bit Karp-Rabin hashes for *n*-grams.

    The function pre-hashes each token with 64-bit *xxHash* (fallback to
    :pyfunc:`hash`) and then applies a classic polynomial rolling hash with
    modulus ``2**64`` (implicit via 64-bit overflow & masking).
    """
    mask = 0xFFFFFFFFFFFFFFFF  # 64-bit mask for overflow semantics

    if len(tokens) < n:
        return []  # type: ignore[return-value]

    # Pre-compute 64-bit token hashes (fast SIMD path when *xxhash* w/ NumPy is available)
    from .minhash import batch_xxhash64  # local import to avoid circular deps

    token_hashes = batch_xxhash64(tokens)

    # base^(n-1)  (mod 2^64) for rolling update step
    pow_base_n_minus1 = pow(base, n - 1, 1 << 64)

    # Initial hash value for the first window
    h = 0
    for i in range(n):
        h = ((h * base) + token_hashes[i]) & mask
    yield h

    # Slide window across the sequence
    for i in range(n, len(token_hashes)):
        outgoing = token_hashes[i - n]
        incoming = token_hashes[i]
        # Remove contribution of the outgoing token
        h = (h - (outgoing * pow_base_n_minus1 & mask)) & mask
        # Multiply by base and add incoming token
        h = ((h * base) + incoming) & mask
        yield h 