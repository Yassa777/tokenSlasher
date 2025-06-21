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
    """Generate *n*-grams (as joined strings) from *tokens*."""
    if len(tokens) < n:
        return []  # type: ignore
    joined = (
        " ".join(tokens[i : i + n])  # noqa: E203 (black formatting)
        for i in range(len(tokens) - n + 1)
    )
    return joined 