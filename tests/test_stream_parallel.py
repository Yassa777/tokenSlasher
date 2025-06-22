"""Parallel vs single-thread equivalence tests for TokenSlasher pipeline."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from detector.dedup import process_corpus


@pytest.mark.parametrize("processes", [1, 4])
def test_parallel_equivalence(processes: int) -> None:
    """Run pipeline with *processes* and compare outputs to baseline single-threaded run."""

    # Build synthetic corpus (~0.5 MB) with intentional duplicates.
    rng = random.Random(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        corpus_dir.mkdir()
        tokens = [f"tok{i}" for i in range(500)]
        prev_tokens: list[str] | None = None

        for i in range(30):
            doc_tokens = rng.choices(tokens, k=200)
            if prev_tokens is not None and i % 5 == 0:
                # Every 5th doc duplicates the previous one.
                doc_tokens = prev_tokens
            prev_tokens = doc_tokens
            (corpus_dir / f"file_{i}.txt").write_text(" ".join(doc_tokens), encoding="utf-8")

        # Output dirs
        out_single = Path(tmpdir) / "out_single"
        out_multi = Path(tmpdir) / f"out_multi_{processes}"

        # Baseline single-thread
        process_corpus(
            data_dir=corpus_dir,
            ngram=3,
            threshold=0.8,
            topk=100,
            out_dir=out_single,
            processes=1,
            metric="jaccard",
        )

        # Parallel variant
        process_corpus(
            data_dir=corpus_dir,
            ngram=3,
            threshold=0.8,
            topk=100,
            out_dir=out_multi,
            processes=processes,
            metric="jaccard",
        )

        def _load_pairs(p: Path):
            fp = p / "samples.jsonl"
            if not fp.exists():
                return []
            pairs = []
            with fp.open() as f:
                for line in f:
                    rec = json.loads(line)
                    # order-independent pair + rounded score for determinism
                    pair = tuple(sorted([rec["a"], rec["b"]])) + (round(rec["score"], 3),)
                    pairs.append(pair)
            return sorted(pairs)

        assert _load_pairs(out_single) == _load_pairs(out_multi) 