"""Near-duplicate detection pipeline for TokenSlasher.

Example
-------
python -m detector.dedup \
    --data_dir path/to/txts \
    --ngram 6 \
    --threshold 0.8 \
    --processes 8 \
    --topk 5000 \
    --out results/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Iterable
import concurrent.futures
from datasketch import MinHash

from . import SENTINEL
from .ingest import slab_generator, hashed_ngrams
from .minhash import compute_minhash_from_hashes
from .lsh_index import LSHIndex

# Pipeline
# -----------------------------------------------------------


def _compute_minhash_worker(args: Tuple[str, List[str], int]) -> Tuple[str, MinHash]:
    """Helper executed in a separate process to compute the MinHash sketch.

    Parameters
    ----------
    args : Tuple[str, List[str], int]
        (doc_id, tokens, ngram)
    """
    doc_id, tokens, ngram = args
    mh = compute_minhash_from_hashes(hashed_ngrams(tokens, n=ngram))
    return doc_id, mh


def _handle_minhash(
    doc_id: str,
    mh,  # MinHash
    lsh: LSHIndex,
    threshold: float,
    topk: int,
    dup_pairs: List[Tuple[str, str, float]],
) -> None:
    """Same as _handle_doc but starts with a pre-computed MinHash."""
    candidates = lsh.get_candidates(mh)

    # Verify with real Jaccard.
    verified: List[Tuple[str, float]] = []
    for cand_key in candidates:
        score = mh.jaccard(lsh.get_minhash(cand_key))
        if score >= threshold:
            verified.append((cand_key, score))
    if verified:
        dup_pairs.extend([(doc_id, other, score) for other, score in verified])
        dup_pairs.sort(key=lambda x: x[2], reverse=True)
        dup_pairs[:] = dup_pairs[:topk]

    lsh.add(doc_id, mh)


def process_corpus(
    data_dir: Path,
    ngram: int,
    threshold: float,
    topk: int,
    out_dir: Path,
    processes: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    speed_path = out_dir / "speed.txt"
    dup_pairs_path = out_dir / "samples.jsonl"

    lsh = LSHIndex(threshold=threshold)
    dup_pairs: List[Tuple[str, str, float]] = []

    total_tokens = 0
    t0 = time.time()

    # Prepare executor for concurrent MinHash computation
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    if processes and processes > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=processes)

    futures: List[concurrent.futures.Future] = []

    def _submit(job_doc_id: str, job_tokens: List[str]):
        """Submit a MinHash computation job – serial fallback when *executor* is None."""
        nonlocal futures
        if executor is None:
            # Synchronous path (useful for unit tests)
            mh = compute_minhash_from_hashes(hashed_ngrams(job_tokens, n=ngram))
            _handle_minhash(job_doc_id, mh, lsh, threshold, topk, dup_pairs)
        else:
            fut = executor.submit(_compute_minhash_worker, (job_doc_id, job_tokens, ngram))

            def _done_callback(f: concurrent.futures.Future):  # noqa: N802
                _doc_id, _mh = f.result()
                _handle_minhash(_doc_id, _mh, lsh, threshold, topk, dup_pairs)

            fut.add_done_callback(_done_callback)
            futures.append(fut)

    # Walk through files and feed slabs.
    txt_files = sorted(p for p in data_dir.glob("**/*") if p.is_file())
    doc_counter = 0
    for fp in txt_files:
        for slab in slab_generator(fp):
            # naive doc segmentation on SENTINEL token.
            doc_tokens: List[str] = []
            for tok in slab:
                if tok == SENTINEL:
                    if doc_tokens:
                        doc_id = f"doc_{doc_counter}"
                        doc_counter += 1
                        _submit(doc_id, doc_tokens)
                        total_tokens += len(doc_tokens)
                        doc_tokens = []
                else:
                    doc_tokens.append(tok)
            # flush last
            if doc_tokens:
                doc_id = f"doc_{doc_counter}"
                doc_counter += 1
                _submit(doc_id, doc_tokens)
                total_tokens += len(doc_tokens)

    # Ensure all concurrent jobs finished first
    if executor is not None:
        executor.shutdown(wait=True)

    # Save sample pairs.
    with dup_pairs_path.open("w", encoding="utf-8") as f:
        for a, b, score in dup_pairs[:5]:
            json.dump({"a": a[:120], "b": b[:120], "score": score}, f)
            f.write("\n")

    elapsed = time.time() - t0
    throughput = total_tokens / max(elapsed, 1e-9)
    with speed_path.open("w", encoding="utf-8") as f:
        f.write(
            f"tokens_processed\tseconds\tthroughput_tokens_per_s\n"
            f"{total_tokens}\t{elapsed:.2f}\t{throughput:.2f}\n"
        )

    print(
        f"Processed {total_tokens:,} tokens from {len(txt_files)} files in "
        f"{elapsed:.2f}s ({throughput:.1f} tokens/s)."
    )


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TokenSlasher duplicate-removal pipeline")
    p.add_argument("--data_dir", type=Path, required=True, help="Directory containing raw text files")
    p.add_argument("--ngram", type=int, default=6, help="N-gram size (default: 6)")
    p.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold")
    p.add_argument("--processes", type=int, default=8, help="Process count for parallel MinHash computation")
    p.add_argument("--topk", type=int, default=5000, help="Max verified dup pairs stored")
    p.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    process_corpus(
        args.data_dir,
        args.ngram,
        args.threshold,
        args.topk,
        args.out,
        args.processes,
    )


if __name__ == "__main__":  # pragma: no cover
    main() 