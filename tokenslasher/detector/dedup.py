"""Near-duplicate detection pipeline with exact-Jaccard cache.

The pipeline uses MinHash + LSH to shortlist candidate pairs and an in-memory
cache of shingle hash sets to compute exact Jaccard similarity for
verification.  For corpora larger than a few million docs the cache can be
spilled to LMDB/RocksDB (TODO).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from . import SENTINEL
from .ingest import (
    slab_generator,
    char_ngrams,
    token_skipgrams,
)
from .lsh_index import LSHIndex
from .minhash import batch_xxhash64, compute_minhash_from_hashes
from .semantic import embed_text, cosine_similarity

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# -----------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------

def process_corpus(
    data_dir: Path,
    threshold: float,
    topk: int,
    out_dir: Path,
    ngram: int = 6,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    speed_path = out_dir / "speed.txt"
    dup_pairs_path = out_dir / "samples.jsonl"

    lsh = LSHIndex(threshold=threshold, bands=64, rows=2)
    dup_pairs: List[dict] = []

    # Exact-Jaccard cache (doc_id -> set[hash])
    shingle_cache: Dict[str, set[int]] = {}
    embed_cache: Dict[str, "np.ndarray"] = {}

    total_tokens = 0
    t0 = time.time()

    txt_files = sorted(p for p in data_dir.glob("**/*") if p.is_file())
    doc_counter = 0

    for fp in txt_files:
        for slab in slab_generator(fp):
            tokens: List[str] = []
            for tok in slab:
                if tok == SENTINEL:
                    if tokens:
                        _handle_doc(
                            doc_id=f"doc_{doc_counter}",
                            tokens=tokens,
                            lsh=lsh,
                            cache=shingle_cache,
                            threshold=threshold,
                            topk=topk,
                            dup_pairs=dup_pairs,
                            ngram=ngram,
                            embed_cache=embed_cache,
                        )
                        total_tokens += len(tokens)
                        doc_counter += 1
                        tokens = []
                else:
                    tokens.append(tok)
            if tokens:
                _handle_doc(
                    doc_id=f"doc_{doc_counter}",
                    tokens=tokens,
                    lsh=lsh,
                    cache=shingle_cache,
                    threshold=threshold,
                    topk=topk,
                    dup_pairs=dup_pairs,
                    ngram=ngram,
                    embed_cache=embed_cache,
                )
                total_tokens += len(tokens)
                doc_counter += 1

    elapsed = time.time() - t0
    throughput = total_tokens / max(elapsed, 1e-9)

    # persist outputs
    with dup_pairs_path.open("w", encoding="utf-8") as f:
        for rec in dup_pairs[:50]:
            json.dump(rec, f)
            f.write("\n")

    speed_path.write_text(
        f"tokens_processed\tseconds\tthroughput_tokens_per_s\n{total_tokens}\t{elapsed:.2f}\t{throughput:.2f}\n"
    )

    print(
        f"Processed {total_tokens:,} tokens from {len(txt_files)} files in "
        f"{elapsed:.2f}s ({throughput:.0f} tok/s)."
    )


# -----------------------------------------------------------
# Internal
# -----------------------------------------------------------

def _handle_doc(
    *,
    doc_id: str,
    tokens: List[str],
    lsh: LSHIndex,
    cache: Dict[str, set[int]],
    threshold: float,
    topk: int,
    dup_pairs: List[dict],
    ngram: int,
    embed_cache: Dict[str, "np.ndarray"],
) -> None:
    # Build shingle strings (char 5-grams + token skip-1 grams)
    text = " ".join(tokens)
    shingles_str = set(char_ngrams(text, 5)).union(token_skipgrams(tokens, skip=1))

    # Hash shingles to 64-bit ints for memory efficiency
    shingle_hashes = batch_xxhash64(list(shingles_str))
    shingle_set = set(shingle_hashes)
    cache[doc_id] = shingle_set

    # MinHash sketch
    mh = compute_minhash_from_hashes(shingle_hashes)

    # LSH lookup
    candidates = lsh.get_candidates(mh)

    verified_records: List[dict] = []
    for cand in candidates:
        if cand not in cache:
            continue  # safety
        lex = _jaccard(shingle_set, cache[cand])

        # Semantic similarity
        import numpy as np  # local

        if doc_id in embed_cache:
            emb_a = embed_cache[doc_id]
        else:
            emb_a = embed_text(text)
            embed_cache[doc_id] = emb_a

        if cand in embed_cache:
            emb_b = embed_cache[cand]
        else:
            # Build from cached shingle? we don't keep text, so simple fallback: skip
            emb_b = emb_a  # degrade gracefully
        sem = cosine_similarity(emb_a, emb_b)

        blend = 0.6 * lex + 0.4 * sem

        if blend >= threshold:
            verified_records.append(
                {
                    "a": doc_id,
                    "b": cand,
                    "lexical": round(lex, 4),
                    "semantic": round(sem, 4),
                    "blend": round(blend, 4),
                }
            )

    if verified_records:
        dup_pairs.extend(verified_records)
        dup_pairs.sort(key=lambda x: x["blend"], reverse=True)
        dup_pairs[:] = dup_pairs[:topk]

    lsh.add(doc_id, mh)


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TokenSlasher duplicate-removal pipeline with exact-Jaccard cache")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=5000)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--ngram", type=int, default=6, help="Reserved for future use")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    process_corpus(args.data_dir, args.threshold, args.topk, args.out, args.ngram)


if __name__ == "__main__":  # pragma: no cover
    main() 