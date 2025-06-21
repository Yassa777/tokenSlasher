TokenSlasher slashes redundant tokens from massive text corpora at warp speed. Train on 100% quality with 0% bloat.

| Dataset | Before Tokens | After Tokens | Tokens Saved | Processing Speed |
|---------|---------------|--------------|--------------|------------------|
| (fill)  |               |              |              |                  |

## Overview
Efficient near-duplicate detection and removal for large-scale text datasets using 6-gram MinHash + LSH.

## Quick Start
```
sh train.sh
```
Produces cleaned shards and logs under `results/` in one command.

## Method Highlights
* Slab streaming: processes 100 k-token slabs with mmap so resident set size stays flat.
* Sentinel `<doc>` injected between docs so 6-grams never bleed across boundaries.
* 128-perm MinHash + 8×16 LSH buckets (≈0.8 Jaccard).
* Vectorised xxhash64 batch hashing with benchmarked fallback path.
* Junk-doc filter drops tiny or low-alpha content automatically.
* Throughput & accuracy metrics auto-logged for every run.

## False-Positive / False-Negative Rates
| Threshold | FP Rate | FN Rate |
|-----------|--------|---------|
| 0.8       | (fill) | (fill)  |

## Runtime Environment
* Python ≥3.9
* `datasketch`, `numpy`, `xxhash`, `tqdm`, `pandas`, `matplotlib`

## One-Command Repro
See `train.sh` for the exact flags used to replicate the results table.

## License
Apache-2.0 