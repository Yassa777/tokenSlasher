#!/usr/bin/env bash
set -euo pipefail

python -m detector.dedup \
  --data_dir "${1:-data}" \
  --ngram 6 \
  --threshold 0.8 \
  --processes "$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
  --topk 5000 \
  --out results/ 