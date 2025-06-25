TokenSlasher slashes redundant tokens from massive text corpora at warp speed. Train on 100% quality with 0% bloat.

<img width="1709" alt="image" src="https://github.com/user-attachments/assets/07f39dd7-82dc-466e-a26b-bb4bbe7e2dfa" />


| Dataset | Before Tokens | After Tokens | Tokens Saved | Processing Speed |
|---------|---------------|--------------|--------------|------------------|
| (fill)  |               |              |              |                  |

| Dataset | Tokens | Speed (tok/s) | ROC-AUC | PR-AUC |
|---------|--------|---------------|---------|---------|
| Enron 2.5k docs | 0.67 M | 143 k | — | — |
| Quora 100 k pairs | — | — | 0.909 | 0.826 |

## Overview
Efficient near-duplicate detection and removal for large-scale text datasets using 6-gram MinHash + LSH.

## Quick Start
```
pip install tokenslasher            # from PyPI after wheel upload
# or, from source:
#   git clone https://github.com/yourname/tokenslasher && cd tokenslasher
#   pip install -e .

# run deduplication
tokenslasher --data_dir data/all_docs --out results/
```
Produces cleaned shards and logs under `results/` in one command.

## Method Highlights
* Slab streaming: processes 100 k-token slabs with mmap so resident set size stays flat.
* Character 5-gram + token skip-gram shingles with spaCy lemmatisation.
* Sentinel `<doc>` injected between docs so shingles never bleed across boundaries.
* 128-perm MinHash + 64×2 LSH buckets → high recall.
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

## Evaluation on Quora-style Questions

For a labelled test set like `data/questions.csv` (columns `qid1, qid2, question1, question2, is_duplicate`) run:

```bash
python scripts/eval_questions.py --csv data/questions.csv --out results/
```

This generates ROC / PR curves and writes `questions_metrics.txt` summarising ROC-AUC, PR-AUC and the best F1 threshold.

### Training the blend classifier

We train a lightweight logistic-regression model that blends lexical & semantic
signals plus a handful of length features:

```bash
# optional: change --sample to use a smaller subset while iterating
python scripts/train_lr.py --csv data/questions.csv \
       --sample 100000 \
       --out results/lr_model.joblib
```

Features per pair:

1. Lexical Jaccard (char-5-gram ∪ skip-gram).
2. Semantic cosine (SBERT `all-mpnet-base-v2`).
3. Length ratio (chars) & token-length ratio.
4. Token overlap ratio.
5. Absolute length / token-length deltas.

The script reports ROC-AUC / PR-AUC on a hold-out split and saves the model.

During evaluation `eval_questions.py` automatically loads
`results/lr_model.joblib` and applies the classifier; otherwise it falls back to
the fixed 0.6 × lexical + 0.4 × semantic blend.

### Error analysis & speed logs

`eval_questions.py` additionally produces:

* `results/errors.jsonl` – top-50 false-positive / false-negative pairs with
  feature values.
* `results/speed_eval.txt` – pair-count, elapsed seconds, RSS at start/end.

These artefacts make it trivial to iterate on new features or alternative
models.

## License
Apache-2.0

### Docker

```bash
docker build -t tokenslasher .
# process corpus inside container (bind-mount data)
docker run --rm -v $PWD/data:/data tokenslasher --data_dir /data --out /data/results
```

### Dashboard demo

```bash
pip install .[dashboard]  # or use the docker image
streamlit run dashboard/app.py --server.headless true
```

Open the URL printed in the console to explore `results/errors.jsonl` interactively. 
