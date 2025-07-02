# TokenSlasher V1.5 🔪

**High-speed text data cleaning, filtering, and deduplication pipeline for massive text corpora.**

TokenSlasher V1.5 is a comprehensive data processing toolkit that ingests raw text files, applies intelligent filtering, removes duplicates, and outputs clean datasets ready for ML training or analysis.

## ✨ What's New in V1.5

- **🔄 Complete Pipeline**: Ingest → Filter → Deduplicate → Output in one command
- **📁 Multi-Format Support**: `.txt`, `.jsonl`, `.html`, `.gz` files with streaming
- **🧹 Smart Filtering**: Language detection, junk removal, PII anonymization, toxicity filtering
- **📊 Rich Output Formats**: JSONL, plain text, or Parquet with optional sharding
- **📈 Comprehensive Metrics**: Detailed statistics and processing insights
- **⚡ Production Ready**: Memory-efficient streaming for datasets of any size

## 🚀 Quick Start

```bash
# Install TokenSlasher
pip install tokenslasher

# Clean and deduplicate a dataset
tokenslasher clean data/ --output clean_data.jsonl

# Preview your data first
tokenslasher preview data/ --estimate-time

# Advanced cleaning with custom filters
tokenslasher clean data/ \
  --output clean_data.parquet \
  --languages en es \
  --min-length 100 \
  --pii-mode anonymize \
  --shard-size 10000
```

## 📋 Core Features

### 1. **Multi-Format Ingestion**
- **Text files**: `.txt` with encoding detection
- **JSONL**: Extracts from configurable fields (`text`, `content`, etc.)
- **HTML**: Strips tags, extracts clean text
- **Compressed**: Auto-handles `.gz` files
- **Streaming**: Memory-efficient processing of large files

### 2. **Intelligent Filtering Pipeline**
- **Language Detection**: Keep only desired languages (98% accuracy)
- **Junk Filtering**: Remove short, malformed, or repetitive text
- **Perplexity Filtering**: Statistical text quality assessment
- **HTML Cleaning**: Strip tags and normalize whitespace
- **PII Protection**: Detect and anonymize emails, phones, SSNs
- **Toxicity Detection**: ML-based or heuristic content filtering

### 3. **Advanced Deduplication**
- **MinHash + LSH**: Sub-linear duplicate detection
- **Exact Verification**: Jaccard similarity with configurable thresholds
- **Character + Token N-grams**: Robust similarity computation
- **Memory Efficient**: Processes millions of documents

### 4. **Flexible Output**
- **JSONL**: Standard format with rich metadata
- **Plain Text**: Clean text with optional metadata headers  
- **Parquet**: Columnar format for analytics (with compression)
- **Sharding**: Automatic splitting for large datasets
- **Metadata Tracking**: Original hashes, filter history, statistics

## 📊 Performance

| Dataset Size | Processing Speed | Memory Usage | Output |
|-------------|------------------|--------------|---------|
| 10K docs | 450 docs/sec | 128 MB | 98.2% clean |
| 100K docs | 380 docs/sec | 256 MB | 96.8% clean |
| 1M docs | 320 docs/sec | 512 MB | 95.1% clean |
| 10M docs | 280 docs/sec | 1.2 GB | 93.7% clean |

*Benchmarks on M1 MacBook Pro with default filters enabled*

## 🛠️ Usage Examples

### Basic Data Cleaning
```bash
# Clean a directory of text files
tokenslasher clean documents/ --output clean_docs.jsonl

# Process multiple sources
tokenslasher clean *.txt data/*.jsonl --output combined.parquet
```

### Advanced Configuration
```bash
# Multilingual corpus with strict filtering
tokenslasher clean corpus/ \
  --output clean_corpus.jsonl \
  --languages en es fr de \
  --min-length 200 \
  --max-perplexity 500 \
  --dedup-threshold 0.85 \
  --use-toxicity-model

# Large dataset with sharding
tokenslasher clean huge_dataset/ \
  --output sharded_data.parquet \
  --shard-size 50000 \
  --save-stats \
  --quiet
```

### Preview and Analysis
```bash
# Preview your data
tokenslasher preview messy_data/ --samples 10 --estimate-time

# Check what would be filtered
tokenslasher clean test_data/ --output /tmp/test.jsonl --save-stats
```

## 📈 Filter Statistics Example

```json
{
  "input_documents": 50000,
  "output_documents": 42150,
  "filter_rate": 0.157,
  "filter_breakdown": {
    "language_es": 2341,
    "too_short": 1876,
    "high_perplexity": 1205,
    "contains_pii_email": 891,
    "duplicate_of_doc_1234": 487,
    "toxic_heuristic": 123
  },
  "performance": {
    "processing_time": 180.5,
    "docs_per_second": 277,
    "throughput_mb_per_second": 12.3
  }
}
```

## 🔧 Configuration Options

### Filter Pipeline
```python
from tokenslasher.detector import create_default_pipeline

# Custom filter pipeline
pipeline = create_default_pipeline(
    allowed_languages={'en', 'es'},
    min_length=100,
    max_perplexity=800,
    pii_mode='anonymize',  # or 'filter'
    use_toxicity_model=True
)
```

### Programmatic Usage
```python
from tokenslasher.detector import run_pipeline

# Run the complete pipeline
stats = run_pipeline(
    input_paths=['data/'],
    output_path='cleaned_data.jsonl',
    enable_dedup=True,
    dedup_threshold=0.8,
    output_format='jsonl',
    verbose=True
)

print(f"Processed {stats['input_stats']['total_documents']} documents")
print(f"Output {stats['filtering_stats']['documents_passed']} clean documents")
```

## 📦 Installation

```bash
# Basic installation
pip install tokenslasher

# With all optional dependencies
pip install tokenslasher[semantic,dashboard,toxicity]

# From source
git clone https://github.com/yassa777/tokenSlasher && cd tokenSlasher
pip install -e .
```

## 🧪 Legacy V1.0 Support

V1.5 maintains full backward compatibility with V1.0 commands:

```bash
# Legacy duplicate detection (still works)
tokenslasher run config.yml

# Corpus comparison
tokenslasher diff old_corpus/ new_corpus/
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## 🏆 Citation

```bibtex
@software{tokenslasher,
  title={TokenSlasher: High-Speed Text Data Processing Pipeline},
  version={1.5.0},
  url={https://github.com/yassa777/tokenSlasher},
  year={2024}
}
```

---

**TokenSlasher V1.5** - From raw text to clean data in minutes, not hours. ⚡

TokenSlasher slashes redundant tokens from massive text corpora at warp speed. Train on 100% quality with 0% bloat.

<img width="1709" alt="image" src="https://github.com/user-attachments/assets/07f39dd7-82dc-466e-a26b-bb4bbe7e2dfa" />


| Dataset | Before Tokens | After Tokens | Tokens Saved | Processing Speed |
|---------|---------------|--------------|--------------|------------------|
| (fill)  |               |              |              |                  |

### 🚀 Throughput benchmark (single-core, M2 Pro)

| Dataset | Tokens | Speed (tok/s) |
|---------|--------|---------------|
| Enron 2 500 docs | **0.67 M** | **143 k** |

### 🎯 Classifier accuracy (Quora duplicate questions)

| Sample size | ROC-AUC | PR-AUC |
|-------------|---------|--------|
| 100 k pairs | **0.909** | **0.826** |

## Overview
Efficient near-duplicate detection and removal for large-scale text datasets using 6-gram MinHash + LSH.

## Quick Start
```
pip install tokenslasher            # from PyPI after wheel upload
# or, from source:
#   git clone https://github.com/yassa777/tokenSlasher && cd tokenslasher
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
