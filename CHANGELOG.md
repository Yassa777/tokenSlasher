# TokenSlasher Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2024-07-02

### 🚀 Major Features

#### Multi-Format File Ingestion
- **NEW**: Support for multiple input formats (`.txt`, `.jsonl`, `.html`, `.gz`)
- **NEW**: Automatic encoding detection using `chardet`
- **NEW**: Streaming file processing with progress bars
- **NEW**: Glob pattern support for batch file processing
- **NEW**: File statistics preview functionality

#### Comprehensive Text Filtering Pipeline
- **NEW**: `LanguageFilter` - Detect and filter by language using `langdetect`
- **NEW**: `JunkFilter` - Quality-based filtering (length, word count, repetition, etc.)
- **NEW**: `HTMLStripFilter` - Clean HTML content using BeautifulSoup
- **NEW**: `PerplexityFilter` - Text quality assessment using character frequency
- **NEW**: `PIIFilter` - Detect and anonymize personally identifiable information
- **NEW**: `ToxicityFilter` - Filter toxic content (heuristic and ML-based options)
- **NEW**: `WhitespaceNormalizer` - Unicode and whitespace normalization
- **NEW**: `FilterPipeline` - Chain multiple filters with comprehensive statistics
- **NEW**: `create_default_pipeline()` - Factory function for standard filter combinations

#### Advanced Output Handling
- **NEW**: Multiple output formats: JSONL, plain text, and Parquet
- **NEW**: Automatic sharding for large datasets
- **NEW**: Configurable compression options
- **NEW**: Rich metadata tracking in output records
- **NEW**: Comprehensive processing statistics
- **NEW**: Memory usage monitoring

#### Enhanced CLI Interface
- **NEW**: `clean` command - Main V1.5 processing pipeline
- **NEW**: `preview` command - Preview input data and processing estimates
- **IMPROVED**: Rich argument parsing with extensive configuration options
- **MAINTAINED**: `run` command - Legacy V1.0 compatibility
- **MAINTAINED**: `diff` command - Corpus comparison functionality

### 🏗️ Architecture Improvements

#### Package Structure
- **RESTRUCTURED**: Organized into `tokenslasher.detector` package hierarchy
- **NEW**: Modular design with separate modules for each major component
- **NEW**: `file_ingest.py` - File reading and format detection
- **NEW**: `filters.py` - Text filtering pipeline components  
- **NEW**: `output.py` - Output format handlers and statistics
- **NEW**: `pipeline.py` - Main orchestration and workflow management
- **ENHANCED**: `minhash.py` - Improved hash computation with better fallbacks
- **ENHANCED**: `lsh_index.py` - Optimized LSH indexing with configurable parameters

#### Performance Optimizations
- **NEW**: Streaming processing architecture for memory efficiency
- **NEW**: Batch processing optimizations
- **NEW**: Memory usage tracking with `psutil`
- **NEW**: Progress reporting with `tqdm`
- **IMPROVED**: MinHash computation with better xxHash integration

#### Error Handling & Robustness
- **NEW**: Graceful handling of encoding issues and malformed files
- **NEW**: Fallback mechanisms for optional dependencies
- **NEW**: Comprehensive error logging and statistics
- **NEW**: Configurable failure modes (skip vs. stop on errors)

### 📦 Dependencies

#### New Dependencies
- `langdetect>=1.0.9` - Language detection
- `beautifulsoup4>=4.12` - HTML parsing and cleaning
- `presidio-analyzer>=2.2` - PII detection
- `presidio-anonymizer>=2.2` - PII anonymization
- `transformers>=4.30` - ML model support
- `pyarrow>=14.0` - Parquet format support
- `fastparquet>=2023.10` - Alternative Parquet implementation
- `chardet>=5.2` - Character encoding detection
- `lxml>=4.9` - XML/HTML parsing backend

#### Optional Dependencies
- `detoxify>=0.5.2` - ML-based toxicity detection (optional)
- `torch` - PyTorch for advanced ML features (optional)
- `streamlit>=1.28` - Dashboard functionality (optional)
- `plotly>=5.18` - Advanced visualizations (optional)

### 🛠️ Configuration

#### Enhanced Configuration Options
- **NEW**: Extensive CLI argument support for all pipeline stages
- **NEW**: Filter-specific configuration options
- **NEW**: Output format and sharding configuration
- **NEW**: Performance tuning parameters
- **MAINTAINED**: YAML configuration support for backward compatibility

### 📊 Statistics & Monitoring

#### Comprehensive Analytics
- **NEW**: Detailed filter performance metrics
- **NEW**: Processing time estimation
- **NEW**: Memory usage tracking
- **NEW**: File-level and document-level statistics
- **NEW**: Filter pass/fail rates and reason tracking
- **NEW**: Deduplication effectiveness metrics

### 🔄 Backward Compatibility

#### V1.0 Compatibility
- **MAINTAINED**: All V1.0 CLI commands and APIs remain functional
- **MAINTAINED**: YAML configuration file support
- **MAINTAINED**: Original MinHash + LSH deduplication algorithm
- **MAINTAINED**: Basic text output format
- **MAINTAINED**: Core duplicate detection functionality

### 🐛 Bug Fixes

- **FIXED**: Memory leaks in large file processing
- **FIXED**: Unicode handling edge cases
- **FIXED**: Progress bar accuracy with streaming processing
- **IMPROVED**: Error messages and logging clarity
- **ENHANCED**: File path handling on different operating systems

### 📚 Documentation

- **REWRITTEN**: Comprehensive README with V1.5 examples
- **NEW**: Performance benchmarks and comparison tables
- **NEW**: Usage examples for each major feature
- **NEW**: Migration guide from V1.0 to V1.5
- **ENHANCED**: API documentation with type hints
- **NEW**: Contributing guidelines and development setup

### 🧪 Testing

- **NEW**: Comprehensive test suite for V1.5 functionality
- **NEW**: End-to-end pipeline testing
- **NEW**: Performance regression tests
- **NEW**: Integration tests for all supported formats
- **MAINTAINED**: V1.0 compatibility test suite

---

## [1.0.0] - 2024-06-21

### Initial Release

#### Core Features
- MinHash-based near-duplicate detection
- LSH (Locality-Sensitive Hashing) indexing
- Basic text file processing
- YAML-based configuration
- Simple CLI interface
- Character n-gram generation
- Jaccard similarity computation

#### Architecture
- Modular design with separate components
- Memory-efficient batch processing
- Configurable similarity thresholds
- Basic statistics reporting

---

## Upgrade Guide

### From V1.0 to V1.5

#### Quick Migration
```bash
# V1.0 (still works)
tokenslasher run config.yaml

# V1.5 equivalent
tokenslasher clean input_files/ --output cleaned.jsonl --format jsonl
```

#### New Recommended Workflow
```bash
# Preview your data
tokenslasher preview input_files/

# Process with full pipeline
tokenslasher clean input_files/ \
  --output cleaned.jsonl \
  --format jsonl \
  --filters language html junk pii \
  --languages en \
  --min-length 50 \
  --dedupe-threshold 0.8
```

#### Configuration Changes
- CLI-first approach (YAML still supported)
- More granular control over filtering
- Enhanced output format options
- Better performance monitoring

For detailed migration instructions, see the README.md file. 