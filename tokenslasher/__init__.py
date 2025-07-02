"""TokenSlasher - High-speed text data cleaning and deduplication pipeline.

TokenSlasher V1.5 provides comprehensive data processing for massive text corpora:
- Multi-format ingestion (.txt, .jsonl, .html, .gz)
- Smart filtering (language, quality, PII, toxicity)
- Advanced deduplication (MinHash + LSH)
- Flexible output formats (JSONL, TXT, Parquet)

Quick Start:
    # CLI usage
    tokenslasher clean data/ --output clean_data.jsonl
    
    # Python API
    from tokenslasher.detector import run_pipeline
    stats = run_pipeline(['data/'], 'clean_data.jsonl')
"""

from .detector import __version__, SENTINEL

# Re-export main API
from .detector import (
    run_pipeline,
    preview_input,
    estimate_processing_time,
    create_default_pipeline,
    FilterPipeline,
    ingest_files,
    get_file_stats,
    create_writer,
    create_record,
    ProcessingStats
)

__all__ = [
    "__version__",
    "SENTINEL",
    # V1.5 Pipeline API
    "run_pipeline",
    "preview_input",
    "estimate_processing_time", 
    "create_default_pipeline",
    "FilterPipeline",
    "ingest_files",
    "get_file_stats",
    "create_writer",
    "create_record",
    "ProcessingStats"
] 