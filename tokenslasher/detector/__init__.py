"""TokenSlasher detector package.

Core public API lives here so external users can::

    import tokenslasher as ts
    ts.SENTINEL
    ts.__version__
    
V1.5 includes comprehensive data processing pipeline:
    from tokenslasher.detector.pipeline import run_pipeline
    from tokenslasher.detector.filters import create_default_pipeline
    from tokenslasher.detector.file_ingest import ingest_files
    from tokenslasher.detector.output import create_writer
"""

from importlib.metadata import version as _pkg_version


# Semantic version of the installed package
try:
    __version__: str = _pkg_version("tokenslasher")
except Exception:  # pragma: no cover – local dev path
    __version__ = "1.5.0"


# Special token inserted between documents so shingles never span boundaries
SENTINEL: str = "<doc>"

# V1.5 Public API
from .pipeline import run_pipeline, preview_input, estimate_processing_time
from .filters import create_default_pipeline, FilterPipeline
from .file_ingest import ingest_files, get_file_stats
from .output import create_writer, create_record, ProcessingStats

__all__ = [
    "SENTINEL",
    "__version__",
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