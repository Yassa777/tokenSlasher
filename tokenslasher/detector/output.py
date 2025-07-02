"""Output module for TokenSlasher V1.5.

Supports writing cleaned datasets to various formats:
- .jsonl (JSON Lines)
- .txt (plain text)
- .parquet (columnar format)
- Optional sharding for large datasets
- Metadata tracking (original hash, applied filters, etc.)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pd = None
    pa = None
    pq = None


class DatasetWriter:
    """Base class for dataset writers."""
    
    def __init__(self, output_path: Path, shard_size: Optional[int] = None):
        self.output_path = Path(output_path)
        self.shard_size = shard_size
        self.current_shard = 0
        self.current_shard_count = 0
        self.total_written = 0
        
    def write(self, record: Dict[str, Any]) -> None:
        """Write a single record."""
        raise NotImplementedError
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize writing and return stats."""
        raise NotImplementedError


class JSONLWriter(DatasetWriter):
    """Writer for JSONL format."""
    
    def __init__(self, output_path: Path, shard_size: Optional[int] = None, 
                 compress: bool = False):
        super().__init__(output_path, shard_size)
        self.compress = compress
        self.current_file = None
        self._open_new_shard()
    
    def _open_new_shard(self) -> None:
        """Open a new shard file."""
        if self.current_file:
            self.current_file.close()
        
        if self.shard_size:
            suffix = f".part{self.current_shard:04d}.jsonl"
        else:
            suffix = ".jsonl"
        
        if self.compress:
            suffix += ".gz"
        
        shard_path = self.output_path.with_suffix(suffix)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.compress:
            self.current_file = gzip.open(shard_path, 'wt', encoding='utf-8')
        else:
            self.current_file = open(shard_path, 'w', encoding='utf-8')
        
        self.current_shard_count = 0
    
    def write(self, record: Dict[str, Any]) -> None:
        """Write a single record as JSON line."""
        json.dump(record, self.current_file, ensure_ascii=False)
        self.current_file.write('\n')
        
        self.current_shard_count += 1
        self.total_written += 1
        
        if self.shard_size and self.current_shard_count >= self.shard_size:
            self.current_shard += 1
            self._open_new_shard()
    
    def finalize(self) -> Dict[str, Any]:
        """Close files and return stats."""
        if self.current_file:
            self.current_file.close()
        
        return {
            'format': 'jsonl',
            'total_records': self.total_written,
            'shards': self.current_shard + 1 if self.shard_size else 1,
            'compressed': self.compress
        }


class TextWriter(DatasetWriter):
    """Writer for plain text format."""
    
    def __init__(self, output_path: Path, shard_size: Optional[int] = None,
                 separator: str = '\n\n', include_metadata: bool = False):
        super().__init__(output_path, shard_size)
        self.separator = separator
        self.include_metadata = include_metadata
        self.current_file = None
        self._open_new_shard()
    
    def _open_new_shard(self) -> None:
        """Open a new shard file."""
        if self.current_file:
            self.current_file.close()
        
        if self.shard_size:
            suffix = f".part{self.current_shard:04d}.txt"
        else:
            suffix = ".txt"
        
        shard_path = self.output_path.with_suffix(suffix)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.current_file = open(shard_path, 'w', encoding='utf-8')
        self.current_shard_count = 0
    
    def write(self, record: Dict[str, Any]) -> None:
        """Write text content."""
        text = record.get('text', '')
        
        if self.include_metadata:
            metadata = {k: v for k, v in record.items() if k != 'text'}
            if metadata:
                self.current_file.write(f"# {json.dumps(metadata)}\n")
        
        self.current_file.write(text)
        self.current_file.write(self.separator)
        
        self.current_shard_count += 1
        self.total_written += 1
        
        if self.shard_size and self.current_shard_count >= self.shard_size:
            self.current_shard += 1
            self._open_new_shard()
    
    def finalize(self) -> Dict[str, Any]:
        """Close files and return stats."""
        if self.current_file:
            self.current_file.close()
        
        return {
            'format': 'txt',
            'total_records': self.total_written,
            'shards': self.current_shard + 1 if self.shard_size else 1,
            'separator': self.separator,
            'includes_metadata': self.include_metadata
        }


class ParquetWriter(DatasetWriter):
    """Writer for Parquet format."""
    
    def __init__(self, output_path: Path, shard_size: Optional[int] = None,
                 compression: str = 'snappy'):
        if not PARQUET_AVAILABLE:
            raise ImportError("pandas and pyarrow are required for Parquet output")
        super().__init__(output_path, shard_size)
        self.compression = compression
        self.buffer = []
        self.schema = None
    
    def write(self, record: Dict[str, Any]) -> None:
        """Buffer record for batch writing."""
        # Ensure consistent schema
        if self.schema is None:
            self.schema = list(record.keys())
        
        # Pad record with missing fields
        padded_record = {}
        for field in self.schema:
            padded_record[field] = record.get(field, None)
        
        # Add any new fields
        for field, value in record.items():
            if field not in self.schema:
                self.schema.append(field)
                # Backfill existing records
                for buffered in self.buffer:
                    if field not in buffered:
                        buffered[field] = None
            padded_record[field] = value
        
        self.buffer.append(padded_record)
        self.current_shard_count += 1
        self.total_written += 1
        
        if self.shard_size and self.current_shard_count >= self.shard_size:
            self._write_shard()
    
    def _write_shard(self) -> None:
        """Write current buffer to a shard file."""
        if not self.buffer:
            return
        
        df = pd.DataFrame(self.buffer)
        
        if self.shard_size:
            suffix = f".part{self.current_shard:04d}.parquet"
        else:
            suffix = ".parquet"
        
        shard_path = self.output_path.with_suffix(suffix)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(shard_path, compression=self.compression, index=False)
        
        self.buffer = []
        self.current_shard += 1
        self.current_shard_count = 0
    
    def finalize(self) -> Dict[str, Any]:
        """Write remaining buffer and return stats."""
        if self.buffer:
            self._write_shard()
        
        return {
            'format': 'parquet',
            'total_records': self.total_written,
            'shards': self.current_shard if self.shard_size else 1,
            'compression': self.compression,
            'schema': self.schema
        }


def create_writer(
    output_path: Union[str, Path],
    format: str = "auto",
    shard_size: Optional[int] = None,
    **kwargs
) -> DatasetWriter:
    """
    Create appropriate writer based on format.
    
    Args:
        output_path: Output file path
        format: Output format ('jsonl', 'txt', 'parquet', 'auto')
        shard_size: Number of records per shard (None for no sharding)
        **kwargs: Format-specific options
    """
    output_path = Path(output_path)
    
    if format == "auto":
        suffix = output_path.suffix.lower()
        if suffix in ['.jsonl', '.json']:
            format = "jsonl"
        elif suffix == '.txt':
            format = "txt"
        elif suffix == '.parquet':
            format = "parquet"
        else:
            raise ValueError(f"Cannot auto-detect format from suffix '{suffix}'")
    
    if format == "jsonl":
        return JSONLWriter(output_path, shard_size, **kwargs)
    elif format == "txt":
        return TextWriter(output_path, shard_size, **kwargs)
    elif format == "parquet":
        return ParquetWriter(output_path, shard_size, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_record(
    doc_id: str,
    text: str,
    source_file: Path,
    applied_filters: List[str] = None,
    filter_reasons: List[str] = None,
    original_hash: Optional[str] = None,
    **metadata
) -> Dict[str, Any]:
    """
    Create a standardized record for output.
    
    Args:
        doc_id: Unique document identifier
        text: Cleaned text content
        source_file: Original source file path
        applied_filters: List of filters that were applied
        filter_reasons: List of filter modification reasons
        original_hash: Hash of original text
        **metadata: Additional metadata fields
    """
    if original_hash is None:
        original_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    record = {
        'doc_id': doc_id,
        'text': text,
        'source_file': str(source_file),
        'original_hash': original_hash,
        'char_count': len(text),
        'word_count': len(text.split()),
        'applied_filters': applied_filters or [],
        'filter_reasons': filter_reasons or []
    }
    
    # Add any additional metadata
    record.update(metadata)
    
    return record


class ProcessingStats:
    """Track processing statistics."""
    
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.total_filtered = 0
        self.filter_stats = {}
        self.language_distribution = {}
        self.source_file_stats = {}
        self.char_count_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        self.word_count_stats = {'min': float('inf'), 'max': 0, 'total': 0}
    
    def add_input(self, source_file: Path) -> None:
        """Record input document."""
        self.total_input += 1
        source_str = str(source_file)
        self.source_file_stats[source_str] = self.source_file_stats.get(source_str, 0) + 1
    
    def add_output(self, record: Dict[str, Any]) -> None:
        """Record output document."""
        self.total_output += 1
        
        # Update character stats
        char_count = record.get('char_count', 0)
        self.char_count_stats['min'] = min(self.char_count_stats['min'], char_count)
        self.char_count_stats['max'] = max(self.char_count_stats['max'], char_count)
        self.char_count_stats['total'] += char_count
        
        # Update word stats
        word_count = record.get('word_count', 0)
        self.word_count_stats['min'] = min(self.word_count_stats['min'], word_count)
        self.word_count_stats['max'] = max(self.word_count_stats['max'], word_count)
        self.word_count_stats['total'] += word_count
    
    def add_filtered(self, reason: str) -> None:
        """Record filtered document."""
        self.total_filtered += 1
        self.filter_stats[reason] = self.filter_stats.get(reason, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing statistics summary."""
        if self.char_count_stats['min'] == float('inf'):
            self.char_count_stats['min'] = 0
        if self.word_count_stats['min'] == float('inf'):
            self.word_count_stats['min'] = 0
        
        return {
            'input_documents': self.total_input,
            'output_documents': self.total_output,
            'filtered_documents': self.total_filtered,
            'filter_rate': self.total_filtered / max(self.total_input, 1),
            'retention_rate': self.total_output / max(self.total_input, 1),
            'filter_breakdown': dict(self.filter_stats),
            'source_files': len(self.source_file_stats),
            'source_file_distribution': dict(self.source_file_stats),
            'character_stats': {
                'min': self.char_count_stats['min'],
                'max': self.char_count_stats['max'],
                'total': self.char_count_stats['total'],
                'average': self.char_count_stats['total'] / max(self.total_output, 1)
            },
            'word_stats': {
                'min': self.word_count_stats['min'],
                'max': self.word_count_stats['max'],
                'total': self.word_count_stats['total'],
                'average': self.word_count_stats['total'] / max(self.total_output, 1)
            }
        }
    
    def save_stats(self, output_path: Path) -> None:
        """Save statistics to JSON file."""
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)


def estimate_shard_size(
    total_records: int,
    target_file_size_mb: int = 100,
    avg_record_size_bytes: int = 1000
) -> int:
    """
    Estimate optimal shard size based on target file size.
    
    Args:
        total_records: Total number of records
        target_file_size_mb: Target file size in MB
        avg_record_size_bytes: Average record size in bytes
    
    Returns:
        Estimated shard size
    """
    target_size_bytes = target_file_size_mb * 1024 * 1024
    records_per_shard = target_size_bytes // avg_record_size_bytes
    
    # Ensure at least 1000 records per shard and not more than total
    records_per_shard = max(1000, min(records_per_shard, total_records))
    
    return records_per_shard 