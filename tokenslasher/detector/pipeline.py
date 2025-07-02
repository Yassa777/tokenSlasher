"""Main data processing pipeline for TokenSlasher V1.5.

Integrates:
- File ingestion (multiple formats)
- Data cleaning and filtering
- Deduplication
- Output in various formats
- Comprehensive metrics and logging
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

import psutil
from tqdm import tqdm

from . import SENTINEL
from .file_ingest import ingest_files, get_file_stats
from .filters import FilterPipeline, create_default_pipeline
from .output import create_writer, create_record, ProcessingStats
from .lsh_index import LSHIndex
from .minhash import batch_xxhash64, compute_minhash_from_hashes
from .ingest import char_ngrams, token_skipgrams


class TokenSlasherPipeline:
    """Complete data processing pipeline."""
    
    def __init__(
        self,
        input_paths: Union[str, Path, List[Union[str, Path]]],
        output_path: Union[str, Path],
        filter_pipeline: Optional[FilterPipeline] = None,
        enable_dedup: bool = True,
        dedup_threshold: float = 0.8,
        output_format: str = "auto",
        shard_size: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            input_paths: Input files/directories/patterns
            output_path: Output file path
            filter_pipeline: Custom filter pipeline (uses default if None)
            enable_dedup: Whether to perform deduplication
            dedup_threshold: Similarity threshold for deduplication
            output_format: Output format ('jsonl', 'txt', 'parquet', 'auto')
            shard_size: Records per shard (None for no sharding)
            verbose: Enable verbose logging
        """
        self.input_paths = input_paths
        self.output_path = Path(output_path)
        self.filter_pipeline = filter_pipeline or create_default_pipeline()
        self.enable_dedup = enable_dedup
        self.dedup_threshold = dedup_threshold
        self.output_format = output_format
        self.shard_size = shard_size
        self.verbose = verbose
        
        # Stats tracking
        self.stats = ProcessingStats()
        self.start_time = None
        self.end_time = None
        
        # Deduplication state
        if enable_dedup:
            self.lsh_index = LSHIndex(threshold=dedup_threshold, bands=64, rows=2)
            self.doc_cache = {}  # doc_id -> shingle hash set
            self.duplicates_found = []
        
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        self.start_time = time.time()
        
        if self.verbose:
            print("🚀 Starting TokenSlasher V1.5 Pipeline")
            self._print_input_stats()
        
        # Create output writer
        writer = create_writer(
            self.output_path, 
            format=self.output_format,
            shard_size=self.shard_size
        )
        
        try:
            # Process documents
            self._process_documents(writer)
            
            # Finalize output
            output_stats = writer.finalize()
            
            self.end_time = time.time()
            
            # Generate final report
            final_stats = self._generate_final_stats(output_stats)
            
            if self.verbose:
                self._print_final_stats(final_stats)
            
            # Save detailed stats
            self.stats.save_stats(self.output_path)
            
            return final_stats
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Pipeline failed: {e}")
            raise
        finally:
            # Ensure writer is closed
            try:
                writer.finalize()
            except:
                pass
    
    def _print_input_stats(self) -> None:
        """Print input statistics."""
        file_stats = get_file_stats(self.input_paths)
        print(f"📁 Input: {file_stats['total_files']} files "
              f"({file_stats['total_size_bytes'] / 1024 / 1024:.1f} MB)")
        print(f"   - TXT: {file_stats['txt_files']}")
        print(f"   - JSONL: {file_stats['jsonl_files']}")
        print(f"   - HTML: {file_stats['html_files']}")
        print(f"   - GZ: {file_stats['gz_files']}")
        print()
    
    def _process_documents(self, writer) -> None:
        """Process all documents through the pipeline."""
        doc_generator = ingest_files(
            self.input_paths, 
            show_progress=self.verbose
        )
        
        if self.verbose:
            doc_generator = tqdm(doc_generator, desc="Processing documents")
        
        for doc_data in doc_generator:
            self.stats.add_input(doc_data['source_file'])
            
            # Apply filters
            should_keep, filtered_text, filter_reasons = self.filter_pipeline.apply(
                doc_data['text']
            )
            
            if not should_keep:
                self.stats.add_filtered(filter_reasons[0] if filter_reasons else "unknown")
                continue
            
            # Check for duplicates if enabled
            if self.enable_dedup:
                is_duplicate, duplicate_of = self._check_duplicate(
                    doc_data['doc_id'], 
                    filtered_text
                )
                
                if is_duplicate:
                    self.stats.add_filtered(f"duplicate_of_{duplicate_of}")
                    continue
            
            # Create output record
            original_hash = hashlib.sha256(doc_data['text'].encode('utf-8')).hexdigest()[:16]
            
            record = create_record(
                doc_id=doc_data['doc_id'],
                text=filtered_text,
                source_file=doc_data['source_file'],
                applied_filters=[f.name for f in self.filter_pipeline.filters],
                filter_reasons=filter_reasons,
                original_hash=original_hash
            )
            
            # Write record
            writer.write(record)
            self.stats.add_output(record)
    
    def _check_duplicate(self, doc_id: str, text: str) -> tuple[bool, Optional[str]]:
        """Check if document is a duplicate."""
        # Generate shingles
        shingles_str = set(char_ngrams(text, 5)).union(
            set(token_skipgrams(text.split(), skip=1))
        )
        
        # Hash shingles
        shingle_hashes = batch_xxhash64(list(shingles_str))
        shingle_set = set(shingle_hashes)
        
        # Store in cache
        self.doc_cache[doc_id] = shingle_set
        
        # Compute MinHash
        minhash = compute_minhash_from_hashes(shingle_hashes)
        
        # Check LSH candidates
        candidates = self.lsh_index.get_candidates(minhash)
        
        for candidate in candidates:
            if candidate in self.doc_cache:
                # Compute exact Jaccard similarity
                jaccard = self._jaccard_similarity(shingle_set, self.doc_cache[candidate])
                
                if jaccard >= self.dedup_threshold:
                    self.duplicates_found.append({
                        'doc_id': doc_id,
                        'duplicate_of': candidate,
                        'similarity': jaccard
                    })
                    return True, candidate
        
        # Add to LSH index
        self.lsh_index.add(doc_id, minhash)
        
        return False, None
    
    def _jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    def _generate_final_stats(self, output_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final statistics."""
        elapsed_time = self.end_time - self.start_time
        
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get filter statistics
        filter_stats = self.filter_pipeline.get_stats()
        
        final_stats = {
            'pipeline_version': '1.5.0',
            'processing_time_seconds': elapsed_time,
            'peak_memory_mb': memory_mb,
            'input_stats': {
                'total_files': len(set(self.stats.source_file_stats.keys())),
                'total_documents': self.stats.total_input
            },
            'filtering_stats': {
                'documents_passed': self.stats.total_output,
                'documents_filtered': self.stats.total_filtered,
                'filter_rate': self.stats.total_filtered / max(self.stats.total_input, 1),
                'filter_breakdown': dict(self.stats.filter_stats),
                'filter_details': filter_stats
            },
            'output_stats': output_stats,
            'performance': {
                'documents_per_second': self.stats.total_input / max(elapsed_time, 1),
                'characters_per_second': self.stats.char_count_stats['total'] / max(elapsed_time, 1),
                'throughput_mb_per_second': (self.stats.char_count_stats['total'] / 1024 / 1024) / max(elapsed_time, 1)
            }
        }
        
        # Add deduplication stats if enabled
        if self.enable_dedup:
            final_stats['deduplication_stats'] = {
                'enabled': True,
                'threshold': self.dedup_threshold,
                'duplicates_found': len(self.duplicates_found),
                'duplicate_rate': len(self.duplicates_found) / max(self.stats.total_input, 1),
                'unique_documents': self.stats.total_output
            }
        else:
            final_stats['deduplication_stats'] = {'enabled': False}
        
        # Add detailed statistics
        final_stats['detailed_stats'] = self.stats.get_summary()
        
        return final_stats
    
    def _print_final_stats(self, stats: Dict[str, Any]) -> None:
        """Print final statistics summary."""
        print("\n" + "="*60)
        print("📊 TOKENSLASHER V1.5 PROCESSING COMPLETE")
        print("="*60)
        
        # Basic stats
        print(f"⏱️  Processing Time: {stats['processing_time_seconds']:.2f}s")
        print(f"💾 Peak Memory: {stats['peak_memory_mb']:.1f} MB")
        print()
        
        # Input/Output
        input_stats = stats['input_stats']
        filtering_stats = stats['filtering_stats']
        
        print(f"📥 Input: {input_stats['total_documents']:,} documents")
        print(f"📤 Output: {filtering_stats['documents_passed']:,} documents")
        print(f"🗑️  Filtered: {filtering_stats['documents_filtered']:,} documents "
              f"({filtering_stats['filter_rate']:.1%})")
        print()
        
        # Top filter reasons
        print("🔍 Top Filter Reasons:")
        filter_breakdown = filtering_stats['filter_breakdown']
        for reason, count in sorted(filter_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = count / max(input_stats['total_documents'], 1) * 100
            print(f"   - {reason}: {count:,} ({percentage:.1f}%)")
        print()
        
        # Deduplication
        if stats['deduplication_stats']['enabled']:
            dedup_stats = stats['deduplication_stats']
            print(f"🔄 Deduplication: {dedup_stats['duplicates_found']:,} duplicates removed "
                  f"({dedup_stats['duplicate_rate']:.1%})")
            print()
        
        # Performance
        perf = stats['performance']
        print(f"🚀 Performance:")
        print(f"   - {perf['documents_per_second']:.0f} docs/sec")
        print(f"   - {perf['characters_per_second']:.0f} chars/sec")
        print(f"   - {perf['throughput_mb_per_second']:.1f} MB/sec")
        print()
        
        # Output info
        output_stats = stats['output_stats']
        print(f"💾 Output: {output_stats['format'].upper()} format")
        if output_stats.get('shards', 1) > 1:
            print(f"   - {output_stats['shards']} shards")
        if output_stats.get('compressed'):
            print(f"   - Compressed")
        print(f"   - {output_stats['total_records']:,} records")
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Output saved to: {self.output_path}")


def run_pipeline(
    input_paths: Union[str, Path, List[Union[str, Path]]],
    output_path: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        input_paths: Input files/directories/patterns
        output_path: Output file path
        **kwargs: Additional pipeline configuration
    
    Returns:
        Processing statistics
    """
    pipeline = TokenSlasherPipeline(input_paths, output_path, **kwargs)
    return pipeline.run()


def preview_input(
    input_paths: Union[str, Path, List[Union[str, Path]]],
    max_samples: int = 5
) -> None:
    """
    Preview input data without processing.
    
    Args:
        input_paths: Input files/directories/patterns  
        max_samples: Maximum number of sample documents to show
    """
    print("🔍 Input Preview")
    print("="*40)
    
    # Show file stats
    file_stats = get_file_stats(input_paths)
    print(f"Files found: {file_stats['total_files']}")
    print(f"Total size: {file_stats['total_size_bytes'] / 1024 / 1024:.1f} MB")
    print()
    
    # Show sample documents
    print(f"Sample documents (up to {max_samples}):")
    print("-" * 40)
    
    count = 0
    for doc_data in ingest_files(input_paths, show_progress=False):
        if count >= max_samples:
            break
            
        print(f"\nDocument {count + 1}:")
        print(f"  ID: {doc_data['doc_id']}")
        print(f"  Source: {doc_data['source_file']}")
        print(f"  Length: {len(doc_data['text'])} chars")
        print(f"  Preview: {doc_data['text'][:200]}...")
        
        count += 1
    
    if count == 0:
        print("No documents found!")
    else:
        print(f"\nShowing {count} of {file_stats['total_files']} files")


def estimate_processing_time(
    input_paths: Union[str, Path, List[Union[str, Path]]],
    docs_per_second: float = 100.0
) -> Dict[str, float]:
    """
    Estimate processing time for given input.
    
    Args:
        input_paths: Input files/directories/patterns
        docs_per_second: Estimated processing speed
        
    Returns:
        Time estimates in various units
    """
    file_stats = get_file_stats(input_paths)
    
    # Rough estimate: assume average 1KB per document
    estimated_docs = file_stats['total_size_bytes'] // 1024
    estimated_seconds = estimated_docs / docs_per_second
    
    return {
        'estimated_documents': estimated_docs,
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_seconds / 60,
        'estimated_hours': estimated_seconds / 3600,
        'processing_speed_docs_per_sec': docs_per_second
    }
