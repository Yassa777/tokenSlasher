"""File ingestion module for TokenSlasher V1.5.

Supports streaming ingestion of:
- .txt, .jsonl, .html files  
- .gz compressed files
- Folder traversal with glob patterns
- Large file streaming with chunking
"""
from __future__ import annotations

import gzip
import json
import mimetypes
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import chardet
from bs4 import BeautifulSoup
from tqdm import tqdm


def detect_encoding(file_path: Path, sample_size: int = 8192) -> str:
    """Detect file encoding using chardet."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8') or 'utf-8'
    except Exception:
        return 'utf-8'


def read_text_file(file_path: Path, encoding: Optional[str] = None) -> Generator[str, None, None]:
    """Read a text file line by line with encoding detection."""
    if encoding is None:
        encoding = detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")


def read_jsonl_file(file_path: Path, text_fields: List[str] = None) -> Generator[str, None, None]:
    """Read a JSONL file and extract text from specified fields."""
    if text_fields is None:
        text_fields = ['text', 'content', 'body', 'message', 'document']
    
    encoding = detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    
                    # Extract text from specified fields
                    text_parts = []
                    for field in text_fields:
                        if field in obj and obj[field]:
                            text_parts.append(str(obj[field]))
                    
                    if text_parts:
                        yield " ".join(text_parts)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at {file_path}:{line_num}: {e}")
                    
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")


def read_html_file(file_path: Path, encoding: Optional[str] = None) -> Generator[str, None, None]:
    """Read an HTML file and extract text content."""
    if encoding is None:
        encoding = detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
            
        soup = BeautifulSoup(content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if text:
            yield text
            
    except Exception as e:
        print(f"Warning: Failed to read HTML {file_path}: {e}")


def read_gz_file(file_path: Path, text_fields: List[str] = None) -> Generator[str, None, None]:
    """Read a gzipped file, auto-detecting the inner format."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as f:
            # Try to detect if it's JSONL
            first_line = f.readline().strip()
            f.seek(0)
            
            if first_line.startswith('{'):
                # Looks like JSONL
                if text_fields is None:
                    text_fields = ['text', 'content', 'body', 'message', 'document']
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                        text_parts = []
                        for field in text_fields:
                            if field in obj and obj[field]:
                                text_parts.append(str(obj[field]))
                        
                        if text_parts:
                            yield " ".join(text_parts)
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at {file_path}:{line_num}: {e}")
            else:
                # Treat as plain text
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
                        
    except Exception as e:
        print(f"Warning: Failed to read gzipped {file_path}: {e}")


def ingest_files(
    paths: Union[str, Path, List[Union[str, Path]]], 
    recursive: bool = True,
    show_progress: bool = True
) -> Generator[Dict[str, Union[str, Path]], None, None]:
    """
    Ingest files from paths (files, directories, or glob patterns).
    
    Args:
        paths: Single path, list of paths, or glob patterns
        recursive: Whether to search directories recursively
        show_progress: Whether to show progress bar
        
    Yields:
        Dict with keys: 'text', 'source_file', 'doc_id'
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    # Collect all files
    all_files = []
    for path in paths:
        path = Path(path)
        
        if '*' in str(path) or '?' in str(path):
            # Glob pattern
            if recursive:
                all_files.extend(Path('.').glob(str(path)))
            else:
                all_files.extend(Path('.').glob(str(path)))
        elif path.is_file():
            all_files.append(path)
        elif path.is_dir():
            if recursive:
                all_files.extend(path.rglob('*'))
            else:
                all_files.extend(path.glob('*'))
    
    # Filter to supported file types
    supported_extensions = {'.txt', '.jsonl', '.html', '.htm', '.gz'}
    files = [f for f in all_files if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not files:
        print("Warning: No supported files found")
        return
    
    doc_id = 0
    
    if show_progress:
        files = tqdm(files, desc="Processing files")
    
    for file_path in files:
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                for text in read_text_file(file_path):
                    yield {
                        'text': text,
                        'source_file': file_path,
                        'doc_id': f"doc_{doc_id}"
                    }
                    doc_id += 1
                    
            elif suffix == '.jsonl':
                for text in read_jsonl_file(file_path):
                    yield {
                        'text': text,
                        'source_file': file_path,
                        'doc_id': f"doc_{doc_id}"
                    }
                    doc_id += 1
                    
            elif suffix in {'.html', '.htm'}:
                for text in read_html_file(file_path):
                    yield {
                        'text': text,
                        'source_file': file_path,
                        'doc_id': f"doc_{doc_id}"
                    }
                    doc_id += 1
                    
            elif suffix == '.gz':
                for text in read_gz_file(file_path):
                    yield {
                        'text': text,
                        'source_file': file_path,
                        'doc_id': f"doc_{doc_id}"
                    }
                    doc_id += 1
                    
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue


def get_file_stats(paths: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, int]:
    """Get statistics about files that would be processed."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    stats = {
        'total_files': 0,
        'txt_files': 0,
        'jsonl_files': 0,
        'html_files': 0,
        'gz_files': 0,
        'total_size_bytes': 0
    }
    
    all_files = []
    for path in paths:
        path = Path(path)
        
        if '*' in str(path) or '?' in str(path):
            all_files.extend(Path('.').glob(str(path)))
        elif path.is_file():
            all_files.append(path)
        elif path.is_dir():
            all_files.extend(path.rglob('*'))
    
    supported_extensions = {'.txt', '.jsonl', '.html', '.htm', '.gz'}
    
    for file_path in all_files:
        if not file_path.is_file():
            continue
            
        suffix = file_path.suffix.lower()
        if suffix not in supported_extensions:
            continue
            
        stats['total_files'] += 1
        stats['total_size_bytes'] += file_path.stat().st_size
        
        if suffix == '.txt':
            stats['txt_files'] += 1
        elif suffix == '.jsonl':
            stats['jsonl_files'] += 1
        elif suffix in {'.html', '.htm'}:
            stats['html_files'] += 1
        elif suffix == '.gz':
            stats['gz_files'] += 1
    
    return stats 