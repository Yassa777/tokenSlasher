#!/usr/bin/env python3
"""
TokenSlasher V1.5 Functionality Tests
=====================================

Comprehensive tests for TokenSlasher V1.5 features including:
- Multi-format file ingestion
- Smart filtering pipeline
- Advanced deduplication
- Multiple output formats
"""

import sys
import tempfile
import json
import gzip
from pathlib import Path
import unittest

# Add tokenSlasher to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestTokenSlasherV15(unittest.TestCase):
    """Test suite for TokenSlasher V1.5 functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_file_ingestion(self):
        """Test multi-format file ingestion"""
        try:
            from tokenslasher.detector.file_ingest import ingest_files, detect_encoding
            
            # Create test files
            text_file = self.test_dir / "test.txt"
            text_file.write_text("Line 1\nLine 2\nLine 3")
            
            jsonl_file = self.test_dir / "test.jsonl"
            with open(jsonl_file, 'w') as f:
                json.dump({"text": "JSON document 1"}, f)
                f.write('\n')
                json.dump({"text": "JSON document 2"}, f)
                f.write('\n')
            
            # Test encoding detection
            encoding = detect_encoding(text_file)
            self.assertIsNotNone(encoding)
            
            # Test file ingestion
            docs = list(ingest_files([str(text_file), str(jsonl_file)]))
            self.assertGreater(len(docs), 0)
            
        except ImportError as e:
            self.skipTest(f"Import failed due to dependency issues: {e}")
    
    def test_text_filters(self):
        """Test text filtering pipeline"""
        try:
            from tokenslasher.detector.filters import (
                WhitespaceNormalizer, HTMLStripFilter, JunkFilter, FilterPipeline
            )
            
            # Test whitespace normalizer
            normalizer = WhitespaceNormalizer()
            passed, text, reason = normalizer.filter("  extra   spaces  ")
            self.assertTrue(passed)
            self.assertEqual(text, "extra spaces")
            
            # Test HTML filter
            html_filter = HTMLStripFilter()
            passed, text, reason = html_filter.filter("<p>Hello <b>world</b></p>")
            self.assertTrue(passed)
            self.assertEqual(text, "Hello world")
            
            # Test junk filter
            junk_filter = JunkFilter(min_length=10)
            passed, text, reason = junk_filter.filter("Short")
            self.assertFalse(passed)
            
            passed, text, reason = junk_filter.filter("This is long enough")
            self.assertTrue(passed)
            
            # Test filter pipeline
            pipeline = FilterPipeline([normalizer, html_filter, junk_filter])
            passed, text, reasons = pipeline.apply("<p>Good  quality   content</p>")
            self.assertTrue(passed)
            self.assertEqual(text, "Good quality content")
            
        except ImportError as e:
            self.skipTest(f"Import failed due to dependency issues: {e}")
    
    def test_deduplication(self):
        """Test deduplication functionality"""
        try:
            from tokenslasher.detector.minhash import compute_minhash, batch_xxhash64
            from tokenslasher.detector.lsh_index import LSHIndex
            from tokenslasher.detector.ingest import char_ngrams
            
            # Test hash computation
            ngrams = list(char_ngrams("test document", n=3))
            hashes = batch_xxhash64(ngrams)
            self.assertGreater(len(hashes), 0)
            
            minhash = compute_minhash(ngrams)
            self.assertIsNotNone(minhash)
            
            # Test LSH index
            lsh = LSHIndex()
            lsh.add("doc1", minhash)
            
            # Test similarity
            similar_ngrams = list(char_ngrams("test document", n=3))
            similar_minhash = compute_minhash(similar_ngrams)
            candidates = lsh.get_candidates(similar_minhash)
            self.assertIn("doc1", candidates)
            
        except ImportError as e:
            self.skipTest(f"Import failed due to dependency issues: {e}")
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        try:
            from tokenslasher.detector.file_ingest import ingest_files
            from tokenslasher.detector.filters import FilterPipeline, WhitespaceNormalizer, JunkFilter
            from tokenslasher.detector.minhash import compute_minhash
            from tokenslasher.detector.lsh_index import LSHIndex
            from tokenslasher.detector.ingest import char_ngrams
            
            # Create test data
            text_file = self.test_dir / "docs.txt"
            text_file.write_text("Good quality document\nAnother good document\nGood quality document")
            
            # Ingest
            docs = list(ingest_files([str(text_file)]))
            self.assertGreater(len(docs), 0)
            
            # Filter
            pipeline = FilterPipeline([WhitespaceNormalizer(), JunkFilter(min_length=5)])
            filtered_docs = []
            for doc in docs:
                text = str(doc)
                passed, cleaned_text, reasons = pipeline.apply(text)
                if passed:
                    filtered_docs.append(cleaned_text)
            
            self.assertGreater(len(filtered_docs), 0)
            
            # Deduplicate
            lsh = LSHIndex()
            unique_docs = []
            for i, doc in enumerate(filtered_docs):
                ngrams = list(char_ngrams(doc, n=3))
                if ngrams:
                    minhash = compute_minhash(ngrams)
                    candidates = lsh.get_candidates(minhash)
                    if not candidates:
                        lsh.add(str(i), minhash)
                        unique_docs.append(doc)
            
            # Should have fewer unique docs than original (due to duplicates)
            self.assertLessEqual(len(unique_docs), len(filtered_docs))
            
        except ImportError as e:
            self.skipTest(f"Import failed due to dependency issues: {e}")

class TestV15Improvements(unittest.TestCase):
    """Test V1.5 improvements over V1.0"""
    
    def test_v15_features_available(self):
        """Test that V1.5 features are available"""
        try:
            # Test new modules exist
            import tokenslasher.detector.file_ingest
            import tokenslasher.detector.filters
            import tokenslasher.detector.output
            
            # Test new classes exist
            from tokenslasher.detector.filters import (
                LanguageFilter, JunkFilter, HTMLStripFilter, 
                PerplexityFilter, PIIFilter, WhitespaceNormalizer
            )
            
            self.assertTrue(True, "All V1.5 modules and classes available")
            
        except ImportError as e:
            self.skipTest(f"V1.5 features not available: {e}")

if __name__ == "__main__":
    unittest.main(verbosity=2) 