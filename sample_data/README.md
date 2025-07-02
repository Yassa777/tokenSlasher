# Sample Data for TokenSlasher V1.5 Testing

This directory contains sample data files to test TokenSlasher V1.5 functionality.

## Files

- `sample_texts.txt` - Clean English text samples
- `mixed_quality.jsonl` - JSONL with varying quality text  
- `multilingual.txt` - Text in multiple languages
- `with_html.html` - HTML content to test stripping
- `short_docs.txt` - Short documents (should be filtered)
- `test_with_pii.txt` - Contains PII for anonymization testing

## Usage

```bash
# Test the complete pipeline
tokenslasher clean sample_data/ --output test_output.jsonl

# Preview the sample data  
tokenslasher preview sample_data/

# Test with different filters
tokenslasher clean sample_data/ --output filtered.jsonl --min-length 100 --languages en
``` 