"""Data cleaning and filtering module for TokenSlasher V1.5.

Provides filters for:
- Language detection
- Junk filtering (short docs, malformed text, non-printable chars)  
- Perplexity filtering
- HTML stripping
- PII detection and removal
- Toxicity detection
- Whitespace normalization
"""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException


class TextFilter:
    """Base class for text filters."""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = {
            'passed': 0,
            'filtered': 0,
            'reasons': {}
        }
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        """
        Filter text.
        
        Returns:
            (should_keep, filtered_text, reason_if_filtered)
        """
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        total = self.stats['passed'] + self.stats['filtered']
        return {
            'name': self.name,
            'passed': self.stats['passed'],
            'filtered': self.stats['filtered'],
            'total': total,
            'filter_rate': self.stats['filtered'] / max(total, 1),
            'reasons': dict(self.stats['reasons'])
        }


class LanguageFilter(TextFilter):
    """Filter by detected language."""
    
    def __init__(self, allowed_languages: Set[str] = None, min_confidence: float = 0.9):
        super().__init__("language")
        self.allowed_languages = allowed_languages or {'en'}
        self.min_confidence = min_confidence
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text or len(text.strip()) < 10:
            self.stats['filtered'] += 1
            reason = "too_short_for_detection"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        try:
            # Simple detection for short texts
            if len(text) < 50:
                # Check for common English patterns
                english_words = {'the', 'and', 'is', 'to', 'a', 'of', 'in', 'that', 'it', 'with'}
                words = set(text.lower().split())
                if len(words & english_words) >= 2:
                    detected_lang = 'en'
                else:
                    detected_lang = detect(text)
            else:
                detected_lang = detect(text)
                
            if detected_lang in self.allowed_languages:
                self.stats['passed'] += 1
                return True, text, ""
            else:
                self.stats['filtered'] += 1
                reason = f"language_{detected_lang}"
                self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
                return False, text, reason
                
        except LangDetectException:
            self.stats['filtered'] += 1
            reason = "detection_failed"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason


class JunkFilter(TextFilter):
    """Filter junk/malformed text."""
    
    def __init__(self, min_length: int = 50, max_length: int = 1000000, 
                 min_words: int = 5, max_non_printable_ratio: float = 0.1,
                 max_digit_ratio: float = 0.5, max_repetition_ratio: float = 0.3):
        super().__init__("junk")
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_non_printable_ratio = max_non_printable_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_repetition_ratio = max_repetition_ratio
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text:
            self.stats['filtered'] += 1
            reason = "empty"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        # Length checks
        if len(text) < self.min_length:
            self.stats['filtered'] += 1
            reason = "too_short"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
            
        if len(text) > self.max_length:
            self.stats['filtered'] += 1
            reason = "too_long"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        # Word count
        words = text.split()
        if len(words) < self.min_words:
            self.stats['filtered'] += 1
            reason = "too_few_words"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        # Non-printable character ratio
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\n\t\r')
        if non_printable / len(text) > self.max_non_printable_ratio:
            self.stats['filtered'] += 1
            reason = "too_many_non_printable"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        # Digit ratio check
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count / len(text) > self.max_digit_ratio:
            self.stats['filtered'] += 1
            reason = "too_many_digits"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        # Repetition check (simple)
        if self._has_excessive_repetition(text):
            self.stats['filtered'] += 1
            reason = "excessive_repetition"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        self.stats['passed'] += 1
        return True, text, ""
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character or word repetition."""
        # Check for repeated characters
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                # Found 3+ repeated chars, count total
                char = text[i]
                count = 0
                for c in text:
                    if c == char:
                        count += 1
                if count / len(text) > self.max_repetition_ratio:
                    return True
        
        # Check for repeated words
        words = text.lower().split()
        if len(words) < 10:
            return False
            
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_word_count = max(word_counts.values())
        if max_word_count / len(words) > self.max_repetition_ratio:
            return True
            
        return False


def create_default_pipeline(
    allowed_languages: Set[str] = None,
    min_length: int = 50,
    max_perplexity: float = 1000.0,
    pii_mode: str = "anonymize",
    use_toxicity_model: bool = False
) -> 'FilterPipeline':
    """Create a default filter pipeline."""
    if allowed_languages is None:
        allowed_languages = {'en'}
    
    filters = [
        WhitespaceNormalizer(),
        HTMLStripFilter(),
        JunkFilter(min_length=min_length),
        LanguageFilter(allowed_languages=allowed_languages),
        PerplexityFilter(max_perplexity=max_perplexity),
        PIIFilter(mode=pii_mode),
        ToxicityFilter(use_model=use_toxicity_model),
    ]
    
    return FilterPipeline(filters)


class HTMLStripFilter(TextFilter):
    """Strip HTML tags and entities."""
    
    def __init__(self):
        super().__init__("html_strip")
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text:
            self.stats['passed'] += 1
            return True, text, ""
        
        # Check if text contains HTML
        if '<' not in text or '>' not in text:
            self.stats['passed'] += 1
            return True, text, ""
        
        try:
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            clean_text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in clean_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            self.stats['passed'] += 1
            return True, clean_text, ""
            
        except Exception:
            # If parsing fails, return original text
            self.stats['passed'] += 1
            return True, text, ""


class PerplexityFilter(TextFilter):
    """Filter by text perplexity using a simple heuristic."""
    
    def __init__(self, max_perplexity: float = 1000.0):
        super().__init__("perplexity")
        self.max_perplexity = max_perplexity
        
        # Simple English character frequency model
        self.char_freqs = {
            'e': 0.1202, 't': 0.0910, 'a': 0.0812, 'o': 0.0768, 'i': 0.0731,
            'n': 0.0695, 's': 0.0628, 'h': 0.0592, 'r': 0.0592, 'd': 0.0432,
            'l': 0.0398, 'u': 0.0288, 'c': 0.0271, 'm': 0.0261, 'w': 0.0209,
            'f': 0.0230, 'g': 0.0203, 'y': 0.0181, 'p': 0.0182, 'b': 0.0149,
            'v': 0.0111, 'k': 0.0069, 'j': 0.0010, 'x': 0.0017, 'q': 0.0011,
            'z': 0.0007, ' ': 0.1918
        }
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text or len(text) < 50:
            self.stats['passed'] += 1
            return True, text, ""
        
        perplexity = self._calculate_perplexity(text.lower())
        
        if perplexity <= self.max_perplexity:
            self.stats['passed'] += 1
            return True, text, ""
        else:
            self.stats['filtered'] += 1
            reason = f"high_perplexity_{perplexity:.1f}"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate simple character-based perplexity."""
        if not text:
            return float('inf')
        
        log_prob_sum = 0.0
        char_count = 0
        
        for char in text:
            if char.isalpha() or char == ' ':
                prob = self.char_freqs.get(char, 0.0001)  # Small prob for unknown chars
                log_prob_sum += np.log(prob)
                char_count += 1
        
        if char_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / char_count
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity


class PIIFilter(TextFilter):
    """Filter or anonymize PII (Personally Identifiable Information)."""
    
    def __init__(self, mode: str = "filter", confidence_threshold: float = 0.8):
        super().__init__("pii")
        self.mode = mode  # "filter" or "anonymize"
        self.confidence_threshold = confidence_threshold
        
        # Regex patterns for common PII
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        }
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text:
            self.stats['passed'] += 1
            return True, text, ""
        
        # Check for PII patterns
        pii_found = []
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                pii_found.extend([(pii_type, match) for match in matches])
        
        if not pii_found:
            self.stats['passed'] += 1
            return True, text, ""
        
        if self.mode == "filter":
            # Filter out documents with PII
            self.stats['filtered'] += 1
            reason = f"contains_pii_{','.join(set(pii_type for pii_type, _ in pii_found))}"
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return False, text, reason
        
        elif self.mode == "anonymize":
            # Anonymize PII
            cleaned_text = text
            for pii_type, match in pii_found:
                if pii_type == 'email':
                    cleaned_text = cleaned_text.replace(match, '[EMAIL]')
                elif pii_type == 'phone':
                    cleaned_text = cleaned_text.replace(match, '[PHONE]')
                elif pii_type == 'ssn':
                    cleaned_text = cleaned_text.replace(match, '[SSN]')
                elif pii_type == 'credit_card':
                    cleaned_text = cleaned_text.replace(match, '[CREDIT_CARD]')
            
            self.stats['passed'] += 1
            return True, cleaned_text, ""
        
        self.stats['passed'] += 1
        return True, text, ""


class ToxicityFilter(TextFilter):
    """Filter toxic content."""
    
    def __init__(self, max_toxicity: float = 0.8, use_model: bool = False):
        super().__init__("toxicity")
        self.max_toxicity = max_toxicity
        self.use_model = use_model
        self.model = None
        
        if use_model:
            try:
                from detoxify import Detoxify
                self.model = Detoxify('original')
            except ImportError:
                print("Warning: detoxify not available, using simple heuristics")
                self.use_model = False
        
        # Simple toxic word list (partial)
        self.toxic_words = {
            'hate', 'kill', 'murder', 'die', 'stupid', 'idiot', 'moron',
            'retard', 'dumb', 'loser', 'failure', 'worthless', 'pathetic'
        }
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text:
            self.stats['passed'] += 1
            return True, text, ""
        
        if self.use_model and self.model:
            try:
                results = self.model.predict(text)
                toxicity_score = results.get('toxicity', 0.0)
                
                if toxicity_score > self.max_toxicity:
                    self.stats['filtered'] += 1
                    reason = f"toxic_model_{toxicity_score:.2f}"
                    self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
                    return False, text, reason
            except Exception as e:
                # Fall back to simple heuristics
                pass
        
        # Simple heuristic check
        text_lower = text.lower()
        toxic_count = sum(1 for word in self.toxic_words if word in text_lower)
        
        if toxic_count > 0:
            words = text_lower.split()
            toxic_ratio = toxic_count / len(words) if words else 0
            
            if toxic_ratio > 0.05:  # 5% toxic words threshold
                self.stats['filtered'] += 1
                reason = f"toxic_heuristic_{toxic_count}_words"
                self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
                return False, text, reason
        
        self.stats['passed'] += 1
        return True, text, ""


class WhitespaceNormalizer(TextFilter):
    """Normalize whitespace."""
    
    def __init__(self):
        super().__init__("whitespace_normalize")
    
    def filter(self, text: str, metadata: Dict = None) -> Tuple[bool, str, str]:
        if not text:
            self.stats['passed'] += 1
            return True, text, ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        self.stats['passed'] += 1
        return True, text, ""


class FilterPipeline:
    """Pipeline for applying multiple filters."""
    
    def __init__(self, filters: List[TextFilter]):
        self.filters = filters
    
    def apply(self, text: str, metadata: Dict = None) -> Tuple[bool, str, List[str]]:
        """
        Apply all filters in sequence.
        
        Returns:
            (should_keep, final_text, list_of_filter_reasons)
        """
        current_text = text
        reasons = []
        
        for filter_obj in self.filters:
            should_keep, current_text, reason = filter_obj.filter(current_text, metadata)
            
            if not should_keep:
                return False, current_text, [reason]
            
            if reason:  # Filter modified text but didn't reject
                reasons.append(f"{filter_obj.name}:{reason}")
        
        return True, current_text, reasons
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all filters."""
        return {f.name: f.get_stats() for f in self.filters}
