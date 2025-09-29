import re
from rapidfuzz import fuzz, process
from src.config import BRAND_KEYWORDS, FUZZY_THRESHOLD, MATCH_MIN_LENGTH
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class BrandMatcher:
    def __init__(self):
        self.keywords = [kw.strip() for kw in BRAND_KEYWORDS if kw.strip()]
        self.threshold = FUZZY_THRESHOLD
        self.min_length = MATCH_MIN_LENGTH

        # Preprocess keywords for faster matching
        self.normalized_keywords = {
            keyword: self._normalize_text(keyword)
            for keyword in self.keywords
        }

        logger.info(f"Initialized BrandMatcher with {len(self.keywords)} keywords, threshold: {self.threshold}")

    def find_matches(self, text: str) -> List[Dict]:
        """
        Find brand mentions in text using exact and fuzzy matching.
        Returns list of matches with confidence scores.
        """
        if not text or not self.keywords:
            return []

        matches = []
        normalized_text = self._normalize_text(text)

        # First pass: exact substring matching (highest priority)
        for keyword in self.keywords:
            normalized_keyword = self.normalized_keywords[keyword]
            if normalized_keyword.lower() in normalized_text.lower():
                matches.append({
                    'brand': keyword,
                    'matched_text': keyword,
                    'confidence': 100,
                    'match_type': 'exact',
                    'position': normalized_text.lower().find(normalized_keyword.lower())
                })
                continue  # Skip fuzzy matching for this keyword

        # Second pass: fuzzy matching on individual words
        exact_brands = {m['brand'] for m in matches}
        words = self._extract_words(normalized_text)

        for keyword in self.keywords:
            if keyword in exact_brands:
                continue  # Already found exact match

            normalized_keyword = self.normalized_keywords[keyword]
            best_match = self._find_best_fuzzy_match(normalized_keyword, words)
            if best_match and best_match['score'] >= self.threshold:
                matches.append({
                    'brand': keyword,
                    'matched_text': best_match['text'],
                    'confidence': best_match['score'],
                    'match_type': 'fuzzy',
                    'position': best_match.get('position', -1)
                })

        # Sort matches by confidence and position
        matches.sort(key=lambda x: (x['confidence'], -x.get('position', 0)), reverse=True)
        logger.debug(f"Found {len(matches)} brand matches in text")
        return matches

    def _find_best_fuzzy_match(self, keyword: str, words: List[str]) -> Dict:
        """Find best fuzzy match for keyword in word list"""
        if not words:
            return None

        best_score = 0
        best_match = None
        matched_word = None

        for word in words:
            # Try different fuzzy matching algorithms
            ratio_score = fuzz.ratio(keyword.lower(), word.lower())
            partial_ratio_score = fuzz.partial_ratio(keyword.lower(), word.lower())
            token_sort_ratio_score = fuzz.token_sort_ratio(keyword.lower(), word.lower())

            # Take the highest score
            score = max(ratio_score, partial_ratio_score, token_sort_ratio_score)

            if score > best_score:
                best_score = score
                best_match = word
                matched_word = word

        if best_match and best_score >= self.threshold:
            return {
                'text': best_match,
                'score': best_score,
                'position': words.index(matched_word) if matched_word in words else -1
            }

        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""

        # Remove excessive whitespace and special characters
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_words(self, text: str) -> List[str]:
        """Extract individual words from text"""
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) >= self.min_length]

    def add_keyword(self, keyword: str):
        """Dynamically add new brand keyword"""
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)
            self.normalized_keywords[keyword] = self._normalize_text(keyword)
            logger.info(f"Added new brand keyword: {keyword}")

    def remove_keyword(self, keyword: str):
        """Remove brand keyword"""
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            if keyword in self.normalized_keywords:
                del self.normalized_keywords[keyword]
            logger.info(f"Removed brand keyword: {keyword}")

    def get_keywords(self) -> List[str]:
        """Get list of current keywords"""
        return self.keywords.copy()

    def health_check(self) -> bool:
        """Verify brand matcher is working correctly"""
        try:
            # Test with sample text containing known brands
            test_text = "Welcome to CloudWalk Payment System powered by InfinitePay"
            matches = self.find_matches(test_text)

            # Should find at least one match if keywords are configured
            if not self.keywords:
                logger.warning("No brand keywords configured")
                return True  # Not an error if no keywords configured

            if not matches:
                logger.warning("Brand matcher health check: no matches found in test text")
                return False

            logger.info(f"Brand matcher health check: found {len(matches)} matches")
            return True

        except Exception as e:
            logger.error(f"Brand matcher health check failed: {e}")
            return False

    def get_match_statistics(self, text: str) -> Dict:
        """Get detailed matching statistics for debugging"""
        matches = self.find_matches(text)

        stats = {
            'total_matches': len(matches),
            'exact_matches': len([m for m in matches if m['match_type'] == 'exact']),
            'fuzzy_matches': len([m for m in matches if m['match_type'] == 'fuzzy']),
            'avg_confidence': 0,
            'text_length': len(text),
            'word_count': len(self._extract_words(text)),
            'matches': matches
        }

        if matches:
            stats['avg_confidence'] = sum(m['confidence'] for m in matches) / len(matches)

        return stats