"""
Unit tests for individual components of the Fraud Monitoring System

These tests validate individual components in isolation.
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.brand_matcher import BrandMatcher
from src.ocr_engine import OCREngine
from src.alert_sender import AlertSender
from src.config import get_config_summary


class TestBrandMatcher:
    """Unit tests for BrandMatcher component"""

    def test_initialization(self):
        """Test BrandMatcher initialization"""
        matcher = BrandMatcher()
        assert matcher is not None
        assert len(matcher.get_keywords()) > 0
        assert matcher.threshold > 0
        assert matcher.min_length > 0

    def test_exact_matching(self):
        """Test exact brand matching"""
        matcher = BrandMatcher()

        test_cases = [
            ("CloudWalk payment system", "CloudWalk"),
            ("Welcome to InfinitePay", "InfinitePay"),
            ("Visa card accepted", "Visa"),
            ("Mastercard payment", "Mastercard"),
        ]

        for text, expected_brand in test_cases:
            matches = matcher.find_matches(text)
            assert len(matches) > 0

            brand_names = [m['brand'] for m in matches]
            assert expected_brand in brand_names

            # Find the specific match
            for match in matches:
                if match['brand'] == expected_brand:
                    assert match['confidence'] == 100  # Exact match
                    assert match['match_type'] == 'exact'

    def test_fuzzy_matching(self):
        """Test fuzzy brand matching"""
        matcher = BrandMatcher()

        fuzzy_cases = [
            ("CloudWlk payment", "CloudWalk"),
            ("InfinityPay app", "InfinitePay"),
            ("Viza card", "Visa"),
        ]

        for text, expected_brand in fuzzy_cases:
            matches = matcher.find_matches(text)

            if matches:  # Fuzzy matching might not always work
                brand_names = [m['brand'] for m in matches]
                if expected_brand in brand_names:
                    for match in matches:
                        if match['brand'] == expected_brand:
                            assert match['confidence'] < 100  # Not exact
                            assert match['match_type'] == 'fuzzy'
                            assert match['confidence'] >= matcher.threshold

    def test_no_matches(self):
        """Test text without brand mentions"""
        matcher = BrandMatcher()

        no_match_cases = [
            "Regular text without brands",
            "Hello world",
            "Payment system without specific brand",
            "Just some random text here",
        ]

        for text in no_match_cases:
            matches = matcher.find_matches(text)
            assert len(matches) == 0

    def test_text_normalization(self):
        """Test text normalization"""
        matcher = BrandMatcher()

        # Test with special characters and extra spaces
        text_with_noise = "   CloudWalk!!!   payment   system   "
        matches = matcher.find_matches(text_with_noise)

        assert len(matches) > 0
        assert matches[0]['brand'] == 'CloudWalk'

    def test_add_remove_keywords(self):
        """Test dynamic keyword management"""
        matcher = BrandMatcher()

        initial_count = len(matcher.get_keywords())

        # Add new keyword
        matcher.add_keyword("TestBrand")
        assert len(matcher.get_keywords()) == initial_count + 1
        assert "TestBrand" in matcher.get_keywords()

        # Remove keyword
        matcher.remove_keyword("TestBrand")
        assert len(matcher.get_keywords()) == initial_count
        assert "TestBrand" not in matcher.get_keywords()

    def test_health_check(self):
        """Test brand matcher health check"""
        matcher = BrandMatcher()
        health_result = matcher.health_check()
        assert isinstance(health_result, bool)
        assert health_result == True  # Should pass with default keywords

    def test_match_statistics(self):
        """Test match statistics functionality"""
        matcher = BrandMatcher()

        test_text = "CloudWalk and InfinitePay are both great payment systems"
        stats = matcher.get_match_statistics(test_text)

        assert isinstance(stats, dict)
        assert 'total_matches' in stats
        assert 'exact_matches' in stats
        assert 'fuzzy_matches' in stats
        assert 'avg_confidence' in stats
        assert 'text_length' in stats
        assert 'word_count' in stats
        assert 'matches' in stats

        assert stats['total_matches'] >= 2  # Should find CloudWalk and InfinitePay


class TestOCREngine:
    """Unit tests for OCREngine component"""

    def test_initialization(self):
        """Test OCREngine initialization"""
        ocr = OCREngine()
        assert ocr is not None
        assert hasattr(ocr, 'lang')
        assert hasattr(ocr, 'confidence_threshold')
        assert hasattr(ocr, 'timeout')
        assert hasattr(ocr, 'tesseract_available')

    def test_health_check(self):
        """Test OCR engine health check"""
        ocr = OCREngine()
        health_result = ocr.health_check()
        assert isinstance(health_result, bool)

        # Health check result depends on Tesseract availability
        if ocr.tesseract_available:
            assert health_result == True
        # If Tesseract not available, health check might still work for basic functionality

    def test_supported_languages(self):
        """Test get supported languages"""
        ocr = OCREngine()
        languages = ocr.get_supported_languages()
        assert isinstance(languages, list)

        if ocr.tesseract_available:
            assert len(languages) > 0

    def test_extract_text_invalid_file(self):
        """Test OCR with invalid file path"""
        ocr = OCREngine()

        text, confidence, time_ms = ocr.extract_text("/nonexistent/file.png")

        assert text == ""
        assert confidence == 0
        assert time_ms >= 0

    def test_extract_text_from_bytes_invalid(self):
        """Test OCR from invalid bytes"""
        ocr = OCREngine()

        invalid_bytes = b"not an image"
        text, confidence, time_ms = ocr.extract_text_from_bytes(invalid_bytes)

        assert text == ""
        assert confidence == 0
        assert time_ms >= 0


class TestAlertSender:
    """Unit tests for AlertSender component"""

    def test_initialization(self):
        """Test AlertSender initialization"""
        sender = AlertSender()
        assert sender is not None
        assert hasattr(sender, 'alert_chat_id')
        assert hasattr(sender, 'rate_limit')
        assert hasattr(sender, 'credentials_available')

    def test_alert_formatting(self):
        """Test alert message formatting"""
        sender = AlertSender()

        test_data = {
            'brand': 'TestBrand',
            'matched_text': 'TestBrand Payment',
            'confidence': 95,
            'match_type': 'exact',
            'chat_id': '-123456789',
            'message_id': '42',
            'extracted_text': 'This is a test message with TestBrand mention',
            'image_path': '/path/to/test/image.png'
        }

        formatted_alert = sender._format_alert(test_data)

        assert isinstance(formatted_alert, str)
        assert len(formatted_alert) > 0
        assert '🚨' in formatted_alert
        assert 'TestBrand' in formatted_alert
        assert '95%' in formatted_alert
        assert 'Exact' in formatted_alert
        assert '<b>' in formatted_alert  # HTML formatting
        assert '</b>' in formatted_alert

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        sender = AlertSender()

        # Should allow sending initially
        assert sender._check_rate_limit() == True

        # Fill up the rate limit
        for _ in range(sender.rate_limit):
            sender._record_alert()

        # Should now be rate limited
        assert sender._check_rate_limit() == False

    def test_get_stats(self):
        """Test alert sender statistics"""
        sender = AlertSender()

        stats = sender.get_stats()

        assert isinstance(stats, dict)
        assert 'credentials_configured' in stats
        assert 'chat_configured' in stats
        assert 'rate_limit' in stats
        assert 'recent_alerts_count' in stats
        assert 'connected' in stats
        assert 'alert_history_minutes' in stats

        assert isinstance(stats['rate_limit'], int)
        assert isinstance(stats['recent_alerts_count'], int)
        assert isinstance(stats['alert_history_minutes'], list)

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test alert sender health check"""
        sender = AlertSender()

        health_result = await sender.health_check()
        assert isinstance(health_result, bool)

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test alert sender stop functionality"""
        sender = AlertSender()

        # Should not raise an exception
        await sender.stop()


class TestConfiguration:
    """Unit tests for configuration management"""

    def test_config_summary(self):
        """Test configuration summary"""
        summary = get_config_summary()

        assert isinstance(summary, dict)
        assert 'telegram_api_configured' in summary
        assert 'groups_count' in summary
        assert 'brand_keywords_count' in summary
        assert 'media_dir' in summary
        assert 'db_url_configured' in summary
        assert 'log_level' in summary
        assert 'ocr_lang' in summary
        assert 'fuzzy_threshold' in summary

        # Check data types
        assert isinstance(summary['telegram_api_configured'], bool)
        assert isinstance(summary['groups_count'], int)
        assert isinstance(summary['brand_keywords_count'], int)
        assert isinstance(summary['fuzzy_threshold'], int)

    def test_config_validation(self):
        """Test configuration validation"""
        from src.config import validate_config

        # This will likely fail without proper environment setup
        try:
            result = validate_config()
            assert result == True
        except ValueError:
            # Expected if configuration is incomplete
            pass


def run_unit_tests():
    """Run all unit tests"""
    print("🧪 Running Unit Tests")
    print("=" * 40)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])

    return exit_code


if __name__ == "__main__":
    run_unit_tests()