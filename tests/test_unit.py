"""
Basic unit tests for the Fraud Monitoring System core components
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brand_matcher import BrandMatcher
from src.ocr_engine import OCREngine
from src.alert_sender import AlertSender
from src.config import get_config_summary


def test_brand_matcher_initialization():
    """Test BrandMatcher can be initialized"""
    matcher = BrandMatcher()
    assert matcher is not None
    assert len(matcher.get_keywords()) > 0


def test_brand_matcher_exact_match():
    """Test exact brand matching works"""
    matcher = BrandMatcher()

    text = "CloudWalk payment system"
    matches = matcher.find_matches(text)

    assert len(matches) > 0
    assert any(m['brand'] == 'CloudWalk' for m in matches)
    assert any(m['confidence'] == 100 for m in matches if m['brand'] == 'CloudWalk')


def test_brand_matcher_no_match():
    """Test text without brands returns no matches"""
    matcher = BrandMatcher()

    text = "Regular text without any brands"
    matches = matcher.find_matches(text)

    assert len(matches) == 0


def test_ocr_engine_initialization():
    """Test OCREngine can be initialized"""
    ocr = OCREngine()
    assert ocr is not None
    assert hasattr(ocr, 'lang')
    assert hasattr(ocr, 'confidence_threshold')


def test_ocr_engine_invalid_file():
    """Test OCR handles invalid files gracefully"""
    ocr = OCREngine()

    text, confidence, time_ms = ocr.extract_text("/nonexistent/file.png")

    assert text == ""
    assert confidence == 0
    assert time_ms >= 0


def test_alert_sender_initialization():
    """Test AlertSender can be initialized"""
    sender = AlertSender()
    assert sender is not None
    assert hasattr(sender, 'alert_chat_id')


def test_alert_formatting():
    """Test alert message formatting"""
    sender = AlertSender()

    test_data = {
        'brand': 'TestBrand',
        'matched_text': 'TestBrand Payment',
        'confidence': 95,
        'match_type': 'exact',
        'chat_id': '-123456789',
        'message_id': '42',
        'extracted_text': 'This is a test message',
        'image_path': '/path/to/image.png'
    }

    formatted_alert = sender._format_alert(test_data)

    assert isinstance(formatted_alert, str)
    assert len(formatted_alert) > 0
    assert '🚨' in formatted_alert
    assert 'TestBrand' in formatted_alert


def test_configuration_loading():
    """Test configuration can be loaded"""
    config = get_config_summary()

    assert isinstance(config, dict)
    assert 'brand_keywords_count' in config
    assert config['brand_keywords_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])