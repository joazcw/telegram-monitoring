"""
Basic integration tests for the Fraud Monitoring System
Tests that core components work together correctly.
"""
import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher
from src.alert_sender import AlertSender
from src.processor import FraudProcessor
from src.telegram_collector import TelegramCollector
from src.config import get_config_summary


def test_system_configuration():
    """Test that system configuration loads properly"""
    config = get_config_summary()

    assert isinstance(config, dict)
    assert 'brand_keywords_count' in config
    assert config['brand_keywords_count'] > 0


def test_component_initialization():
    """Test that all core components can be initialized"""
    # Test each component can be created without errors
    ocr_engine = OCREngine()
    assert ocr_engine is not None

    brand_matcher = BrandMatcher()
    assert brand_matcher is not None
    assert len(brand_matcher.get_keywords()) > 0

    alert_sender = AlertSender()
    assert alert_sender is not None

    telegram_collector = TelegramCollector()
    assert telegram_collector is not None

    processor = FraudProcessor()
    assert processor is not None


def test_brand_detection_pipeline():
    """Test the complete brand detection pipeline"""
    brand_matcher = BrandMatcher()

    # Test various brand scenarios
    test_cases = [
        ("Welcome to CloudWalk Payment System", True, "CloudWalk"),
        ("InfinitePay mobile app is great", True, "InfinitePay"),
        ("Visa and Mastercard accepted", True, "Visa"),
        ("Regular text without brands", False, None),
    ]

    for text, should_match, expected_brand in test_cases:
        matches = brand_matcher.find_matches(text)

        if should_match:
            assert len(matches) > 0, f"Should find brand in: '{text}'"
            brand_names = [m['brand'] for m in matches]
            assert expected_brand in brand_names
        else:
            assert len(matches) == 0, f"Should not find brands in: '{text}'"


def test_alert_system():
    """Test alert formatting and basic functionality"""
    alert_sender = AlertSender()

    test_alert_data = {
        'brand': 'CloudWalk',
        'matched_text': 'CloudWalk Payment',
        'confidence': 95,
        'match_type': 'exact',
        'chat_id': '-123456789',
        'message_id': '42',
        'extracted_text': 'Welcome to CloudWalk Payment System',
        'image_path': '/test/image.png'
    }

    formatted_alert = alert_sender._format_alert(test_alert_data)

    # Validate alert content
    assert '🚨' in formatted_alert
    assert 'CloudWalk' in formatted_alert
    assert '95%' in formatted_alert
    assert '<b>' in formatted_alert  # HTML formatting


@pytest.mark.asyncio
async def test_processor_health_check():
    """Test processor health check works"""
    processor = FraudProcessor()

    # Basic health check
    health_result = await processor._health_check()
    assert isinstance(health_result, bool)

    # Get basic stats
    stats = processor.get_stats()
    assert isinstance(stats, dict)
    assert 'processed_images' in stats
    assert 'brand_hits' in stats


def test_telegram_collector_stats():
    """Test Telegram collector provides basic stats"""
    collector = TelegramCollector()

    stats = collector.get_stats()
    assert isinstance(stats, dict)
    assert 'configured_groups' in stats
    assert 'authenticated' in stats


@pytest.mark.asyncio
async def test_complete_pipeline_simulation():
    """Test complete processing pipeline with simulated data"""
    # Initialize components
    brand_matcher = BrandMatcher()
    alert_sender = AlertSender()

    # Simulate OCR extraction result
    extracted_text = "Welcome to CloudWalk Payment System - Secure transactions!"

    # Test brand matching
    matches = brand_matcher.find_matches(extracted_text)
    assert len(matches) > 0, "Should find brand matches"

    best_match = matches[0]
    assert best_match['brand'] == 'CloudWalk'
    assert best_match['confidence'] >= 85

    # Test alert generation
    alert_data = {
        'brand': best_match['brand'],
        'matched_text': best_match['matched_text'],
        'confidence': best_match['confidence'],
        'match_type': best_match['match_type'],
        'chat_id': '-123456789',
        'message_id': '12345',
        'extracted_text': extracted_text,
        'image_path': '/test/image.png'
    }

    formatted_alert = alert_sender._format_alert(alert_data)
    assert 'CloudWalk' in formatted_alert
    assert '🚨' in formatted_alert


if __name__ == "__main__":
    pytest.main([__file__, "-v"])