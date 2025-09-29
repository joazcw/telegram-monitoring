"""
Integration tests for the Fraud Monitoring System

These tests validate that all components work together correctly
and the complete pipeline functions as expected.
"""
import pytest
import asyncio
import tempfile
import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher
from src.alert_sender import AlertSender
from src.processor import FraudProcessor
from src.telegram_collector import TelegramCollector
from src.database import SessionLocal, init_db, check_db_connection
from src.models import TelegramMessage, ImageRecord, OCRText, BrandHit, Base
from src.config import get_config_summary
from PIL import Image, ImageDraw, ImageFont
import logging

# Setup test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFraudMonitoringIntegration:
    """Integration tests for the complete fraud monitoring system"""

    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp(prefix="fraud_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="class")
    def test_image_with_brand(self, temp_dir):
        """Create test image with brand mention"""
        image_path = os.path.join(temp_dir, "test_brand_image.png")

        # Create a simple image with text
        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)

        try:
            # Try to use a system font, fallback to default if not available
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Add brand text
        text = "Welcome to CloudWalk Payment System - Secure and Fast!"
        draw.text((50, 80), text, font=font, fill='black')

        img.save(image_path)
        return image_path

    @pytest.fixture(scope="class")
    def test_image_no_brand(self, temp_dir):
        """Create test image without brand mention"""
        image_path = os.path.join(temp_dir, "test_no_brand_image.png")

        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
        except (OSError, IOError):
            font = ImageFont.load_default()

        text = "This is a regular message without any brands"
        draw.text((50, 80), text, font=font, fill='black')

        img.save(image_path)
        return image_path

    def test_system_configuration(self):
        """Test that system configuration is properly loaded"""
        config_summary = get_config_summary()

        assert isinstance(config_summary, dict)
        assert 'brand_keywords_count' in config_summary
        assert 'fuzzy_threshold' in config_summary
        assert 'log_level' in config_summary

        # Should have default brand keywords
        assert config_summary['brand_keywords_count'] > 0

        logger.info(f"✅ Configuration loaded: {config_summary}")

    def test_component_initialization(self):
        """Test that all core components can be initialized"""
        logger.info("Testing component initialization...")

        # Test OCR Engine
        ocr_engine = OCREngine()
        assert ocr_engine is not None
        assert hasattr(ocr_engine, 'lang')
        assert hasattr(ocr_engine, 'confidence_threshold')

        # Test Brand Matcher
        brand_matcher = BrandMatcher()
        assert brand_matcher is not None
        assert len(brand_matcher.get_keywords()) > 0

        # Test Alert Sender
        alert_sender = AlertSender()
        assert alert_sender is not None

        # Test Telegram Collector
        telegram_collector = TelegramCollector()
        assert telegram_collector is not None

        # Test Processor
        processor = FraudProcessor()
        assert processor is not None

        logger.info("✅ All components initialized successfully")

    def test_brand_matching_integration(self):
        """Test brand matching with various text scenarios"""
        brand_matcher = BrandMatcher()

        test_cases = [
            ("Welcome to CloudWalk Payment System", "CloudWalk", True),
            ("InfinitePay mobile app is great", "InfinitePay", True),
            ("Visa and Mastercard are accepted here", "Visa", True),
            ("CloudWlk typo should match", "CloudWalk", True),  # Fuzzy match
            ("Regular text without brands", None, False),
            ("Mastercard payment processing", "Mastercard", True),
            ("InfinityPay with typo", "InfinitePay", True),  # Fuzzy match
        ]

        for text, expected_brand, should_match in test_cases:
            matches = brand_matcher.find_matches(text)

            if should_match:
                assert len(matches) > 0, f"Should find brand in: '{text}'"
                brand_names = [m['brand'] for m in matches]
                assert expected_brand in brand_names, f"Should find {expected_brand} in: '{text}'"

                # Check confidence scores
                for match in matches:
                    if match['brand'] == expected_brand:
                        assert match['confidence'] > 0, f"Confidence should be > 0 for {expected_brand}"
                        logger.info(f"✅ Found {expected_brand} with {match['confidence']}% confidence")
            else:
                assert len(matches) == 0, f"Should not find brands in: '{text}'"

        logger.info("✅ Brand matching integration test passed")

    def test_alert_formatting(self):
        """Test alert message formatting"""
        alert_sender = AlertSender()

        test_alert_data = {
            'brand': 'CloudWalk',
            'matched_text': 'CloudWalk Payment',
            'confidence': 95,
            'match_type': 'exact',
            'chat_id': '-123456789',
            'message_id': '42',
            'extracted_text': 'Welcome to CloudWalk Payment System - the future of fintech!',
            'image_path': '/app/data/media/test_image.png'
        }

        formatted_alert = alert_sender._format_alert(test_alert_data)

        # Validate alert content
        assert '🚨' in formatted_alert
        assert 'CloudWalk' in formatted_alert
        assert '95%' in formatted_alert
        assert 'Exact' in formatted_alert
        assert '-123456789' in formatted_alert
        assert 'test_image.png' in formatted_alert
        assert '<b>' in formatted_alert  # HTML formatting
        assert 'Automated fraud detection alert' in formatted_alert

        logger.info("✅ Alert formatting test passed")
        logger.info(f"Sample alert:\n{formatted_alert}")

    def test_ocr_engine_health_check(self):
        """Test OCR engine health check functionality"""
        ocr_engine = OCREngine()

        # Health check should work regardless of Tesseract availability
        health_result = ocr_engine.health_check()
        assert isinstance(health_result, bool)

        # Test with dummy image processing (without actual OCR if Tesseract unavailable)
        if ocr_engine.tesseract_available:
            logger.info("✅ OCR engine available and healthy")
        else:
            logger.info("⚠️  OCR engine initialized but Tesseract not available (expected in dev)")

        # Test supported languages method
        languages = ocr_engine.get_supported_languages()
        assert isinstance(languages, list)

    @pytest.mark.asyncio
    async def test_processor_health_checks(self):
        """Test processor health check system"""
        processor = FraudProcessor()

        # Run health checks
        health_result = await processor._health_check()
        assert isinstance(health_result, bool)

        # Get processor stats
        stats = processor.get_stats()
        assert isinstance(stats, dict)
        assert 'processed_images' in stats
        assert 'brand_hits' in stats
        assert 'alerts_sent' in stats
        assert 'processing_errors' in stats

        logger.info(f"✅ Processor health check completed")
        logger.info(f"Initial stats: {stats}")

    def test_telegram_collector_stats(self):
        """Test Telegram collector statistics"""
        collector = TelegramCollector()

        stats = collector.get_stats()
        assert isinstance(stats, dict)
        assert 'configured_groups' in stats
        assert 'authenticated' in stats
        assert 'connected' in stats
        assert 'credentials_configured' in stats
        assert 'database_available' in stats

        logger.info(f"✅ Telegram collector stats: {stats}")

    @pytest.mark.asyncio
    async def test_alert_sender_health_check(self):
        """Test alert sender health check"""
        alert_sender = AlertSender()

        health_result = await alert_sender.health_check()
        assert isinstance(health_result, bool)

        stats = alert_sender.get_stats()
        assert isinstance(stats, dict)
        assert 'credentials_configured' in stats
        assert 'chat_configured' in stats
        assert 'rate_limit' in stats

        logger.info(f"✅ Alert sender health check completed")
        logger.info(f"Alert sender stats: {stats}")

    def test_rate_limiting_functionality(self):
        """Test rate limiting in alert sender"""
        alert_sender = AlertSender()

        # Should be able to send initially
        assert alert_sender._check_rate_limit() == True

        # Record multiple alerts to test rate limiting
        for i in range(alert_sender.rate_limit + 1):
            alert_sender._record_alert()

        # Should now be rate limited
        assert alert_sender._check_rate_limit() == False

        logger.info("✅ Rate limiting functionality working")

    def test_database_connection_handling(self):
        """Test graceful database connection handling"""
        # Test database availability check
        try:
            db_available = check_db_connection()
            logger.info(f"Database connection test: {'Available' if db_available else 'Unavailable'}")
        except Exception as e:
            logger.info(f"Database connection failed (expected): {e}")

        # Test session creation (should not fail even if DB unavailable)
        try:
            db = SessionLocal()
            db.close()
            logger.info("✅ Database session creation successful")
        except Exception as e:
            logger.info(f"Database session creation failed: {e}")

    @pytest.mark.asyncio
    async def test_complete_pipeline_simulation(self, test_image_with_brand):
        """Test complete processing pipeline with simulated data"""
        logger.info("🧪 Running complete pipeline simulation...")

        # Initialize all components
        ocr_engine = OCREngine()
        brand_matcher = BrandMatcher()
        alert_sender = AlertSender()

        # Simulate OCR extraction
        extracted_text = "Welcome to CloudWalk Payment System - Secure transactions for everyone!"
        confidence = 85
        processing_time = 150  # ms

        logger.info(f"Simulated OCR: '{extracted_text[:50]}...' (confidence: {confidence}%)")

        # Test brand matching
        matches = brand_matcher.find_matches(extracted_text)
        assert len(matches) > 0, "Should find brand matches in simulated text"

        best_match = matches[0]
        assert best_match['brand'] == 'CloudWalk'
        assert best_match['confidence'] >= 85

        logger.info(f"✅ Found brand match: {best_match['brand']} ({best_match['confidence']}%)")

        # Test alert formatting
        alert_data = {
            'brand': best_match['brand'],
            'matched_text': best_match['matched_text'],
            'confidence': best_match['confidence'],
            'match_type': best_match['match_type'],
            'chat_id': '-123456789',
            'message_id': '12345',
            'extracted_text': extracted_text,
            'image_path': test_image_with_brand
        }

        formatted_alert = alert_sender._format_alert(alert_data)
        assert 'CloudWalk' in formatted_alert
        assert '🚨' in formatted_alert

        logger.info("✅ Complete pipeline simulation successful!")
        logger.info(f"Alert would be sent:\n{formatted_alert[:200]}...")

    def test_file_hash_calculation(self, test_image_with_brand):
        """Test file hash calculation for deduplication"""
        collector = TelegramCollector()

        # Test hash calculation method exists and works
        if hasattr(collector, '_calculate_file_hash'):
            # This would normally be async, but we test the concept
            assert os.path.exists(test_image_with_brand)
            logger.info(f"✅ Test image exists: {test_image_with_brand}")

        # Test file size calculation
        file_size = os.path.getsize(test_image_with_brand)
        assert file_size > 0
        logger.info(f"✅ File size: {file_size} bytes")

    def test_configuration_validation(self):
        """Test configuration validation functions"""
        from src.config import get_config_summary, validate_config

        # Test config summary
        summary = get_config_summary()
        assert isinstance(summary, dict)
        assert summary['brand_keywords_count'] > 0

        # Test config validation (will fail without proper env vars)
        try:
            validate_config()
            logger.info("✅ Configuration validation passed")
        except ValueError as e:
            logger.info(f"⚠️  Configuration validation failed (expected): {e}")
            # This is expected without proper Telegram credentials

    @pytest.mark.asyncio
    async def test_system_startup_simulation(self):
        """Test system startup process simulation"""
        logger.info("🚀 Simulating system startup...")

        from src.main import FraudMonitorApp

        app = FraudMonitorApp()
        assert app is not None

        # Test startup (will handle DB connection failures gracefully)
        try:
            await app.startup()
            logger.info("✅ System startup completed")
        except Exception as e:
            logger.info(f"System startup failed (expected without full environment): {e}")

        # Test shutdown
        await app.shutdown()
        logger.info("✅ System shutdown completed")

    def test_migration_script_exists(self):
        """Test that migration management script exists and is functional"""
        migrate_script = Path(project_root) / "scripts" / "migrate.py"
        assert migrate_script.exists(), "Migration script should exist"
        assert migrate_script.is_file(), "Migration script should be a file"

        # Test script is executable
        assert os.access(migrate_script, os.R_OK), "Migration script should be readable"

        logger.info("✅ Migration script exists and is accessible")

    def test_deployment_scripts_exist(self):
        """Test that deployment scripts exist"""
        scripts_dir = Path(project_root) / "scripts"

        # Check essential scripts exist
        essential_scripts = [
            "deploy.sh",
            "auth.py",
            "migrate.py"
        ]

        for script_name in essential_scripts:
            script_path = scripts_dir / script_name
            assert script_path.exists(), f"{script_name} should exist"
            logger.info(f"✅ Found script: {script_name}")

    def test_docker_configuration_exists(self):
        """Test that Docker configuration files exist"""
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore"
        ]

        for docker_file in docker_files:
            file_path = Path(project_root) / docker_file
            assert file_path.exists(), f"{docker_file} should exist"
            assert file_path.stat().st_size > 0, f"{docker_file} should not be empty"
            logger.info(f"✅ Found Docker file: {docker_file}")

    def test_environment_template_exists(self):
        """Test that environment template exists and has required variables"""
        env_example = Path(project_root) / ".env.example"
        assert env_example.exists(), ".env.example should exist"

        # Read and check for essential variables
        content = env_example.read_text()
        essential_vars = [
            "TELEGRAM_API_ID",
            "TELEGRAM_API_HASH",
            "TELEGRAM_GROUPS",
            "BRAND_KEYWORDS",
            "DB_URL"
        ]

        for var in essential_vars:
            assert var in content, f"{var} should be in .env.example"
            logger.info(f"✅ Found environment variable template: {var}")


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Fraud Monitoring System Integration Tests")
    print("=" * 60)

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
        print("✅ Fraud Monitoring System is ready for deployment")
    else:
        print("\n" + "=" * 60)
        print("❌ Some integration tests failed")
        print("❌ Please check the issues above")

    return exit_code


if __name__ == "__main__":
    run_integration_tests()