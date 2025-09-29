# Testing Plan

## Testing Strategy Overview

**Goal**: Ensure reliable fraud detection with minimal false positives and zero message loss.

**Testing Pyramid**:
- **Unit Tests (70%)**: Core logic - OCR, matching, data models
- **Integration Tests (20%)**: Database, external APIs, component interactions
- **End-to-End Tests (10%)**: Full pipeline with real images and mock Telegram

**Acceptance Criteria**:
- All unit tests pass with >95% coverage on core logic
- OCR accuracy >90% on clear text test images
- Brand matching precision >95% (few false positives)
- Zero message loss during processing pipeline
- Alert delivery within 30 seconds end-to-end

## Unit Tests

### Text Processing and Matching
```python
# tests/test_brand_matcher.py
import pytest
from src.brand_matcher import BrandMatcher

class TestBrandMatcher:

    def setup_method(self):
        self.matcher = BrandMatcher()
        # Override config for testing
        self.matcher.keywords = ['CloudWalk', 'InfinitePay', 'Visa', 'Mastercard']
        self.matcher.threshold = 85

    def test_exact_match_detection(self):
        """Should detect exact brand mentions"""
        text = "I love using CloudWalk for payments"
        matches = self.matcher.find_matches(text)

        assert len(matches) == 1
        assert matches[0]['brand'] == 'CloudWalk'
        assert matches[0]['confidence'] == 100
        assert matches[0]['match_type'] == 'exact'

    def test_fuzzy_match_with_typos(self):
        """Should detect brands with minor typos"""
        test_cases = [
            ("CloudWlk payment", "CloudWalk", 90),  # Missing letter
            ("Cloud Walk systems", "CloudWalk", 85),  # Space inserted
            ("Cloudwalk mobile", "CloudWalk", 95),  # Case variation
            ("InfinityPay app", "InfinitePay", 85),  # Similar word
        ]

        for text, expected_brand, min_confidence in test_cases:
            matches = self.matcher.find_matches(text)
            assert len(matches) >= 1
            brand_match = next(m for m in matches if m['brand'] == expected_brand)
            assert brand_match['confidence'] >= min_confidence

    def test_no_false_positives(self):
        """Should not match unrelated text"""
        false_positive_texts = [
            "I went for a walk in the clouds",  # Contains "walk" and "cloud"
            "The master card game was fun",     # Contains "master" and "card"
            "Infinite possibilities ahead",     # Contains "infinite"
            "Visual design is important",       # Contains "visa"-like text
        ]

        for text in false_positive_texts:
            matches = self.matcher.find_matches(text)
            # Should either find no matches or matches with very low confidence
            high_confidence_matches = [m for m in matches if m['confidence'] > 70]
            assert len(high_confidence_matches) == 0

    def test_text_normalization(self):
        """Should handle punctuation and special characters"""
        test_cases = [
            "***CloudWalk*** is amazing!",
            "CloudWalk... the best payment system?",
            "Check out CloudWalk (really good)",
            "CloudWalk: payment solution #1",
        ]

        for text in test_cases:
            matches = self.matcher.find_matches(text)
            assert len(matches) == 1
            assert matches[0]['brand'] == 'CloudWalk'

    def test_multiple_brands_in_text(self):
        """Should detect multiple brands in single text"""
        text = "CloudWalk and InfinitePay are both great for Visa transactions"
        matches = self.matcher.find_matches(text)

        found_brands = {m['brand'] for m in matches}
        assert 'CloudWalk' in found_brands
        assert 'InfinitePay' in found_brands
        assert 'Visa' in found_brands
        assert len(matches) == 3

    def test_case_insensitive_matching(self):
        """Should match regardless of case"""
        test_cases = [
            "CLOUDWALK",
            "cloudwalk",
            "CloudWalk",
            "cLoudWaLk"
        ]

        for text in test_cases:
            matches = self.matcher.find_matches(text)
            assert len(matches) == 1
            assert matches[0]['brand'] == 'CloudWalk'
```

### OCR Engine Testing
```python
# tests/test_ocr_engine.py
import pytest
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
from src.ocr_engine import OCREngine

class TestOCREngine:

    def setup_method(self):
        self.ocr = OCREngine()

    def create_test_image(self, text, font_size=40, image_size=(400, 100)):
        """Helper to create test images with text"""
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)

        # Use default font or try to load a better one
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2

        draw.text((x, y), text, fill='black', font=font)
        return img

    def test_clear_text_extraction(self):
        """Should extract clear text with high confidence"""
        test_text = "CloudWalk Payment System"
        img = self.create_test_image(test_text)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)

            extracted_text, confidence = self.ocr.extract_text(tmp.name)

            os.unlink(tmp.name)

        # Should extract the text with reasonable accuracy
        assert confidence > 70
        assert "CloudWalk" in extracted_text or "cloudwalk" in extracted_text.lower()
        assert "Payment" in extracted_text or "payment" in extracted_text.lower()

    def test_multiple_brands_in_image(self):
        """Should extract all text from image with multiple brands"""
        test_text = "CloudWalk & InfinitePay Solutions"
        img = self.create_test_image(test_text, font_size=36, image_size=(600, 120))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)

            extracted_text, confidence = self.ocr.extract_text(tmp.name)

            os.unlink(tmp.name)

        extracted_lower = extracted_text.lower()
        assert confidence > 60
        assert "cloudwalk" in extracted_lower
        assert "infinitepay" in extracted_lower or "infinite" in extracted_lower

    def test_noisy_image_preprocessing(self):
        """Should handle noisy images through preprocessing"""
        import numpy as np

        # Create image with noise
        test_text = "Visa Card"
        img = self.create_test_image(test_text)

        # Add noise
        img_array = np.array(img)
        noise = np.random.randint(0, 50, img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img_array)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            noisy_img.save(tmp.name)

            extracted_text, confidence = self.ocr.extract_text(tmp.name)

            os.unlink(tmp.name)

        # Preprocessing should help extract text even from noisy image
        extracted_lower = extracted_text.lower()
        assert "visa" in extracted_lower or confidence > 30

    def test_empty_image(self):
        """Should handle empty/blank images gracefully"""
        blank_img = Image.new('RGB', (400, 100), color='white')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            blank_img.save(tmp.name)

            extracted_text, confidence = self.ocr.extract_text(tmp.name)

            os.unlink(tmp.name)

        assert extracted_text.strip() == "" or len(extracted_text.strip()) < 5
        assert confidence < 50

    def test_invalid_image_handling(self):
        """Should handle corrupted/invalid images"""
        # Create invalid image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b"Not an image file")

            extracted_text, confidence = self.ocr.extract_text(tmp.name)

            os.unlink(tmp.name)

        assert extracted_text == ""
        assert confidence == 0
```

### Database Model Testing
```python
# tests/test_models.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base, TelegramMessage, ImageRecord, OCRText, BrandHit
from datetime import datetime

class TestDatabaseModels:

    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup in-memory SQLite for testing"""
        self.engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        yield

        self.session.close()

    def test_message_creation_and_constraints(self):
        """Should create message and enforce unique constraints"""
        # Create message
        message = TelegramMessage(
            chat_id=123456,
            message_id=789,
            sender_id=111,
            text="Test message with CloudWalk mention",
            has_media=True
        )

        self.session.add(message)
        self.session.commit()

        # Verify creation
        saved_message = self.session.query(TelegramMessage).first()
        assert saved_message.chat_id == 123456
        assert saved_message.message_id == 789
        assert saved_message.text == "Test message with CloudWalk mention"
        assert saved_message.has_media == True
        assert saved_message.processed == False  # Default

        # Test unique constraint
        duplicate_message = TelegramMessage(
            chat_id=123456,
            message_id=789,  # Same chat_id + message_id
            sender_id=222,
            text="Different text"
        )

        self.session.add(duplicate_message)

        with pytest.raises(Exception):  # Should raise integrity error
            self.session.commit()

    def test_image_deduplication(self):
        """Should prevent duplicate images by hash"""
        # Create message first
        message = TelegramMessage(chat_id=123, message_id=456)
        self.session.add(message)
        self.session.flush()

        # Create first image
        image1 = ImageRecord(
            message_id=message.id,
            file_path="/path/to/image1.jpg",
            sha256_hash="abc123def456",
            file_size=1024
        )

        self.session.add(image1)
        self.session.commit()

        # Try to add duplicate hash
        image2 = ImageRecord(
            message_id=message.id,
            file_path="/different/path/image2.jpg",  # Different path
            sha256_hash="abc123def456",  # Same hash
            file_size=2048
        )

        self.session.add(image2)

        with pytest.raises(Exception):  # Should raise unique constraint error
            self.session.commit()

    def test_cascade_deletion(self):
        """Should delete child records when parent is deleted"""
        # Create complete record chain
        message = TelegramMessage(chat_id=123, message_id=456)
        self.session.add(message)
        self.session.flush()

        image = ImageRecord(
            message_id=message.id,
            file_path="/test.jpg",
            sha256_hash="test123",
            file_size=1024
        )
        self.session.add(image)
        self.session.flush()

        ocr_text = OCRText(
            image_id=image.id,
            extracted_text="CloudWalk payment system",
            confidence=95
        )
        self.session.add(ocr_text)

        brand_hit = BrandHit(
            message_id=message.id,
            image_id=image.id,
            brand_name="CloudWalk",
            matched_text="CloudWalk",
            confidence_score=100
        )
        self.session.add(brand_hit)
        self.session.commit()

        # Verify all records exist
        assert self.session.query(TelegramMessage).count() == 1
        assert self.session.query(ImageRecord).count() == 1
        assert self.session.query(OCRText).count() == 1
        assert self.session.query(BrandHit).count() == 1

        # Delete parent message
        self.session.delete(message)
        self.session.commit()

        # All child records should be deleted
        assert self.session.query(TelegramMessage).count() == 0
        assert self.session.query(ImageRecord).count() == 0
        assert self.session.query(OCRText).count() == 0
        assert self.session.query(BrandHit).count() == 0
```

## Integration Tests

### Database Integration
```python
# tests/test_integration_database.py
import pytest
import asyncio
from src.database import SessionLocal, init_db
from src.telegram_collector import TelegramCollector
from src.processor import FraudProcessor

class TestDatabaseIntegration:

    @pytest.fixture(autouse=True)
    def setup_integration(self):
        """Setup test database"""
        init_db()
        yield
        # Cleanup handled by test database

    @pytest.mark.asyncio
    async def test_message_processing_pipeline(self):
        """Test complete message -> OCR -> matching pipeline"""
        db = SessionLocal()

        # Create test message with media
        from src.models import TelegramMessage, ImageRecord

        message = TelegramMessage(
            chat_id=123456,
            message_id=789,
            text="Check out this CloudWalk logo",
            has_media=True
        )
        db.add(message)
        db.flush()

        # Create test image record
        image = ImageRecord(
            message_id=message.id,
            file_path="tests/test_images/cloudwalk_logo.png",
            sha256_hash="test_hash_123",
            file_size=1024,
            processed=False
        )
        db.add(image)
        db.commit()

        # Process with fraud processor
        processor = FraudProcessor()
        await processor._process_single_image(image, db)

        # Verify processing results
        from src.models import OCRText, BrandHit

        ocr_results = db.query(OCRText).filter(OCRText.image_id == image.id).all()
        brand_hits = db.query(BrandHit).filter(BrandHit.image_id == image.id).all()

        assert len(ocr_results) == 1
        assert ocr_results[0].extracted_text is not None

        # Should detect CloudWalk in image or text
        cloudwalk_hits = [hit for hit in brand_hits if hit.brand_name == 'CloudWalk']
        assert len(cloudwalk_hits) >= 1

        db.close()
```

### API Integration Tests
```python
# tests/test_integration_telegram.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.telegram_collector import TelegramCollector
from src.alerting import AlertSender

class TestTelegramIntegration:

    @pytest.mark.asyncio
    async def test_telegram_message_handling(self):
        """Test Telegram message event processing"""
        collector = TelegramCollector()

        # Mock Telegram client
        with patch.object(collector, 'client') as mock_client:
            mock_message = AsyncMock()
            mock_message.id = 12345
            mock_message.text = "New CloudWalk feature released!"
            mock_message.media = None
            mock_message.download_media = AsyncMock(return_value=None)

            mock_event = AsyncMock()
            mock_event.message = mock_message
            mock_event.chat_id = 123456
            mock_event.sender_id = 789

            # Process mock message
            await collector._process_message(mock_event)

            # Verify message was stored
            from src.database import SessionLocal
            from src.models import TelegramMessage

            db = SessionLocal()
            stored_message = db.query(TelegramMessage).filter(
                TelegramMessage.message_id == 12345
            ).first()

            assert stored_message is not None
            assert stored_message.text == "New CloudWalk feature released!"
            assert stored_message.chat_id == 123456

            db.close()

    @pytest.mark.asyncio
    async def test_alert_sending(self):
        """Test alert delivery mechanism"""
        alert_sender = AlertSender()

        test_alert = {
            'brand': 'CloudWalk',
            'matched_text': 'CloudWalk',
            'confidence': 95,
            'chat_id': 123456,
            'message_id': 789
        }

        # Mock Telegram client for alert sending
        with patch.object(alert_sender, 'client') as mock_client:
            mock_client.start = AsyncMock()
            mock_client.send_message = AsyncMock()

            await alert_sender.send_brand_alert(test_alert)

            # Verify alert was sent
            mock_client.send_message.assert_called_once()
            call_args = mock_client.send_message.call_args

            assert 'CloudWalk' in call_args[0][1]  # Alert text contains brand
            assert call_args[1]['parse_mode'] == 'html'
```

## End-to-End Tests

### Docker Compose Test Environment
```python
# tests/test_e2e.py
import pytest
import docker
import time
import requests
import tempfile
import os
from PIL import Image, ImageDraw

class TestEndToEnd:

    @pytest.fixture(scope="class", autouse=True)
    def docker_environment(self):
        """Start Docker Compose environment for E2E testing"""
        client = docker.from_env()

        # Start test environment
        os.system("docker-compose -f docker-compose.test.yml up -d")

        # Wait for services to be healthy
        max_wait = 60  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Check if database is ready
                result = os.system("docker-compose -f docker-compose.test.yml exec -T db pg_isready -U fraud_user")
                if result == 0:
                    break
            except:
                pass
            time.sleep(2)

        # Additional wait for application startup
        time.sleep(10)

        yield

        # Cleanup
        os.system("docker-compose -f docker-compose.test.yml down -v")

    def create_test_image_with_brand(self, brand_name, output_path):
        """Create test image containing brand mention"""
        img = Image.new('RGB', (500, 200), color='white')
        draw = ImageDraw.Draw(img)

        # Add some realistic context
        text_lines = [
            f"New {brand_name} Partnership Announcement",
            "Leading payment technology company",
            "Secure and reliable transactions"
        ]

        y_pos = 30
        for line in text_lines:
            draw.text((20, y_pos), line, fill='black')
            y_pos += 40

        img.save(output_path)
        return output_path

    def test_complete_fraud_detection_pipeline(self):
        """Test entire pipeline: image upload -> OCR -> detection -> alert"""

        # Create test image with CloudWalk mention
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image_path = self.create_test_image_with_brand('CloudWalk', tmp.name)

        try:
            # Simulate uploading image to monitored location
            # In real test, this would be done through Telegram mock
            target_path = 'test_data/media/test_cloudwalk.png'
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            os.copy(test_image_path, target_path)

            # Insert test record into database to trigger processing
            import subprocess
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.test.yml', 'exec', '-T', 'db',
                'psql', '-U', 'fraud_user', '-d', 'fraud_db', '-c',
                f"""
                INSERT INTO telegram_messages (chat_id, message_id, text, has_media)
                VALUES (123456, 999, 'Test message', true);

                INSERT INTO images (message_id, file_path, sha256_hash, file_size, processed)
                SELECT id, '{target_path}', 'test_hash_e2e', 1024, false
                FROM telegram_messages WHERE message_id = 999;
                """
            ])

            # Wait for processing (up to 30 seconds)
            max_wait = 30
            start_time = time.time()
            brand_hit_found = False

            while time.time() - start_time < max_wait and not brand_hit_found:
                # Check for brand hit detection
                result = subprocess.run([
                    'docker-compose', '-f', 'docker-compose.test.yml', 'exec', '-T', 'db',
                    'psql', '-U', 'fraud_user', '-d', 'fraud_db', '-t', '-c',
                    "SELECT COUNT(*) FROM brand_hits WHERE brand_name = 'CloudWalk';"
                ], capture_output=True, text=True)

                if int(result.stdout.strip()) > 0:
                    brand_hit_found = True
                    break

                time.sleep(2)

            # Verify brand hit was detected
            assert brand_hit_found, "CloudWalk brand hit should be detected within 30 seconds"

            # Verify alert was triggered (check alert_sent flag)
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.test.yml', 'exec', '-T', 'db',
                'psql', '-U', 'fraud_user', '-d', 'fraud_db', '-t', '-c',
                "SELECT alert_sent FROM brand_hits WHERE brand_name = 'CloudWalk' LIMIT 1;"
            ], capture_output=True, text=True)

            alert_sent = result.stdout.strip().lower() == 't'
            assert alert_sent, "Alert should be marked as sent"

        finally:
            # Cleanup test files
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
            if os.path.exists(target_path):
                os.unlink(target_path)

    def test_system_health_endpoints(self):
        """Test that all system components are healthy"""

        # Test database connectivity
        result = os.system("docker-compose -f docker-compose.test.yml exec -T db pg_isready -U fraud_user")
        assert result == 0, "Database should be accessible"

        # Test application logs for successful startup
        logs = subprocess.run([
            'docker-compose', '-f', 'docker-compose.test.yml', 'logs', 'app'
        ], capture_output=True, text=True)

        assert "Connected to Telegram" in logs.stdout or logs.returncode == 0, \
               "Application should start successfully"

        # Test OCR functionality
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.test.yml', 'exec', '-T', 'app',
            'tesseract', '--version'
        ], capture_output=True, text=True)

        assert result.returncode == 0, "Tesseract should be installed and working"
```

## Performance Tests

### OCR Performance Benchmarks
```python
# tests/test_performance.py
import pytest
import time
import statistics
from src.ocr_engine import OCREngine
from PIL import Image, ImageDraw

class TestPerformance:

    def test_ocr_processing_speed(self):
        """OCR should process typical images within 3 seconds"""
        ocr_engine = OCREngine()

        # Create test images of various sizes
        test_images = []
        image_sizes = [(400, 200), (800, 400), (1200, 600)]

        for size in image_sizes:
            img = Image.new('RGB', size, color='white')
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), f"CloudWalk Payment System {size[0]}x{size[1]}", fill='black')

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                test_images.append(tmp.name)

        processing_times = []

        try:
            for image_path in test_images:
                start_time = time.time()
                text, confidence = ocr_engine.extract_text(image_path)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Verify it extracted something meaningful
                assert len(text.strip()) > 0
                assert confidence > 50

        finally:
            # Cleanup
            for path in test_images:
                if os.path.exists(path):
                    os.unlink(path)

        # Performance assertions
        max_time = max(processing_times)
        avg_time = statistics.mean(processing_times)

        assert max_time < 5.0, f"OCR should process images in <5s, got {max_time:.2f}s"
        assert avg_time < 3.0, f"Average OCR time should be <3s, got {avg_time:.2f}s"

    def test_brand_matching_performance(self):
        """Brand matching should be fast for typical text lengths"""
        from src.brand_matcher import BrandMatcher

        matcher = BrandMatcher()

        # Test with various text lengths
        test_texts = [
            "Short CloudWalk text",
            "Medium length text with CloudWalk mention in the middle of the sentence",
            "Very long text " * 50 + " with CloudWalk somewhere in this massive wall of text " + "more text " * 50,
        ]

        processing_times = []

        for text in test_texts:
            start_time = time.time()
            matches = matcher.find_matches(text)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Should find CloudWalk in all cases
            assert len(matches) >= 1
            assert any(m['brand'] == 'CloudWalk' for m in matches)

        max_time = max(processing_times)
        assert max_time < 0.1, f"Brand matching should be <0.1s, got {max_time:.3f}s"
```

## Test Data and Fixtures

### Golden Sample Images
```python
# tests/test_golden_samples.py
"""
Test against golden sample images to prevent regression in OCR accuracy
"""
import pytest
import os
from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher

class TestGoldenSamples:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup OCR engine and matcher"""
        self.ocr_engine = OCREngine()
        self.brand_matcher = BrandMatcher()

    @pytest.mark.parametrize("image_file,expected_brands,min_confidence", [
        ("cloudwalk_logo_clear.png", ["CloudWalk"], 90),
        ("infinitepay_banner.png", ["InfinitePay"], 85),
        ("visa_mastercard_logos.png", ["Visa", "Mastercard"], 80),
        ("mixed_brands_screenshot.png", ["CloudWalk", "Visa"], 75),
    ])
    def test_golden_sample_detection(self, image_file, expected_brands, min_confidence):
        """Test OCR and brand detection on curated golden samples"""
        image_path = f"tests/golden_samples/{image_file}"

        # Skip if golden sample doesn't exist (not committed to repo)
        if not os.path.exists(image_path):
            pytest.skip(f"Golden sample {image_file} not found")

        # Extract text
        extracted_text, ocr_confidence = self.ocr_engine.extract_text(image_path)

        # Find brand matches
        matches = self.brand_matcher.find_matches(extracted_text)
        found_brands = [m['brand'] for m in matches]

        # Assertions
        assert ocr_confidence >= min_confidence, f"OCR confidence {ocr_confidence} below {min_confidence}"

        for expected_brand in expected_brands:
            assert expected_brand in found_brands, f"Expected brand {expected_brand} not found in {found_brands}"
```

## Continuous Integration Setup

### GitHub Actions Test Pipeline
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: fraud_user
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: fraud_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

    - name: Install Python dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/test_*.py -v --cov=src --cov-report=xml
      env:
        DB_URL: postgresql+psycopg2://fraud_user:test_password@localhost:5432/fraud_test

    - name: Run integration tests
      run: |
        pytest tests/test_integration_*.py -v
      env:
        DB_URL: postgresql+psycopg2://fraud_user:test_password@localhost:5432/fraud_test

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Acceptance Testing Checklist

**Before Release - All Must Pass**:

- [ ] Unit tests: >95% pass rate, >90% code coverage on core logic
- [ ] Integration tests: Database, OCR, and API integration working
- [ ] End-to-end test: Complete pipeline processes test image in <30s
- [ ] Performance tests: OCR <3s average, brand matching <0.1s
- [ ] Golden samples: All curated test images processed correctly
- [ ] Security tests: No secrets in logs, proper error handling
- [ ] Docker build: Images build successfully, containers start healthy
- [ ] Database migration: Schema migrations run without errors
- [ ] Alert delivery: Test alerts sent successfully to configured chat

**Manual Verification**:
- [ ] Deploy to staging environment successfully
- [ ] Process real test image with brand mention
- [ ] Receive alert in test Telegram chat
- [ ] Verify logs show no errors or warnings
- [ ] Confirm database contains expected records