# Implementation Guide

Step-by-step implementation from empty repository to running fraud monitoring system.

## Prerequisites Setup

### Step 1: Project Structure
```bash
mkdir fraud-monitor && cd fraud-monitor
mkdir -p src tests docs data/media data/logs
touch README.md requirements.txt Dockerfile docker-compose.yml .env.example
```

### Step 2: Python Dependencies
Create `requirements.txt`:
```
telethon==1.32.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1
opencv-python-headless==4.8.1.78
pytesseract==0.3.10
rapidfuzz==3.5.2
python-dotenv==1.0.0
asyncio-mqtt==0.13.0
```

### Step 3: Database Schema
Create `alembic.ini` and migration:
```bash
pip install -r requirements.txt
alembic init migrations
# Edit alembic.ini: sqlalchemy.url = postgresql+psycopg2://user:pass@localhost/db
alembic revision --autogenerate -m "initial schema"
```

## Core Implementation

### Step 4: Configuration Management
`src/config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_ID = int(os.getenv('TELEGRAM_API_ID'))
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH')
TELEGRAM_SESSION_PATH = os.getenv('TELEGRAM_SESSION_PATH', './data/session')
TELEGRAM_GROUPS = os.getenv('TELEGRAM_GROUPS', '').split(',')
TELEGRAM_ALERT_CHAT_ID = int(os.getenv('TELEGRAM_ALERT_CHAT_ID'))

DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://fraud_user:fraud_pass@localhost:5432/fraud_db')
MEDIA_DIR = os.getenv('MEDIA_DIR', './data/media')
OCR_LANG = os.getenv('OCR_LANG', 'eng')
BRAND_KEYWORDS = os.getenv('BRAND_KEYWORDS', 'CloudWalk,InfinitePay').split(',')
FUZZY_THRESHOLD = int(os.getenv('FUZZY_THRESHOLD', '85'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

### Step 5: Database Models
`src/models.py`:
```python
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class TelegramMessage(Base):
    __tablename__ = 'telegram_messages'

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, nullable=False)
    message_id = Column(Integer, nullable=False)
    sender_id = Column(Integer)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    has_media = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)

    __table_args__ = (
        Index('ix_chat_message', 'chat_id', 'message_id', unique=True),
        Index('ix_timestamp', 'timestamp'),
        Index('ix_processed', 'processed'),
    )

class ImageRecord(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    sha256_hash = Column(String(64), nullable=False, unique=True)
    file_size = Column(Integer)
    processed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class OCRText(Base):
    __tablename__ = 'ocr_text'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    extracted_text = Column(Text)
    confidence = Column(Integer)
    processing_time = Column(Integer)  # milliseconds
    timestamp = Column(DateTime, default=datetime.utcnow)

class BrandHit(Base):
    __tablename__ = 'brand_hits'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=False)
    image_id = Column(Integer)
    brand_name = Column(String(100), nullable=False)
    matched_text = Column(String(500))
    confidence_score = Column(Integer)
    alert_sent = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_brand_timestamp', 'brand_name', 'timestamp'),
        Index('ix_alert_sent', 'alert_sent'),
    )
```

### Step 6: Database Session Factory
`src/database.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import DB_URL
from src.models import Base

engine = create_engine(DB_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
```

### Step 7: Telethon Message Collector
`src/telegram_collector.py`:
```python
import asyncio
import os
import hashlib
from telethon import TelegramClient, events
from sqlalchemy.exc import IntegrityError
from src.config import *
from src.database import SessionLocal
from src.models import TelegramMessage, ImageRecord
import logging

logger = logging.getLogger(__name__)

class TelegramCollector:
    def __init__(self):
        self.client = TelegramClient(
            TELEGRAM_SESSION_PATH,
            TELEGRAM_API_ID,
            TELEGRAM_API_HASH
        )

    async def start(self):
        await self.client.start()
        logger.info("Connected to Telegram")

        # Register handler for messages in specified groups
        @self.client.on(events.NewMessage(chats=TELEGRAM_GROUPS))
        async def message_handler(event):
            await self._process_message(event)

        await self.client.run_until_disconnected()

    async def _process_message(self, event):
        db = SessionLocal()
        try:
            # Store message
            message = TelegramMessage(
                chat_id=event.chat_id,
                message_id=event.message.id,
                sender_id=event.sender_id,
                text=event.message.text or "",
                has_media=bool(event.message.media)
            )
            db.add(message)
            db.flush()

            # Download and store media
            if event.message.media:
                await self._download_media(event.message, message.id, db)

            db.commit()
            logger.info(f"Processed message {event.message.id} from {event.chat_id}")

        except IntegrityError:
            logger.debug(f"Message {event.message.id} already exists")
            db.rollback()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            db.rollback()
        finally:
            db.close()

    async def _download_media(self, message, db_message_id, db):
        try:
            file_path = await message.download_media(file=MEDIA_DIR)
            if not file_path:
                return

            # Calculate hash for deduplication
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            image_record = ImageRecord(
                message_id=db_message_id,
                file_path=file_path,
                sha256_hash=file_hash,
                file_size=os.path.getsize(file_path)
            )
            db.add(image_record)

        except Exception as e:
            logger.error(f"Error downloading media: {e}")
```

### Step 8: OCR Engine
`src/ocr_engine.py`:
```python
import cv2
import pytesseract
import numpy as np
from PIL import Image
import time
from src.config import OCR_LANG
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        self.lang = OCR_LANG

    def extract_text(self, image_path: str) -> tuple[str, int]:
        """
        Extract text from image with preprocessing.
        Returns: (text, confidence_score)
        """
        start_time = time.time()

        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return "", 0

            processed_image = self._preprocess_image(image)

            # Extract text with confidence
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )

            # Filter high-confidence text
            text_parts = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:  # Filter low-confidence text
                    word = data['text'][i].strip()
                    if word:
                        text_parts.append(word)
                        confidences.append(int(conf))

            extracted_text = ' '.join(text_parts)
            avg_confidence = int(np.mean(confidences)) if confidences else 0

            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"OCR completed in {processing_time}ms, confidence: {avg_confidence}")

            return extracted_text, avg_confidence

        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return "", 0

    def _preprocess_image(self, image):
        """Apply preprocessing to improve OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel)

        return opening
```

### Step 9: Brand Matcher
`src/brand_matcher.py`:
```python
import re
from rapidfuzz import fuzz
from src.config import BRAND_KEYWORDS, FUZZY_THRESHOLD
import logging

logger = logging.getLogger(__name__)

class BrandMatcher:
    def __init__(self):
        self.keywords = [kw.strip() for kw in BRAND_KEYWORDS]
        self.threshold = FUZZY_THRESHOLD

    def find_matches(self, text: str) -> list[dict]:
        """
        Find brand mentions in text using fuzzy matching.
        Returns list of matches with confidence scores.
        """
        if not text:
            return []

        normalized_text = self._normalize_text(text)
        matches = []

        for keyword in self.keywords:
            normalized_keyword = self._normalize_text(keyword)

            # Check for exact matches first
            if normalized_keyword.lower() in normalized_text.lower():
                matches.append({
                    'brand': keyword,
                    'matched_text': keyword,
                    'confidence': 100,
                    'match_type': 'exact'
                })
                continue

            # Fuzzy matching for each word in text
            words = normalized_text.split()
            for word in words:
                if len(word) >= 3:  # Skip very short words
                    score = fuzz.ratio(normalized_keyword.lower(), word.lower())
                    if score >= self.threshold:
                        matches.append({
                            'brand': keyword,
                            'matched_text': word,
                            'confidence': score,
                            'match_type': 'fuzzy'
                        })

        # Deduplicate matches for same brand
        unique_matches = {}
        for match in matches:
            brand = match['brand']
            if brand not in unique_matches or match['confidence'] > unique_matches[brand]['confidence']:
                unique_matches[brand] = match

        return list(unique_matches.values())

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
```

### Step 10: Alert System
`src/alerting.py`:
```python
from telethon import TelegramClient
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_ALERT_CHAT_ID
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertSender:
    def __init__(self):
        self.client = TelegramClient('alert_bot', TELEGRAM_API_ID, TELEGRAM_API_HASH)

    async def send_brand_alert(self, match_data: dict):
        """Send formatted alert about brand mention"""
        try:
            await self.client.start()

            alert_text = self._format_alert(match_data)

            await self.client.send_message(
                TELEGRAM_ALERT_CHAT_ID,
                alert_text,
                parse_mode='html'
            )

            logger.info(f"Alert sent for brand: {match_data.get('brand')}")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            raise

    def _format_alert(self, data: dict) -> str:
        """Format alert message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        return f"""
🚨 <b>Brand Mention Detected</b>

<b>Brand:</b> {data.get('brand', 'Unknown')}
<b>Matched Text:</b> "{data.get('matched_text', '')}"
<b>Confidence:</b> {data.get('confidence', 0)}%
<b>Source:</b> Chat {data.get('chat_id', 'Unknown')}
<b>Message ID:</b> {data.get('message_id', 'Unknown')}
<b>Time:</b> {timestamp}

{data.get('additional_context', '')}
        """.strip()
```

### Step 11: Main Processing Pipeline
`src/processor.py`:
```python
import asyncio
import time
from sqlalchemy import and_
from src.database import SessionLocal
from src.models import ImageRecord, OCRText, BrandHit, TelegramMessage
from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher
from src.alerting import AlertSender
import logging

logger = logging.getLogger(__name__)

class FraudProcessor:
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.brand_matcher = BrandMatcher()
        self.alert_sender = AlertSender()

    async def process_pending_images(self):
        """Process unprocessed images in database"""
        while True:
            try:
                db = SessionLocal()

                # Get unprocessed images
                pending_images = db.query(ImageRecord).filter(
                    ImageRecord.processed == False
                ).limit(10).all()

                for image in pending_images:
                    await self._process_single_image(image, db)

                db.close()

                if not pending_images:
                    await asyncio.sleep(5)  # Wait if no work

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(10)

    async def _process_single_image(self, image: ImageRecord, db):
        """Process single image: OCR -> Match -> Alert"""
        try:
            # OCR extraction
            extracted_text, confidence = self.ocr_engine.extract_text(image.file_path)

            # Store OCR result
            ocr_record = OCRText(
                image_id=image.id,
                extracted_text=extracted_text,
                confidence=confidence
            )
            db.add(ocr_record)

            # Brand matching
            matches = self.brand_matcher.find_matches(extracted_text)

            # Store and alert for each match
            for match in matches:
                brand_hit = BrandHit(
                    message_id=image.message_id,
                    image_id=image.id,
                    brand_name=match['brand'],
                    matched_text=match['matched_text'],
                    confidence_score=match['confidence']
                )
                db.add(brand_hit)

                # Send alert
                await self._send_alert(brand_hit, db)

            # Mark image as processed
            image.processed = True
            db.commit()

            logger.info(f"Processed image {image.id}, found {len(matches)} matches")

        except Exception as e:
            logger.error(f"Error processing image {image.id}: {e}")
            db.rollback()

    async def _send_alert(self, hit: BrandHit, db):
        """Send alert and mark as sent"""
        try:
            # Get message context
            message = db.query(TelegramMessage).filter(
                TelegramMessage.id == hit.message_id
            ).first()

            alert_data = {
                'brand': hit.brand_name,
                'matched_text': hit.matched_text,
                'confidence': hit.confidence_score,
                'chat_id': message.chat_id if message else 'Unknown',
                'message_id': message.message_id if message else 'Unknown'
            }

            await self.alert_sender.send_brand_alert(alert_data)
            hit.alert_sent = True

        except Exception as e:
            logger.error(f"Failed to send alert for hit {hit.id}: {e}")
```

### Step 12: Main Application Entry
`src/main.py`:
```python
import asyncio
import logging
from src.telegram_collector import TelegramCollector
from src.processor import FraudProcessor
from src.database import init_db
from src.config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Run collector and processor concurrently"""
    # Initialize database
    init_db()

    # Create components
    collector = TelegramCollector()
    processor = FraudProcessor()

    # Run both concurrently
    await asyncio.gather(
        collector.start(),
        processor.process_pending_images()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 13: Docker Configuration
`Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p /app/data/media /app/data/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import asyncio; from src.database import engine; print('OK' if engine.execute('SELECT 1').scalar() else 'FAIL')"

CMD ["python", "-m", "src.main"]
```

### Step 14: Docker Compose
`docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: fraud-monitor-app
    environment:
      - DB_URL=postgresql+psycopg2://fraud_user:fraud_pass@db:5432/fraud_db
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/data/logs
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    container_name: fraud-monitor-db
    environment:
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: fraud_pass
      POSTGRES_DB: fraud_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_user -d fraud_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
```

### Step 15: Authentication Helper
`src/auth.py`:
```python
import asyncio
from telethon import TelegramClient
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_PATH

async def authenticate_session():
    """Interactive session authentication"""
    client = TelegramClient(TELEGRAM_SESSION_PATH, TELEGRAM_API_ID, TELEGRAM_API_HASH)

    print("Starting Telegram authentication...")
    await client.start()

    if await client.is_user_authorized():
        print("✅ Already authenticated")
        me = await client.get_me()
        print(f"Logged in as: {me.first_name} (@{me.username})")
    else:
        print("❌ Authentication required")

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(authenticate_session())
```

### Step 16: Final Integration Test
Create `test_integration.py`:
```python
import asyncio
import tempfile
from PIL import Image, ImageDraw, ImageFont
from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher

async def test_end_to_end():
    """Create test image with brand mention, run OCR and matching"""

    # Create test image with "CloudWalk" text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 30), "CloudWalk Payment Solution", fill='black')

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name)
        test_image_path = tmp.name

    # Test OCR
    ocr = OCREngine()
    text, confidence = ocr.extract_text(test_image_path)
    print(f"OCR Result: '{text}' (confidence: {confidence})")

    # Test brand matching
    matcher = BrandMatcher()
    matches = matcher.find_matches(text)
    print(f"Brand matches: {matches}")

    # Verify
    assert len(matches) > 0, "Should detect CloudWalk brand"
    assert matches[0]['brand'] == 'CloudWalk', "Should match CloudWalk exactly"

    print("✅ End-to-end test passed!")

if __name__ == "__main__":
    asyncio.run(test_end_to_end())
```

## Common Pitfalls & Tips

1. **Tesseract Language Packs**: Ensure correct language installed: `apt-get install tesseract-ocr-por` for Portuguese
2. **Session Persistence**: Mount `./data` volume to persist Telegram session across restarts
3. **Rate Limits**: Telegram has flood protection; implement exponential backoff
4. **OCR Accuracy**: Adjust morphology kernel size and threshold values for your image types
5. **Memory Usage**: For high-volume deployments, implement batch processing and cleanup old media files
6. **Database Indexes**: Monitor query performance and add indexes as needed for your access patterns