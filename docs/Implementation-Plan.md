# Implementation Plan

Complete step-by-step plan for implementing the fraud monitoring automation system from scratch to production deployment.

## Project Overview

**System Name**: Fraud Monitoring Telegram Bot
**Primary Goal**: Real-time detection of brand mentions in Telegram group messages with OCR-based image analysis and instant alerting
**Timeline**: 1-2 weeks for v1 implementation
**Team Size**: 2-3 developers

### Success Criteria
- ✅ Detect brand mentions within 30 seconds of message posting
- ✅ 95%+ accuracy for clear text in images (OCR confidence >70)
- ✅ Zero message loss during normal operation
- ✅ Support 10+ concurrent group monitoring
- ✅ Alert delivery within 5 seconds of detection
- ✅ Docker-ready deployment with <2 hour setup time

### Tech Stack
- **Language**: Python 3.11 with asyncio
- **Telegram**: Telethon (message ingestion) + Bot API (alerts)
- **Database**: PostgreSQL 15 with SQLAlchemy ORM
- **OCR**: Tesseract 5.x with OpenCV preprocessing
- **Matching**: RapidFuzz for fuzzy string matching
- **Deployment**: Docker + Docker Compose
- **Storage**: Local filesystem (v1) → S3/MinIO (v2)

## Implementation Phases

### Phase 1: Foundation & Environment Setup
**Duration**: 1-2 days
**Dependencies**: None

### Phase 2: Core Data Architecture
**Duration**: 2-3 days
**Dependencies**: Phase 1 complete

### Phase 3: Processing Engines
**Duration**: 3-4 days
**Dependencies**: Phase 2 complete

### Phase 4: Telegram Integration
**Duration**: 2-3 days
**Dependencies**: Phase 3 complete

### Phase 5: Application Orchestration
**Duration**: 2-3 days
**Dependencies**: Phase 4 complete

### Phase 6: Deployment & Operations
**Duration**: 2-3 days
**Dependencies**: Phase 5 complete

### Phase 7: Testing & Validation
**Duration**: 1-2 days
**Dependencies**: Phase 6 complete

## Detailed Implementation Steps

### Phase 1: Foundation & Environment Setup

#### Step 1: Create Project Foundation
**Estimated Time**: 30 minutes
**Assignee**: Lead Developer

```bash
# Create project structure
mkdir fraud-monitor && cd fraud-monitor
mkdir -p src tests docs data/media data/logs
mkdir -p scripts migrations

# Create essential files
touch requirements.txt
touch Dockerfile docker-compose.yml
touch .env.example .gitignore
touch alembic.ini
```

**Deliverables**:
- [ ] Project directory structure
- [ ] Initial requirements.txt with core dependencies
- [ ] .gitignore configured for Python + secrets
- [ ] .env.example template

**Validation**: Directory structure matches documentation specs

#### Step 2: Setup Python Environment
**Estimated Time**: 45 minutes
**Assignee**: Any Developer

```bash
# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements.txt Contents**:
```
telethon==1.32.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1
opencv-python-headless==4.8.1.78
pytesseract==0.3.10
rapidfuzz==3.5.2
python-dotenv==1.0.0
pytest==7.4.3
pytest-asyncio==0.21.1
```

**Deliverables**:
- [ ] Virtual environment activated
- [ ] All dependencies installed without errors
- [ ] Requirements.txt frozen with exact versions

**Validation**: `python -c "import telethon, sqlalchemy, cv2, pytesseract, rapidfuzz"` succeeds

#### Step 3: Create Configuration Management
**Estimated Time**: 1 hour
**Assignee**: Lead Developer

Create `src/config.py` with comprehensive environment variable handling:

```python
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Telegram Configuration
TELEGRAM_API_ID = int(os.getenv('TELEGRAM_API_ID', '0'))
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH', '')
TELEGRAM_SESSION_PATH = os.getenv('TELEGRAM_SESSION_PATH', './data/telethon.session')
TELEGRAM_GROUPS = [g.strip() for g in os.getenv('TELEGRAM_GROUPS', '').split(',') if g.strip()]
TELEGRAM_ALERT_CHAT_ID = int(os.getenv('TELEGRAM_ALERT_CHAT_ID', '0'))

# Database Configuration
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://fraud_user:fraud_pass@localhost:5432/fraud_db')

# Storage Configuration
MEDIA_DIR = os.getenv('MEDIA_DIR', './data/media')
MAX_MEDIA_SIZE = int(os.getenv('MAX_MEDIA_SIZE', '10485760'))  # 10MB

# OCR Configuration
OCR_LANG = os.getenv('OCR_LANG', 'eng')
OCR_CONFIDENCE_THRESHOLD = int(os.getenv('OCR_CONFIDENCE_THRESHOLD', '30'))
OCR_TIMEOUT = int(os.getenv('OCR_TIMEOUT', '10'))

# Brand Detection
BRAND_KEYWORDS = [k.strip() for k in os.getenv('BRAND_KEYWORDS', 'CloudWalk,InfinitePay,Visa,Mastercard').split(',')]
FUZZY_THRESHOLD = int(os.getenv('FUZZY_THRESHOLD', '85'))
MATCH_MIN_LENGTH = int(os.getenv('MATCH_MIN_LENGTH', '3'))

# Processing Configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
PROCESSING_INTERVAL = int(os.getenv('PROCESSING_INTERVAL', '5'))
RETRY_MAX_ATTEMPTS = int(os.getenv('RETRY_MAX_ATTEMPTS', '3'))
RETRY_BACKOFF_FACTOR = float(os.getenv('RETRY_BACKOFF_FACTOR', '2.0'))

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', './data/logs/app.log')

# Security
SESSION_ENCRYPTION_KEY = os.getenv('SESSION_ENCRYPTION_KEY', '')
ALERT_RATE_LIMIT = int(os.getenv('ALERT_RATE_LIMIT', '10'))
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
```

**Deliverables**:
- [ ] src/config.py with all environment variables
- [ ] Configuration validation function
- [ ] Updated .env.example with all variables

**Validation**: Import config and verify all variables load correctly

### Phase 2: Core Data Architecture

#### Step 4: Implement Database Models
**Estimated Time**: 2 hours
**Assignee**: Backend Developer

Create `src/models.py` with complete SQLAlchemy models:

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class TelegramMessage(Base):
    __tablename__ = 'telegram_messages'

    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, nullable=False)
    message_id = Column(BigInteger, nullable=False)
    sender_id = Column(BigInteger)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    has_media = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)

    __table_args__ = (
        Index('ix_chat_message', 'chat_id', 'message_id', unique=True),
        Index('ix_timestamp', 'timestamp'),
        Index('ix_processed_media', 'processed', 'has_media'),
        Index('ix_chat_recent', 'chat_id', 'timestamp'),
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

    __table_args__ = (
        Index('ix_unprocessed', 'processed', 'timestamp'),
        Index('ix_message_images', 'message_id'),
        Index('ix_file_size', 'file_size'),
    )

class OCRText(Base):
    __tablename__ = 'ocr_text'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    extracted_text = Column(Text)
    confidence = Column(Integer)  # 0-100
    processing_time = Column(Integer)  # milliseconds
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_image_ocr', 'image_id'),
        Index('ix_confidence', 'confidence'),
    )

class BrandHit(Base):
    __tablename__ = 'brand_hits'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=False)
    image_id = Column(Integer)
    brand_name = Column(String(100), nullable=False)
    matched_text = Column(String(500))
    confidence_score = Column(Integer)  # 0-100
    match_type = Column(String(20), default='fuzzy')  # 'exact', 'fuzzy'
    alert_sent = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_brand_recent', 'brand_name', 'timestamp'),
        Index('ix_unsent_alerts', 'alert_sent', 'timestamp'),
        Index('ix_confidence_score', 'confidence_score'),
    )
```

**Deliverables**:
- [ ] Complete SQLAlchemy models with relationships
- [ ] Strategic indexes for performance
- [ ] Unique constraints for deduplication
- [ ] Validation constraints (confidence scores, etc.)

**Validation**: Models can be imported and introspected without errors

#### Step 5: Setup Database Connection & Session Factory
**Estimated Time**: 1 hour
**Assignee**: Backend Developer

Create `src/database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.config import DB_URL
from src.models import Base
import logging

logger = logging.getLogger(__name__)

# Create engine with connection pooling
engine = create_engine(
    DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database schema"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def check_db_connection():
    """Health check for database connectivity"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
```

**Deliverables**:
- [ ] Database engine with connection pooling
- [ ] Session factory for database operations
- [ ] Database initialization function
- [ ] Connection health check function

**Validation**: Database connection and table creation works with PostgreSQL

### Phase 3: Processing Engines

#### Step 6: Implement OCR Engine
**Estimated Time**: 3 hours
**Assignee**: ML/Computer Vision Developer

Create `src/ocr_engine.py`:

```python
import cv2
import pytesseract
import numpy as np
from PIL import Image
import time
import os
from src.config import OCR_LANG, OCR_CONFIDENCE_THRESHOLD, OCR_TIMEOUT
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        self.lang = OCR_LANG
        self.confidence_threshold = OCR_CONFIDENCE_THRESHOLD
        self.timeout = OCR_TIMEOUT

        # Verify tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Tesseract not properly installed: {e}")
            raise

    def extract_text(self, image_path: str) -> tuple[str, int, int]:
        """
        Extract text from image with preprocessing.
        Returns: (extracted_text, confidence_score, processing_time_ms)
        """
        start_time = time.time()

        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return "", 0, 0

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return "", 0, 0

            # Apply preprocessing
            processed_image = self._preprocess_image(image)

            # Extract text with detailed data
            custom_config = f'--oem 3 --psm 6 -l {self.lang}'
            data = pytesseract.image_to_data(
                processed_image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            # Filter and combine text based on confidence
            text_parts = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if int(conf) > self.confidence_threshold:
                    word = data['text'][i].strip()
                    if word and len(word) > 1:
                        text_parts.append(word)
                        confidences.append(int(conf))

            extracted_text = ' '.join(text_parts)
            avg_confidence = int(np.mean(confidences)) if confidences else 0
            processing_time = int((time.time() - start_time) * 1000)

            logger.debug(f"OCR extracted {len(text_parts)} words, avg confidence: {avg_confidence}%")
            return extracted_text, avg_confidence, processing_time

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OCR failed for {image_path}: {e}")
            return "", 0, processing_time

    def _preprocess_image(self, image):
        """Apply preprocessing pipeline for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding for better text separation
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up text
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel, iterations=1)

        return opening

    def health_check(self) -> bool:
        """Verify OCR engine is working correctly"""
        try:
            # Create simple test image
            test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, 'TEST', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Test OCR
            result = pytesseract.image_to_string(test_image, lang=self.lang)
            return 'TEST' in result.upper()
        except Exception as e:
            logger.error(f"OCR health check failed: {e}")
            return False
```

**Deliverables**:
- [ ] OCR engine with Tesseract integration
- [ ] Image preprocessing pipeline
- [ ] Confidence-based text filtering
- [ ] Performance metrics tracking
- [ ] Error handling and logging

**Validation**: OCR can extract text from test images with >80% confidence

#### Step 7: Create Brand Matching System
**Estimated Time**: 2 hours
**Assignee**: Backend Developer

Create `src/brand_matcher.py`:

```python
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

        # Filter words by minimum length
        valid_words = [w for w in words if len(w) >= self.min_length]
        if not valid_words:
            return None

        # Use RapidFuzz for efficient fuzzy matching
        result = process.extractOne(
            keyword.lower(),
            [w.lower() for w in valid_words],
            scorer=fuzz.ratio
        )

        if result and result[1] >= self.threshold:
            matched_word = valid_words[result[2]]  # Get original case
            return {
                'text': matched_word,
                'score': int(result[1]),
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
            self.normalized_keywords.pop(keyword, None)
            logger.info(f"Removed brand keyword: {keyword}")

    def health_check(self) -> bool:
        """Verify brand matcher is working correctly"""
        try:
            test_text = "This is a CloudWalk payment system test"
            matches = self.find_matches(test_text)
            return len(matches) > 0 and any(m['brand'] == 'CloudWalk' for m in matches)
        except Exception as e:
            logger.error(f"Brand matcher health check failed: {e}")
            return False
```

**Deliverables**:
- [ ] Fuzzy string matching with RapidFuzz
- [ ] Text normalization for better accuracy
- [ ] Exact and fuzzy matching modes
- [ ] Configurable confidence thresholds
- [ ] Dynamic keyword management

**Validation**: Can detect "CloudWalk" in various text formats with >90% accuracy

### Phase 4: Telegram Integration

#### Step 8: Build Telegram Message Collector
**Estimated Time**: 3 hours
**Assignee**: Backend Developer

Create `src/telegram_collector.py`:

```python
import asyncio
import os
import hashlib
import aiofiles
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
        self.groups = TELEGRAM_GROUPS
        self.media_dir = MEDIA_DIR
        self.max_media_size = MAX_MEDIA_SIZE
        self.session_authenticated = False

        # Ensure media directory exists
        os.makedirs(self.media_dir, exist_ok=True)

        logger.info(f"Initialized TelegramCollector for {len(self.groups)} groups")

    async def start(self):
        """Start the Telegram client and message collection"""
        try:
            await self.client.start()
            self.session_authenticated = await self.client.is_user_authorized()

            if not self.session_authenticated:
                logger.error("Telegram session not authenticated. Run authentication first.")
                raise Exception("Not authenticated with Telegram")

            me = await self.client.get_me()
            logger.info(f"Connected to Telegram as: {me.first_name} (@{me.username or 'no_username'})")

            # Register message handler
            @self.client.on(events.NewMessage(chats=self.groups))
            async def message_handler(event):
                await self._process_message(event)

            logger.info(f"Listening for messages in {len(self.groups)} groups...")
            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"Failed to start Telegram client: {e}")
            raise

    async def _process_message(self, event):
        """Process incoming message and store in database"""
        db = SessionLocal()
        try:
            # Extract message data
            message_data = {
                'chat_id': event.chat_id,
                'message_id': event.message.id,
                'sender_id': event.sender_id,
                'text': event.message.text or "",
                'has_media': bool(event.message.media),
                'processed': False
            }

            # Create message record
            message = TelegramMessage(**message_data)
            db.add(message)
            db.flush()  # Get the ID

            logger.info(f"Processing message {event.message.id} from chat {event.chat_id}")

            # Download and store media if present
            if event.message.media:
                await self._download_media(event.message, message.id, db)

            db.commit()
            logger.debug(f"Successfully stored message {event.message.id}")

        except IntegrityError:
            logger.debug(f"Message {event.message.id} already exists, skipping")
            db.rollback()
        except Exception as e:
            logger.error(f"Error processing message {event.message.id}: {e}")
            db.rollback()
            # Don't re-raise to keep collector running
        finally:
            db.close()

    async def _download_media(self, message, db_message_id: int, db):
        """Download media file and create image record"""
        try:
            # Check file size before download
            if hasattr(message.media, 'document') and message.media.document:
                if message.media.document.size > self.max_media_size:
                    logger.warning(f"Media file too large ({message.media.document.size} bytes), skipping")
                    return

            # Download media file
            file_path = await message.download_media(file=self.media_dir)
            if not file_path:
                logger.warning("Failed to download media file")
                return

            # Calculate file hash for deduplication
            sha256_hash = await self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)

            # Create image record
            image_record = ImageRecord(
                message_id=db_message_id,
                file_path=file_path,
                sha256_hash=sha256_hash,
                file_size=file_size,
                processed=False
            )
            db.add(image_record)

            logger.info(f"Downloaded media: {file_path} ({file_size} bytes)")

        except IntegrityError:
            # Duplicate file hash, remove downloaded file
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            logger.debug(f"Duplicate media file detected (hash: {sha256_hash[:8]}...)")
            db.rollback()
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise

    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()

        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in f:
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def health_check(self) -> bool:
        """Check if Telegram client is healthy"""
        try:
            if not self.session_authenticated:
                return False

            await self.client.get_me()
            return True
        except Exception as e:
            logger.error(f"Telegram health check failed: {e}")
            return False

    async def stop(self):
        """Gracefully stop the collector"""
        if self.client.is_connected():
            await self.client.disconnect()
        logger.info("Telegram collector stopped")
```

**Deliverables**:
- [ ] Telethon client integration
- [ ] Message event handling
- [ ] Media file downloading with deduplication
- [ ] Database persistence with error handling
- [ ] Health check functionality

**Validation**: Can connect to Telegram and store messages in database

#### Step 9: Implement Alerting System
**Estimated Time**: 2 hours
**Assignee**: Backend Developer

Create `src/alerting.py`:

```python
import asyncio
from telethon import TelegramClient
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_ALERT_CHAT_ID, ALERT_RATE_LIMIT
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import time

logger = logging.getLogger(__name__)

class AlertSender:
    def __init__(self):
        self.client = TelegramClient('alert_bot', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        self.alert_chat_id = TELEGRAM_ALERT_CHAT_ID
        self.rate_limit = ALERT_RATE_LIMIT
        self.alert_history = []  # Track recent alerts for rate limiting

        logger.info(f"Initialized AlertSender for chat {self.alert_chat_id}")

    async def send_brand_alert(self, alert_data: Dict) -> bool:
        """Send formatted alert about brand mention"""
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                logger.warning("Alert rate limit exceeded, queuing alert")
                return False

            # Connect if not already connected
            if not self.client.is_connected():
                await self.client.start()

            # Format and send alert
            alert_text = self._format_alert(alert_data)

            await self.client.send_message(
                self.alert_chat_id,
                alert_text,
                parse_mode='html'
            )

            # Track alert for rate limiting
            self._record_alert()

            logger.info(f"Alert sent for brand: {alert_data.get('brand', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    async def send_system_alert(self, severity: str, message: str) -> bool:
        """Send system status alert"""
        try:
            if not self._check_rate_limit():
                logger.warning("System alert rate limit exceeded")
                return False

            if not self.client.is_connected():
                await self.client.start()

            severity_emoji = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }

            alert_text = f"""
{severity_emoji.get(severity, '📢')} <b>System Alert - {severity}</b>

<b>Message:</b> {message}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
<b>System:</b> Fraud Monitoring Bot

<i>Automated system notification</i>
            """.strip()

            await self.client.send_message(
                self.alert_chat_id,
                alert_text,
                parse_mode='html'
            )

            self._record_alert()
            logger.info(f"System alert sent: {severity} - {message}")
            return True

        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
            return False

    def _format_alert(self, data: Dict) -> str:
        """Format brand detection alert message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        # Build context information
        context_parts = []
        if data.get('extracted_text'):
            text_preview = data['extracted_text'][:100]
            if len(data['extracted_text']) > 100:
                text_preview += "..."
            context_parts.append(f"<b>OCR Text:</b> \"{text_preview}\"")

        if data.get('image_path'):
            context_parts.append(f"<b>Image:</b> {os.path.basename(data['image_path'])}")

        context_section = '\n'.join(context_parts) if context_parts else ""

        alert_text = f"""
🚨 <b>Brand Mention Detected</b>

<b>Brand:</b> {data.get('brand', 'Unknown')}
<b>Matched Text:</b> "{data.get('matched_text', '')}"
<b>Confidence:</b> {data.get('confidence', 0)}%
<b>Match Type:</b> {data.get('match_type', 'unknown').title()}
<b>Source Chat:</b> {data.get('chat_id', 'Unknown')}
<b>Message ID:</b> {data.get('message_id', 'Unknown')}
<b>Detection Time:</b> {timestamp}

{context_section}

<i>🤖 Automated fraud detection alert</i>
        """.strip()

        return alert_text

    def _check_rate_limit(self) -> bool:
        """Check if we can send another alert based on rate limiting"""
        now = datetime.now()
        # Remove alerts older than 1 minute
        self.alert_history = [
            alert_time for alert_time in self.alert_history
            if now - alert_time < timedelta(minutes=1)
        ]

        return len(self.alert_history) < self.rate_limit

    def _record_alert(self):
        """Record that an alert was sent"""
        self.alert_history.append(datetime.now())

    async def health_check(self) -> bool:
        """Check if alert system is working"""
        try:
            if not self.client.is_connected():
                await self.client.start()

            # Try to get chat info
            chat = await self.client.get_entity(self.alert_chat_id)
            return chat is not None
        except Exception as e:
            logger.error(f"Alert system health check failed: {e}")
            return False

    async def test_alert(self) -> bool:
        """Send a test alert to verify functionality"""
        test_data = {
            'brand': 'TEST',
            'matched_text': 'TEST ALERT',
            'confidence': 100,
            'match_type': 'exact',
            'chat_id': 'SYSTEM_TEST',
            'message_id': 'TEST_MSG_001'
        }

        return await self.send_brand_alert(test_data)

    async def stop(self):
        """Gracefully stop the alert sender"""
        if self.client.is_connected():
            await self.client.disconnect()
        logger.info("Alert sender stopped")
```

**Deliverables**:
- [ ] Telegram bot integration for alerts
- [ ] Formatted alert messages with context
- [ ] Rate limiting to prevent spam
- [ ] System alerts for operational issues
- [ ] Health checks and testing functions

**Validation**: Can send test alerts to configured Telegram chat

### Phase 5: Application Orchestration

#### Step 10: Create Main Processing Pipeline
**Estimated Time**: 3 hours
**Assignee**: Lead Developer

Create `src/processor.py`:

```python
import asyncio
import time
from datetime import datetime, timedelta
from sqlalchemy import and_, func
from src.database import SessionLocal
from src.models import ImageRecord, OCRText, BrandHit, TelegramMessage
from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher
from src.alerting import AlertSender
from src.config import BATCH_SIZE, PROCESSING_INTERVAL, RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_FACTOR
import logging

logger = logging.getLogger(__name__)

class FraudProcessor:
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.brand_matcher = BrandMatcher()
        self.alert_sender = AlertSender()
        self.batch_size = BATCH_SIZE
        self.processing_interval = PROCESSING_INTERVAL
        self.max_retries = RETRY_MAX_ATTEMPTS
        self.backoff_factor = RETRY_BACKOFF_FACTOR
        self.running = False
        self.stats = {
            'processed_images': 0,
            'brand_hits': 0,
            'alerts_sent': 0,
            'processing_errors': 0,
            'start_time': None
        }

        logger.info("Initialized FraudProcessor")

    async def start(self):
        """Start the main processing loop"""
        self.running = True
        self.stats['start_time'] = datetime.now()

        logger.info("Starting fraud processor...")

        # Run health checks
        if not await self._health_check():
            raise Exception("Health checks failed, cannot start processor")

        try:
            while self.running:
                await self._process_batch()
                await asyncio.sleep(self.processing_interval)

        except KeyboardInterrupt:
            logger.info("Received stop signal")
        except Exception as e:
            logger.error(f"Fatal error in processor: {e}")
            raise
        finally:
            await self.stop()

    async def _process_batch(self):
        """Process a batch of unprocessed images"""
        db = SessionLocal()
        try:
            # Get unprocessed images
            pending_images = db.query(ImageRecord).filter(
                ImageRecord.processed == False
            ).order_by(ImageRecord.timestamp).limit(self.batch_size).all()

            if not pending_images:
                logger.debug("No pending images to process")
                return

            logger.info(f"Processing batch of {len(pending_images)} images")

            # Process each image
            for image in pending_images:
                try:
                    await self._process_single_image(image, db)
                except Exception as e:
                    logger.error(f"Error processing image {image.id}: {e}")
                    self.stats['processing_errors'] += 1
                    # Continue with next image

            db.commit()

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            db.rollback()
        finally:
            db.close()

    async def _process_single_image(self, image: ImageRecord, db):
        """Process single image: OCR -> Brand Matching -> Alerting"""
        start_time = time.time()

        try:
            logger.debug(f"Processing image {image.id}: {image.file_path}")

            # Step 1: OCR Text Extraction
            extracted_text, confidence, ocr_time = self.ocr_engine.extract_text(image.file_path)

            # Store OCR result
            ocr_record = OCRText(
                image_id=image.id,
                extracted_text=extracted_text,
                confidence=confidence,
                processing_time=ocr_time
            )
            db.add(ocr_record)
            db.flush()

            logger.debug(f"OCR extracted {len(extracted_text)} characters with {confidence}% confidence")

            # Step 2: Brand Matching (only if we have text)
            brand_matches = []
            if extracted_text.strip():
                brand_matches = self.brand_matcher.find_matches(extracted_text)
                logger.debug(f"Found {len(brand_matches)} potential brand matches")

            # Step 3: Store Brand Hits and Send Alerts
            for match in brand_matches:
                brand_hit = BrandHit(
                    message_id=image.message_id,
                    image_id=image.id,
                    brand_name=match['brand'],
                    matched_text=match['matched_text'],
                    confidence_score=match['confidence'],
                    match_type=match['match_type'],
                    alert_sent=False
                )
                db.add(brand_hit)
                db.flush()

                # Send alert
                alert_sent = await self._send_alert(brand_hit, image, extracted_text, db)
                brand_hit.alert_sent = alert_sent

                if alert_sent:
                    self.stats['alerts_sent'] += 1

                self.stats['brand_hits'] += 1
                logger.info(f"Brand hit: {match['brand']} (confidence: {match['confidence']}%)")

            # Mark image as processed
            image.processed = True
            self.stats['processed_images'] += 1

            processing_time = time.time() - start_time
            logger.debug(f"Completed processing image {image.id} in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to process image {image.id}: {e}")
            # Don't mark as processed so it can be retried
            raise

    async def _send_alert(self, hit: BrandHit, image: ImageRecord, extracted_text: str, db) -> bool:
        """Send alert for brand hit"""
        try:
            # Get message context
            message = db.query(TelegramMessage).filter(
                TelegramMessage.id == hit.message_id
            ).first()

            alert_data = {
                'brand': hit.brand_name,
                'matched_text': hit.matched_text,
                'confidence': hit.confidence_score,
                'match_type': hit.match_type,
                'chat_id': message.chat_id if message else 'Unknown',
                'message_id': message.message_id if message else 'Unknown',
                'extracted_text': extracted_text,
                'image_path': image.file_path
            }

            success = await self.alert_sender.send_brand_alert(alert_data)

            if success:
                logger.info(f"Alert sent for brand hit: {hit.brand_name}")
            else:
                logger.warning(f"Failed to send alert for brand hit: {hit.brand_name}")

            return success

        except Exception as e:
            logger.error(f"Error sending alert for hit {hit.id}: {e}")
            return False

    async def _health_check(self) -> bool:
        """Comprehensive health check of all components"""
        logger.info("Running health checks...")

        checks = {
            'ocr_engine': self.ocr_engine.health_check(),
            'brand_matcher': self.brand_matcher.health_check(),
            'alert_sender': await self.alert_sender.health_check()
        }

        all_healthy = all(checks.values())

        for component, healthy in checks.items():
            status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
            logger.info(f"{component}: {status}")

        return all_healthy

    def get_stats(self) -> dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime'] = str(datetime.now() - stats['start_time'])
        return stats

    async def stop(self):
        """Gracefully stop the processor"""
        self.running = False
        await self.alert_sender.stop()

        # Send final stats
        stats = self.get_stats()
        logger.info(f"Processor stopped. Final stats: {stats}")
```

**Deliverables**:
- [ ] Main processing pipeline orchestration
- [ ] Batch processing with configurable intervals
- [ ] Error handling with retry logic
- [ ] Health checks for all components
- [ ] Processing statistics and monitoring

**Validation**: Can process test images end-to-end and send alerts

#### Step 11: Build Main Application Entry Point
**Estimated Time**: 1 hour
**Assignee**: Lead Developer

Create `src/main.py`:

```python
import asyncio
import logging
import signal
import sys
from datetime import datetime
from src.telegram_collector import TelegramCollector
from src.processor import FraudProcessor
from src.database import init_db, check_db_connection
from src.config import LOG_LEVEL, LOG_FILE
import os

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('telethon').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class FraudMonitorApp:
    def __init__(self):
        self.collector = None
        self.processor = None
        self.shutdown_event = asyncio.Event()

    async def startup(self):
        """Initialize application components"""
        logger.info("🚀 Starting Fraud Monitoring System")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")

        # Initialize database
        logger.info("Initializing database...")
        if not check_db_connection():
            raise Exception("Database connection failed")

        init_db()
        logger.info("✅ Database initialized")

        # Initialize components
        self.collector = TelegramCollector()
        self.processor = FraudProcessor()

        logger.info("✅ Components initialized")

    async def run(self):
        """Run the main application"""
        try:
            await self.startup()

            # Setup signal handlers for graceful shutdown
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)

            logger.info("Starting concurrent services...")

            # Run collector and processor concurrently
            tasks = await asyncio.gather(
                self.collector.start(),
                self.processor.start(),
                self._monitor_health(),
                return_exceptions=True
            )

            # Check if any task failed
            for i, task in enumerate(tasks):
                if isinstance(task, Exception):
                    service_name = ['collector', 'processor', 'monitor'][i]
                    logger.error(f"Service {service_name} failed: {task}")

        except Exception as e:
            logger.error(f"Application failed to start: {e}")
            raise
        finally:
            await self.shutdown()

    async def _monitor_health(self):
        """Monitor system health and log statistics"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if self.processor:
                    stats = self.processor.get_stats()
                    logger.info(f"📊 Processing Stats: {stats}")

                # Check component health
                if self.collector and not await self.collector.health_check():
                    logger.error("❌ Telegram collector health check failed")

                if self.processor and not await self.processor._health_check():
                    logger.error("❌ Processor health check failed")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🛑 Shutting down Fraud Monitoring System...")

        # Signal shutdown
        self.shutdown_event.set()

        # Stop components
        if self.processor:
            await self.processor.stop()

        if self.collector:
            await self.collector.stop()

        logger.info("✅ Shutdown complete")

async def main():
    """Main application entry point"""
    setup_logging()

    app = FraudMonitorApp()
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

**Deliverables**:
- [ ] Main application orchestration
- [ ] Graceful startup and shutdown
- [ ] Signal handling for production deployment
- [ ] Health monitoring and statistics logging
- [ ] Comprehensive error handling

**Validation**: Application starts, runs all components, and shuts down gracefully

### Phase 6: Deployment & Operations

#### Step 12: Setup Docker Configuration
**Estimated Time**: 2 hours
**Assignee**: DevOps/Lead Developer

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    tesseract-ocr-spa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Verify tesseract installation
RUN tesseract --version

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/media /app/data/logs
RUN chmod 755 /app/data

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "from src.database import check_db_connection; exit(0 if check_db_connection() else 1)"

# Default command
CMD ["python", "-m", "src.main"]
```

Create `docker-compose.yml`:

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
    networks:
      - fraud-monitor
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  db:
    image: postgres:15-alpine
    container_name: fraud-monitor-db
    environment:
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: fraud_pass
      POSTGRES_DB: fraud_db
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8"
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
    networks:
      - fraud-monitor

volumes:
  postgres_data:
    driver: local

networks:
  fraud-monitor:
    driver: bridge
```

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: fraud-monitor-app-prod
    environment:
      - DB_URL=postgresql+psycopg2://fraud_user:${DB_PASSWORD}@db:5432/fraud_db
      - LOG_LEVEL=INFO
    env_file:
      - .env.prod
    volumes:
      - /opt/fraud-monitor/data:/app/data
      - /opt/fraud-monitor/logs:/app/data/logs
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - fraud-monitor
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"

  db:
    image: postgres:15-alpine
    container_name: fraud-monitor-db-prod
    environment:
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: fraud_db
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    # Remove external port mapping for security
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_user -d fraud_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - fraud-monitor

volumes:
  postgres_prod_data:
    driver: local

networks:
  fraud-monitor:
    driver: bridge
    internal: true  # No external access for production
```

Create `.dockerignore`:

```
# Version control
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.egg-info
.pytest_cache
.coverage

# Environment
.env*
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Documentation
docs/
*.md
!README.md

# Data
data/
logs/
*.log

# Tests
tests/
test_*.py
*_test.py
```

**Deliverables**:
- [ ] Production-ready Dockerfile with Tesseract
- [ ] Docker Compose for development and production
- [ ] Health checks and proper logging
- [ ] Volume management for persistence
- [ ] Network isolation for security

**Validation**: Docker containers build and start successfully

#### Step 13: Create Authentication Helper
**Estimated Time**: 1 hour
**Assignee**: Backend Developer

Create `src/auth.py`:

```python
import asyncio
import sys
from telethon import TelegramClient
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_PATH
import logging

logger = logging.getLogger(__name__)

class TelegramAuthenticator:
    def __init__(self):
        self.client = TelegramClient(
            TELEGRAM_SESSION_PATH,
            TELEGRAM_API_ID,
            TELEGRAM_API_HASH
        )

    async def authenticate_session(self) -> bool:
        """Interactive session authentication"""
        try:
            print("🔐 Starting Telegram authentication...")
            print(f"Session will be saved to: {TELEGRAM_SESSION_PATH}")

            await self.client.start()

            if await self.client.is_user_authorized():
                me = await self.client.get_me()
                print(f"✅ Already authenticated as: {me.first_name}")
                if me.username:
                    print(f"   Username: @{me.username}")
                print(f"   User ID: {me.id}")
                return True
            else:
                print("❌ Authentication required")
                print("Please complete authentication in the Telegram client")
                return False

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            print(f"❌ Authentication error: {e}")
            return False
        finally:
            await self.client.disconnect()

    async def verify_group_access(self) -> bool:
        """Verify access to configured groups"""
        try:
            await self.client.start()

            if not await self.client.is_user_authorized():
                print("❌ Not authenticated. Run authentication first.")
                return False

            from src.config import TELEGRAM_GROUPS

            print(f"🔍 Verifying access to {len(TELEGRAM_GROUPS)} groups...")

            accessible_groups = []
            for group in TELEGRAM_GROUPS:
                try:
                    entity = await self.client.get_entity(group)
                    accessible_groups.append({
                        'identifier': group,
                        'title': getattr(entity, 'title', 'Unknown'),
                        'id': entity.id,
                        'type': 'Channel' if hasattr(entity, 'broadcast') else 'Group'
                    })
                    print(f"   ✅ {group}: {entity.title} (ID: {entity.id})")
                except Exception as e:
                    print(f"   ❌ {group}: Access denied or not found ({e})")

            print(f"\n📊 Summary: {len(accessible_groups)}/{len(TELEGRAM_GROUPS)} groups accessible")

            if len(accessible_groups) == 0:
                print("⚠️ No groups accessible. Please:")
                print("   1. Join the groups you want to monitor")
                print("   2. Ensure the groups allow your account")
                print("   3. Check group usernames/IDs in configuration")
                return False

            return True

        except Exception as e:
            logger.error(f"Group verification failed: {e}")
            print(f"❌ Group verification error: {e}")
            return False
        finally:
            await self.client.disconnect()

    async def test_alert_chat(self) -> bool:
        """Test access to alert chat"""
        try:
            await self.client.start()

            from src.config import TELEGRAM_ALERT_CHAT_ID

            print(f"📢 Testing alert chat access (ID: {TELEGRAM_ALERT_CHAT_ID})...")

            chat = await self.client.get_entity(TELEGRAM_ALERT_CHAT_ID)
            print(f"   ✅ Alert chat: {getattr(chat, 'title', 'Private Chat')}")

            # Test sending a message
            test_message = "🧪 Test message from Fraud Monitor setup"
            await self.client.send_message(TELEGRAM_ALERT_CHAT_ID, test_message)
            print("   ✅ Test message sent successfully")

            return True

        except Exception as e:
            print(f"   ❌ Alert chat test failed: {e}")
            print("   Please ensure:")
            print("   1. The chat/group exists")
            print("   2. Your account has access to it")
            print("   3. You have permission to send messages")
            return False
        finally:
            await self.client.disconnect()

async def main():
    """Main authentication workflow"""
    authenticator = TelegramAuthenticator()

    print("=" * 60)
    print("🤖 Fraud Monitor - Telegram Setup")
    print("=" * 60)

    # Step 1: Authenticate
    if not await authenticator.authenticate_session():
        print("\n❌ Authentication failed. Please check your API credentials.")
        sys.exit(1)

    # Step 2: Verify group access
    print("\n" + "-" * 40)
    if not await authenticator.verify_group_access():
        print("\n⚠️ Group access issues detected.")
        print("The system will start but may not receive messages.")

    # Step 3: Test alert chat
    print("\n" + "-" * 40)
    if not await authenticator.test_alert_chat():
        print("\n⚠️ Alert chat access issues detected.")
        print("Alerts may not be delivered.")

    print("\n" + "=" * 60)
    print("✅ Setup complete! You can now start the fraud monitor.")
    print("💡 Run: docker-compose up -d")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
```

**Deliverables**:
- [ ] Interactive Telegram authentication
- [ ] Group access verification
- [ ] Alert chat testing
- [ ] Comprehensive setup workflow

**Validation**: Can authenticate and verify access to configured chats

#### Step 14: Setup Database Migrations
**Estimated Time**: 1 hour
**Assignee**: Backend Developer

Create `alembic.ini`:

```ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+psycopg2://fraud_user:fraud_pass@localhost:5432/fraud_db

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Initialize Alembic and create initial migration:

```bash
alembic init migrations
alembic revision --autogenerate -m "Initial schema with all tables"
```

Create `migrations/env.py` modifications:

```python
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import Base
from src.config import DB_URL

config = context.config
config.set_main_option('sqlalchemy.url', DB_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create migration management script `scripts/migrate.py`:

```python
#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(cmd):
    """Run shell command and handle errors"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate.py [upgrade|downgrade|current|history]")
        sys.exit(1)

    action = sys.argv[1]

    # Set environment for migrations
    os.environ.setdefault('PYTHONPATH', '.')

    if action == 'upgrade':
        print("Running database migrations...")
        if run_command("alembic upgrade head"):
            print("✅ Database migrations completed successfully")
        else:
            print("❌ Migration failed")
            sys.exit(1)

    elif action == 'downgrade':
        revision = sys.argv[2] if len(sys.argv) > 2 else '-1'
        print(f"Downgrading to revision: {revision}")
        run_command(f"alembic downgrade {revision}")

    elif action == 'current':
        print("Current database revision:")
        run_command("alembic current")

    elif action == 'history':
        print("Migration history:")
        run_command("alembic history")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Deliverables**:
- [ ] Alembic configuration for migrations
- [ ] Initial database schema migration
- [ ] Migration management scripts
- [ ] Environment-aware configuration

**Validation**: Can run migrations and create database schema

### Phase 7: Testing & Validation

#### Step 15: Create Integration Tests
**Estimated Time**: 3 hours
**Assignee**: QA/Backend Developer

Create `tests/test_integration.py`:

```python
import pytest
import asyncio
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import shutil

from src.models import Base, TelegramMessage, ImageRecord, OCRText, BrandHit
from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher
from src.processor import FraudProcessor
from src.database import init_db

class TestIntegration:

    @pytest.fixture(autouse=True)
    def setup_test_db(self):
        """Setup isolated test database"""
        # Use in-memory SQLite for testing
        self.engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.db = Session()

        # Create temp directory for test media
        self.temp_dir = tempfile.mkdtemp()

        yield

        # Cleanup
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def create_test_image(self, text: str, filename: str = "test.png") -> str:
        """Create test image with specified text"""
        img = Image.new('RGB', (400, 150), color='white')
        draw = ImageDraw.Draw(img)

        try:
            # Try to load a better font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Add text to image
        draw.text((20, 50), text, fill='black', font=font)

        file_path = os.path.join(self.temp_dir, filename)
        img.save(file_path)
        return file_path

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test complete processing pipeline"""
        # Create test message
        message = TelegramMessage(
            chat_id=123456789,
            message_id=1001,
            text="Check out this CloudWalk announcement",
            has_media=True
        )
        self.db.add(message)
        self.db.flush()

        # Create test image with brand mention
        image_path = self.create_test_image("CloudWalk Payment Solutions - Now Available!")

        # Create image record
        image_record = ImageRecord(
            message_id=message.id,
            file_path=image_path,
            sha256_hash="test_hash_123",
            file_size=os.path.getsize(image_path),
            processed=False
        )
        self.db.add(image_record)
        self.db.commit()

        # Initialize processor components
        processor = FraudProcessor()

        # Process the image
        await processor._process_single_image(image_record, self.db)

        # Verify OCR results
        ocr_results = self.db.query(OCRText).filter(
            OCRText.image_id == image_record.id
        ).all()

        assert len(ocr_results) == 1
        ocr_result = ocr_results[0]
        assert ocr_result.extracted_text is not None
        assert len(ocr_result.extracted_text) > 0
        assert "CloudWalk" in ocr_result.extracted_text or "cloudwalk" in ocr_result.extracted_text.lower()

        # Verify brand hits
        brand_hits = self.db.query(BrandHit).filter(
            BrandHit.image_id == image_record.id
        ).all()

        assert len(brand_hits) >= 1
        cloudwalk_hits = [hit for hit in brand_hits if hit.brand_name == 'CloudWalk']
        assert len(cloudwalk_hits) == 1

        cloudwalk_hit = cloudwalk_hits[0]
        assert cloudwalk_hit.confidence_score >= 85
        assert cloudwalk_hit.matched_text is not None

        # Verify image marked as processed
        assert image_record.processed == True

    def test_ocr_engine_accuracy(self):
        """Test OCR engine with various text scenarios"""
        ocr_engine = OCREngine()

        test_cases = [
            ("CloudWalk Payment System", "CloudWalk"),
            ("InfinitePay Mobile App", "InfinitePay"),
            ("Visa and Mastercard accepted", "Visa"),
            ("Contact support@cloudwalk.com", "CloudWalk"),
        ]

        for text, expected_brand in test_cases:
            image_path = self.create_test_image(text, f"test_{expected_brand.lower()}.png")

            extracted_text, confidence, processing_time = ocr_engine.extract_text(image_path)

            # Basic OCR validation
            assert confidence > 50, f"OCR confidence too low for '{text}': {confidence}%"
            assert len(extracted_text) > 0, f"No text extracted from '{text}'"
            assert processing_time > 0, "Processing time should be recorded"

            # Check if expected content is present (case-insensitive)
            assert expected_brand.lower() in extracted_text.lower(), \
                   f"Expected '{expected_brand}' in extracted text: '{extracted_text}'"

    def test_brand_matcher_accuracy(self):
        """Test brand matching with various scenarios"""
        matcher = BrandMatcher()

        test_cases = [
            # Exact matches
            ("CloudWalk payment system", "CloudWalk", 100),
            ("Visit InfinitePay website", "InfinitePay", 100),
            ("Visa card accepted", "Visa", 100),

            # Fuzzy matches
            ("CloudWlk systems", "CloudWalk", 85),
            ("Cloud Walk mobile", "CloudWalk", 85),
            ("InfinityPay app", "InfinitePay", 85),

            # Multiple brands
            ("CloudWalk and InfinitePay support Visa", ["CloudWalk", "InfinitePay", "Visa"], 90),
        ]

        for text, expected, min_confidence in test_cases:
            matches = matcher.find_matches(text)

            if isinstance(expected, str):
                expected = [expected]

            found_brands = [match['brand'] for match in matches]

            for expected_brand in expected:
                assert expected_brand in found_brands, \
                       f"Expected '{expected_brand}' in matches: {found_brands}"

                # Check confidence
                brand_match = next(m for m in matches if m['brand'] == expected_brand)
                assert brand_match['confidence'] >= min_confidence, \
                       f"Confidence too low for '{expected_brand}': {brand_match['confidence']}%"

    def test_false_positive_prevention(self):
        """Test that matcher doesn't create false positives"""
        matcher = BrandMatcher()

        false_positive_texts = [
            "I went for a walk in the clouds",  # Contains "walk" and "cloud"
            "The master card game was fun",     # Contains "master" and "card"
            "Infinite possibilities ahead",     # Contains "infinite"
            "Visual design is important",       # Contains "visa"-like text
            "Pay your bills online",           # Contains "pay"
        ]

        for text in false_positive_texts:
            matches = matcher.find_matches(text)
            high_confidence_matches = [m for m in matches if m['confidence'] >= 80]

            assert len(high_confidence_matches) == 0, \
                   f"False positive detected in '{text}': {high_confidence_matches}"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Create `tests/test_end_to_end.py`:

```python
import pytest
import asyncio
import subprocess
import time
import requests
import tempfile
import json
from PIL import Image, ImageDraw

class TestEndToEnd:

    @pytest.fixture(scope="class")
    def docker_environment(self):
        """Start test Docker environment"""
        # Start test containers
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.test.yml", "up", "-d"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            pytest.fail(f"Failed to start test environment: {result.stderr}")

        # Wait for services to be ready
        max_wait = 60
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Check database
                db_result = subprocess.run([
                    "docker-compose", "-f", "docker-compose.test.yml",
                    "exec", "-T", "db", "pg_isready", "-U", "fraud_user"
                ], capture_output=True, text=True)

                if db_result.returncode == 0:
                    break
            except:
                pass
            time.sleep(2)

        # Additional startup time for application
        time.sleep(10)

        yield

        # Cleanup
        subprocess.run([
            "docker-compose", "-f", "docker-compose.test.yml", "down", "-v"
        ])

    def create_brand_image(self, brand_name: str, context: str = "") -> str:
        """Create test image with brand mention"""
        img = Image.new('RGB', (600, 200), color='white')
        draw = ImageDraw.Draw(img)

        # Create realistic brand mention
        lines = [
            f"🎉 New Partnership with {brand_name}!",
            context or "Revolutionary payment technology",
            "Secure • Fast • Reliable"
        ]

        y_pos = 30
        for line in lines:
            draw.text((30, y_pos), line, fill='black')
            y_pos += 40

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name)
        return temp_file.name

    def test_system_health(self, docker_environment):
        """Test that all system components are healthy"""

        # Test database connectivity
        db_result = subprocess.run([
            "docker-compose", "-f", "docker-compose.test.yml",
            "exec", "-T", "db", "pg_isready", "-U", "fraud_user"
        ], capture_output=True, text=True)

        assert db_result.returncode == 0, "Database should be accessible"

        # Test application startup
        app_logs = subprocess.run([
            "docker-compose", "-f", "docker-compose.test.yml", "logs", "app"
        ], capture_output=True, text=True)

        assert app_logs.returncode == 0, "Should be able to get application logs"

        # Check for successful startup indicators
        log_content = app_logs.stdout
        success_indicators = [
            "Database initialized",
            "Components initialized",
            "Starting concurrent services"
        ]

        for indicator in success_indicators:
            assert indicator in log_content, f"Missing startup indicator: {indicator}"

    def test_fraud_detection_pipeline(self, docker_environment):
        """Test complete fraud detection pipeline with real image"""

        # Create test image with CloudWalk brand
        test_image = self.create_brand_image(
            "CloudWalk",
            "Leading fintech company in Brazil"
        )

        try:
            # Simulate image processing by inserting test data
            insert_sql = """
            INSERT INTO telegram_messages (chat_id, message_id, text, has_media)
            VALUES (999888777, 12345, 'Test fraud detection', true);

            INSERT INTO images (message_id, file_path, sha256_hash, file_size, processed)
            SELECT id, '/tmp/test_image.png', 'test_hash_e2e', 1024, false
            FROM telegram_messages WHERE message_id = 12345;
            """

            # Copy test image to container
            subprocess.run([
                "docker", "cp", test_image,
                "fraud-monitor-app-test:/tmp/test_image.png"
            ], check=True)

            # Insert test data
            subprocess.run([
                "docker-compose", "-f", "docker-compose.test.yml",
                "exec", "-T", "db", "psql", "-U", "fraud_user", "-d", "fraud_db", "-c", insert_sql
            ], check=True)

            # Wait for processing (up to 60 seconds)
            max_wait = 60
            start_time = time.time()
            brand_detected = False

            while time.time() - start_time < max_wait and not brand_detected:
                # Check for brand detection
                check_sql = "SELECT COUNT(*) FROM brand_hits WHERE brand_name = 'CloudWalk';"
                result = subprocess.run([
                    "docker-compose", "-f", "docker-compose.test.yml",
                    "exec", "-T", "db", "psql", "-U", "fraud_user", "-d", "fraud_db",
                    "-t", "-c", check_sql
                ], capture_output=True, text=True)

                if result.returncode == 0 and int(result.stdout.strip()) > 0:
                    brand_detected = True
                    break

                time.sleep(3)

            assert brand_detected, "CloudWalk brand should be detected within 60 seconds"

            # Verify processing details
            details_sql = """
            SELECT bh.brand_name, bh.confidence_score, bh.match_type, ot.confidence
            FROM brand_hits bh
            JOIN images i ON bh.image_id = i.id
            LEFT JOIN ocr_text ot ON ot.image_id = i.id
            WHERE bh.brand_name = 'CloudWalk';
            """

            details_result = subprocess.run([
                "docker-compose", "-f", "docker-compose.test.yml",
                "exec", "-T", "db", "psql", "-U", "fraud_user", "-d", "fraud_db",
                "-t", "-c", details_sql
            ], capture_output=True, text=True)

            assert details_result.returncode == 0, "Should be able to query processing details"

            # Verify OCR and matching worked
            details_output = details_result.stdout.strip()
            assert "CloudWalk" in details_output, "Brand hit should be recorded"

            print("✅ End-to-end fraud detection test passed!")

        finally:
            # Cleanup test image
            import os
            if os.path.exists(test_image):
                os.unlink(test_image)

    def test_multiple_brands_detection(self, docker_environment):
        """Test detection of multiple brands in single image"""

        # Create image with multiple brands
        test_image = self.create_brand_image(
            "CloudWalk",
            "Partnership with InfinitePay and Visa support"
        )

        try:
            # Similar setup as previous test but check for multiple brands
            # ... (implementation similar to above test)

            expected_brands = ["CloudWalk", "InfinitePay", "Visa"]

            # Wait and check for all brands
            max_wait = 60
            start_time = time.time()

            while time.time() - start_time < max_wait:
                check_sql = """
                SELECT DISTINCT brand_name FROM brand_hits
                WHERE brand_name IN ('CloudWalk', 'InfinitePay', 'Visa');
                """

                result = subprocess.run([
                    "docker-compose", "-f", "docker-compose.test.yml",
                    "exec", "-T", "db", "psql", "-U", "fraud_user", "-d", "fraud_db",
                    "-t", "-c", check_sql
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    detected_brands = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                    if len(detected_brands) >= 2:  # At least 2 brands detected
                        break

                time.sleep(3)

            # Verify at least primary brands were detected
            final_check = subprocess.run([
                "docker-compose", "-f", "docker-compose.test.yml",
                "exec", "-T", "db", "psql", "-U", "fraud_user", "-d", "fraud_db",
                "-t", "-c", check_sql
            ], capture_output=True, text=True)

            detected_brands = [line.strip() for line in final_check.stdout.strip().split('\n') if line.strip()]

            assert "CloudWalk" in detected_brands, "CloudWalk should be detected"
            print(f"✅ Detected brands: {detected_brands}")

        finally:
            # Cleanup
            import os
            if os.path.exists(test_image):
                os.unlink(test_image)

# Docker compose test configuration
docker_compose_test_content = """
version: '3.8'

services:
  app:
    build: .
    container_name: fraud-monitor-app-test
    environment:
      - DB_URL=postgresql+psycopg2://fraud_user:test_pass@db:5432/fraud_test
      - LOG_LEVEL=DEBUG
      - PROCESSING_INTERVAL=2
      - BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard
    volumes:
      - ./test_data:/app/data
    depends_on:
      db:
        condition: service_healthy
    command: ["python", "-c", "import time; time.sleep(3600)"]  # Keep container running

  db:
    image: postgres:15-alpine
    container_name: fraud-monitor-db-test
    environment:
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: test_pass
      POSTGRES_DB: fraud_test
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_user -d fraud_test"]
      interval: 5s
      timeout: 5s
      retries: 10
    tmpfs:
      - /var/lib/postgresql/data  # In-memory for faster tests
"""

# Save test compose file
with open("docker-compose.test.yml", "w") as f:
    f.write(docker_compose_test_content)
```

**Deliverables**:
- [ ] Comprehensive integration tests
- [ ] End-to-end pipeline testing
- [ ] Docker-based test environment
- [ ] Performance and accuracy validation
- [ ] False positive prevention tests

**Validation**: All tests pass with >90% success rate

#### Step 16: Test End-to-End System Functionality
**Estimated Time**: 2 hours
**Assignee**: QA/Lead Developer

Create final validation script `scripts/validate_system.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import tempfile
import os
from PIL import Image, ImageDraw
import subprocess
import time

class SystemValidator:
    def __init__(self):
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")

        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })

    def run_command(self, cmd: list, timeout: int = 30) -> tuple[bool, str]:
        """Run command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def test_docker_build(self) -> bool:
        """Test Docker image builds successfully"""
        print("🔨 Testing Docker build...")

        success, output = self.run_command(["docker-compose", "build"], timeout=300)
        self.log_test("Docker Build", success, output[-200:] if output else "")
        return success

    def test_services_startup(self) -> bool:
        """Test all services start successfully"""
        print("🚀 Testing service startup...")

        # Start services
        success, output = self.run_command(["docker-compose", "up", "-d"])
        if not success:
            self.log_test("Service Startup", False, output)
            return False

        # Wait for health checks
        time.sleep(30)

        # Check service status
        success, output = self.run_command(["docker-compose", "ps"])

        # Verify all services are running
        running_services = output.count("Up")
        expected_services = 2  # app + db

        if running_services >= expected_services:
            self.log_test("Service Startup", True, f"{running_services} services running")
            return True
        else:
            self.log_test("Service Startup", False, f"Only {running_services}/{expected_services} services running")
            return False

    def test_database_connectivity(self) -> bool:
        """Test database is accessible and schema is created"""
        print("💾 Testing database connectivity...")

        success, output = self.run_command([
            "docker-compose", "exec", "-T", "db",
            "psql", "-U", "fraud_user", "-d", "fraud_db",
            "-c", "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
        ])

        if success and "4" in output:  # Should have 4 tables
            self.log_test("Database Schema", True, "All tables created")
            return True
        else:
            self.log_test("Database Schema", False, output)
            return False

    def test_ocr_functionality(self) -> bool:
        """Test OCR engine works correctly"""
        print("👁️ Testing OCR functionality...")

        # Create test image
        test_image = self.create_test_image("CloudWalk Payment System")

        try:
            # Copy to container
            container_path = "/tmp/test_ocr.png"
            success, output = self.run_command([
                "docker", "cp", test_image, f"fraud-monitor-app:{container_path}"
            ])

            if not success:
                self.log_test("OCR Test", False, "Failed to copy test image")
                return False

            # Test OCR
            success, output = self.run_command([
                "docker-compose", "exec", "-T", "app",
                "python", "-c",
                f"from src.ocr_engine import OCREngine; "
                f"ocr = OCREngine(); "
                f"text, conf, time = ocr.extract_text('{container_path}'); "
                f"print(f'Text: {{text}} Confidence: {{conf}}%')"
            ])

            if success and "CloudWalk" in output and "Confidence:" in output:
                confidence = int(output.split("Confidence: ")[1].split("%")[0])
                self.log_test("OCR Functionality", True, f"Extracted text with {confidence}% confidence")
                return True
            else:
                self.log_test("OCR Functionality", False, output)
                return False

        finally:
            if os.path.exists(test_image):
                os.unlink(test_image)

    def test_brand_matching(self) -> bool:
        """Test brand matching works correctly"""
        print("🔍 Testing brand matching...")

        success, output = self.run_command([
            "docker-compose", "exec", "-T", "app",
            "python", "-c",
            "from src.brand_matcher import BrandMatcher; "
            "matcher = BrandMatcher(); "
            "matches = matcher.find_matches('CloudWalk payment system rocks!'); "
            "print(f'Matches: {len(matches)}'); "
            "[print(f'  - {m[\"brand\"]}: {m[\"confidence\"]}%') for m in matches]"
        ])

        if success and "CloudWalk" in output and "100%" in output:
            self.log_test("Brand Matching", True, "CloudWalk detected with 100% confidence")
            return True
        else:
            self.log_test("Brand Matching", False, output)
            return False

    def test_end_to_end_processing(self) -> bool:
        """Test complete processing pipeline"""
        print("🔄 Testing end-to-end processing...")

        # Create test image with brand
        test_image = self.create_test_image("🎉 New CloudWalk Partnership!")

        try:
            # Copy to container
            container_path = "/app/data/media/e2e_test.png"
            success, output = self.run_command([
                "docker", "cp", test_image, f"fraud-monitor-app:{container_path}"
            ])

            if not success:
                self.log_test("E2E Processing", False, "Failed to copy test image")
                return False

            # Insert test data to trigger processing
            insert_sql = f"""
            INSERT INTO telegram_messages (chat_id, message_id, text, has_media)
            VALUES (777888999, 99999, 'E2E test message', true);

            INSERT INTO images (message_id, file_path, sha256_hash, file_size, processed)
            VALUES ((SELECT id FROM telegram_messages WHERE message_id = 99999),
                   '{container_path}', 'e2e_test_hash', 1024, false);
            """

            success, output = self.run_command([
                "docker-compose", "exec", "-T", "db",
                "psql", "-U", "fraud_user", "-d", "fraud_db", "-c", insert_sql
            ])

            if not success:
                self.log_test("E2E Processing", False, "Failed to insert test data")
                return False

            # Wait for processing (up to 2 minutes)
            max_wait = 120
            start_time = time.time()

            while time.time() - start_time < max_wait:
                # Check for brand hit
                success, output = self.run_command([
                    "docker-compose", "exec", "-T", "db",
                    "psql", "-U", "fraud_user", "-d", "fraud_db", "-t", "-c",
                    "SELECT COUNT(*) FROM brand_hits WHERE brand_name = 'CloudWalk';"
                ])

                if success and int(output.strip()) > 0:
                    self.log_test("E2E Processing", True, "Brand hit detected and stored")
                    return True

                time.sleep(5)

            self.log_test("E2E Processing", False, "Brand hit not detected within timeout")
            return False

        finally:
            if os.path.exists(test_image):
                os.unlink(test_image)

    def create_test_image(self, text: str) -> str:
        """Create test image with specified text"""
        img = Image.new('RGB', (500, 150), color='white')
        draw = ImageDraw.Draw(img)

        # Add text
        draw.text((50, 60), text, fill='black')

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name)
        return temp_file.name

    def cleanup(self):
        """Clean up test environment"""
        print("🧹 Cleaning up test environment...")
        self.run_command(["docker-compose", "down", "-v"])

    def run_all_tests(self) -> bool:
        """Run complete validation suite"""
        print("=" * 60)
        print("🧪 FRAUD MONITOR - SYSTEM VALIDATION")
        print("=" * 60)

        tests = [
            self.test_docker_build,
            self.test_services_startup,
            self.test_database_connectivity,
            self.test_ocr_functionality,
            self.test_brand_matching,
            self.test_end_to_end_processing
        ]

        passed_tests = 0

        try:
            for test in tests:
                if test():
                    passed_tests += 1
                print()  # Empty line between tests

        finally:
            self.cleanup()

        # Print summary
        print("=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100

        for result in self.test_results:
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {result['test']}")

        print(f"\n📈 Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

        if success_rate >= 90:
            print("🎉 SYSTEM VALIDATION PASSED!")
            print("   Ready for production deployment")
        else:
            print("⚠️ SYSTEM VALIDATION FAILED")
            print("   Please address failing tests before deployment")

        return success_rate >= 90

def main():
    validator = SystemValidator()
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

Create `scripts/quick_test.py`:

```python
#!/usr/bin/env python3
"""
Quick smoke test for development
"""
import subprocess
import sys
import tempfile
from PIL import Image, ImageDraw

def create_test_image():
    """Create simple test image"""
    img = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "CloudWalk Test", fill='black')

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(temp_file.name)
    return temp_file.name

def main():
    print("🚀 Quick Smoke Test")

    # Test Python imports
    try:
        print("Testing imports...")
        subprocess.run([
            "python", "-c",
            "from src.config import *; "
            "from src.ocr_engine import OCREngine; "
            "from src.brand_matcher import BrandMatcher; "
            "print('✅ All imports successful')"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Import test failed")
        return False

    # Test OCR
    try:
        print("Testing OCR...")
        test_image = create_test_image()

        result = subprocess.run([
            "python", "-c",
            f"from src.ocr_engine import OCREngine; "
            f"ocr = OCREngine(); "
            f"text, conf, time = ocr.extract_text('{test_image}'); "
            f"print('OCR Result:', text, 'Confidence:', conf)"
        ], capture_output=True, text=True, check=True)

        if "CloudWalk" in result.stdout:
            print("✅ OCR test passed")
        else:
            print("⚠️ OCR test unclear:", result.stdout)

    except subprocess.CalledProcessError as e:
        print("❌ OCR test failed:", e.stderr)
        return False

    # Test brand matching
    try:
        print("Testing brand matching...")
        subprocess.run([
            "python", "-c",
            "from src.brand_matcher import BrandMatcher; "
            "matcher = BrandMatcher(); "
            "matches = matcher.find_matches('CloudWalk payment system'); "
            "print('Matches:', [m['brand'] for m in matches]); "
            "assert 'CloudWalk' in [m['brand'] for m in matches]"
        ], check=True)
        print("✅ Brand matching test passed")

    except subprocess.CalledProcessError:
        print("❌ Brand matching test failed")
        return False

    print("🎉 Quick smoke test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Deliverables**:
- [ ] Complete system validation suite
- [ ] End-to-end pipeline testing
- [ ] Quick smoke test for development
- [ ] Performance benchmarking
- [ ] Production readiness checklist

**Validation**: System passes all validation tests with >90% success rate

## Summary and Next Steps

This comprehensive implementation plan provides:

- **16 detailed implementation steps** across 7 phases
- **Estimated 14-18 days** total implementation time
- **Clear dependencies** and validation criteria for each step
- **Production-ready deliverables** including Docker deployment
- **Comprehensive testing strategy** from unit to end-to-end tests

### Recommended Execution Order

1. **Week 1**: Phases 1-3 (Foundation + Core Data + Processing Engines)
2. **Week 2**: Phases 4-5 (Telegram Integration + Application)
3. **Week 3**: Phases 6-7 (Deployment + Testing + Production Hardening)

### Success Metrics

- ✅ All components pass individual health checks
- ✅ End-to-end processing completes in <30 seconds
- ✅ OCR accuracy >90% on test images
- ✅ Brand detection precision >95%
- ✅ System handles 1000+ messages/day without issues
- ✅ Docker deployment completes in <2 hours

The plan ensures we build a robust, production-ready fraud monitoring system that meets all specified requirements while maintaining clear upgrade paths for future enhancements.