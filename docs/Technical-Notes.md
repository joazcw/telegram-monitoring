# Technical Notes

Basic technical information for developers working on the Fraud Monitoring System.

## Architecture

Simple 2-service architecture:

```
┌─────────────────┐    ┌──────────────────┐
│   Application   │───▶│   PostgreSQL     │
│   (Python)      │    │   Database       │
└─────────────────┘    └──────────────────┘
```

## Core Components

### 1. Message Collection (`telegram_collector.py`)
- Monitors Telegram groups using Telethon
- Stores messages and downloads images
- Handles authentication and group validation

### 2. OCR Processing (`ocr_engine.py`)
- Extracts text from images using Tesseract
- Preprocesses images with OpenCV
- Returns confidence scores

### 3. Brand Detection (`brand_matcher.py`)
- Matches text against brand keywords
- Uses RapidFuzz for fuzzy matching (85% threshold)
- Supports exact and approximate matching

### 4. Alert System (`alert_sender.py`)
- Sends formatted alerts via Telegram
- Includes rate limiting (10 alerts/window)
- HTML-formatted messages with context

### 5. Processing Pipeline (`processor.py`)
- Orchestrates the complete fraud detection flow
- Manages database transactions
- Handles error recovery

## Database Schema

Four simple tables:
- `telegram_messages` - Message storage
- `images` - Image file records
- `ocr_text` - OCR results
- `brand_hits` - Detection matches

## Development Setup

1. **Clone and setup:**
   ```bash
   git clone <repo>
   cd t
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure:**
   ```bash
   cp .env.example .env
   # Edit .env with your Telegram credentials
   ```

3. **Authenticate:**
   ```bash
   python scripts/auth.py
   ```

4. **Find groups:**
   ```bash
   python debug_messages.py
   ```

5. **Run:**
   ```bash
   python -m src.main
   ```

## Testing

Run basic tests:
```bash
python run_tests.py
```

Or use pytest directly:
```bash
pytest tests/ -v
```

## Docker Deployment

Simple deployment:
```bash
./scripts/deploy.sh
```

This builds images, initializes the database, and starts both services.

## Key Technologies

- **Python 3.11+** - Main language
- **Telethon** - Telegram API client
- **PostgreSQL** - Database
- **Tesseract + OpenCV** - OCR processing
- **RapidFuzz** - Fuzzy string matching
- **SQLAlchemy** - ORM
- **Docker** - Containerization

## Performance Notes

- **OCR Processing**: ~1-3 seconds per image
- **Brand Detection**: Sub-second for text processing
- **Alert Delivery**: ~2-5 seconds end-to-end
- **Database**: Handles ~1000s of messages efficiently

## Common Issues

1. **Group Access**: Ensure you're a member of monitored groups
2. **OCR Quality**: Clear, high-contrast images work best
3. **Rate Limits**: System includes built-in rate limiting
4. **Memory**: OCR processing is memory-intensive for large images

## File Structure

```
├── src/           # Main application
├── scripts/       # Setup and utilities
├── tests/         # Basic test suite
├── docs/          # Documentation
├── data/          # Runtime data (created automatically)
└── .env           # Configuration
```