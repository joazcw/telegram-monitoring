# Configuration Reference

Complete reference for all environment variables and configuration options.

## .env.example

```bash
# Telegram API Configuration
# Get from https://my.telegram.org/apps
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

# Session and Groups
TELEGRAM_SESSION_PATH=./data/telethon.session
TELEGRAM_GROUPS=@suspicious_group,@another_group,-1001234567890
TELEGRAM_ALERT_CHAT_ID=-1001987654321

# Database
DB_URL=postgresql+psycopg2://fraud_user:secure_password@db:5432/fraud_db

# File Storage
MEDIA_DIR=./data/media
MAX_MEDIA_SIZE=10485760

# OCR Configuration
OCR_LANG=eng
OCR_CONFIDENCE_THRESHOLD=30
OCR_TIMEOUT=10

# Brand Matching
BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard,PayPal,Stripe
FUZZY_THRESHOLD=85
MATCH_MIN_LENGTH=3

# Processing
BATCH_SIZE=10
PROCESSING_INTERVAL=5
RETRY_MAX_ATTEMPTS=3
RETRY_BACKOFF_FACTOR=2

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/app.log
LOG_MAX_SIZE=104857600
LOG_BACKUP_COUNT=5

# Security
SESSION_ENCRYPTION_KEY=your-32-char-encryption-key-here
ALERT_RATE_LIMIT=10
API_TIMEOUT=30
```

## Core Configuration

### Telegram API Settings

#### TELEGRAM_API_ID
- **Type**: Integer
- **Required**: Yes
- **Description**: API ID from my.telegram.org
- **Example**: `12345678`
- **How to get**:
  1. Go to https://my.telegram.org/apps
  2. Create new application
  3. Note the "App api_id"

#### TELEGRAM_API_HASH
- **Type**: String (32 characters)
- **Required**: Yes
- **Description**: API Hash from my.telegram.org
- **Example**: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
- **Security**: Keep secret, never commit to git
- **How to get**: From same page as API_ID, "App api_hash" field

#### TELEGRAM_SESSION_PATH
- **Type**: File path
- **Required**: No
- **Default**: `./data/telethon.session`
- **Description**: Where to store Telegram session file
- **Example**: `/app/data/sessions/fraud_monitor.session`
- **Notes**:
  - File created on first authentication
  - Backup this file to avoid re-authentication
  - Protect with 600 permissions

#### TELEGRAM_GROUPS
- **Type**: Comma-separated list
- **Required**: Yes
- **Description**: Groups/channels to monitor
- **Example**: `@publicgroup,@anotherchannel,-1001234567890`
- **Formats supported**:
  - Public username: `@groupname`
  - Private group ID: `-1001234567890`
  - Channel ID: `-1001987654321`
- **How to get group IDs**:
  ```bash
  # Method 1: Forward message to @userinfobot
  # Method 2: Use Telegram desktop, right-click group
  # Method 3: Check browser URL: t.me/c/1234567890
  ```

#### TELEGRAM_ALERT_CHAT_ID
- **Type**: Integer (chat ID)
- **Required**: Yes
- **Description**: Where to send fraud alerts
- **Example**: `-1001987654321`
- **Setup**:
  1. Create group/channel for alerts
  2. Add your bot to the group
  3. Get chat ID using methods above

### Database Configuration

#### DB_URL
- **Type**: Database URL
- **Required**: Yes
- **Description**: PostgreSQL connection string
- **Format**: `postgresql+psycopg2://user:password@host:port/database`
- **Example**: `postgresql+psycopg2://fraud_user:secure_pass@localhost:5432/fraud_db`
- **SSL Options**:
  ```bash
  # Require SSL
  postgresql+psycopg2://user:pass@host:5432/db?sslmode=require

  # With client certificates
  postgresql+psycopg2://user:pass@host:5432/db?sslmode=require&sslcert=client.crt&sslkey=client.key&sslrootcert=ca.crt
  ```

### Storage Configuration

#### MEDIA_DIR
- **Type**: Directory path
- **Required**: No
- **Default**: `./data/media`
- **Description**: Where to store downloaded images
- **Example**: `/app/data/media`
- **Notes**:
  - Directory created automatically
  - Ensure sufficient disk space
  - Consider mounting external volume in production

#### MAX_MEDIA_SIZE
- **Type**: Integer (bytes)
- **Required**: No
- **Default**: `10485760` (10MB)
- **Description**: Maximum file size for media download
- **Example**: `52428800` (50MB)
- **Notes**: Larger files skipped to prevent disk exhaustion

## OCR Configuration

#### OCR_LANG
- **Type**: Language code(s)
- **Required**: No
- **Default**: `eng`
- **Description**: Tesseract language(s) for OCR
- **Examples**:
  - Single: `eng`
  - Multiple: `eng+por+spa`
  - Portuguese: `por`
  - Spanish: `spa`
- **Available languages**: Run `tesseract --list-langs`
- **Installation**:
  ```bash
  # Install additional language packs
  sudo apt-get install tesseract-ocr-por tesseract-ocr-spa
  ```

#### OCR_CONFIDENCE_THRESHOLD
- **Type**: Integer (0-100)
- **Required**: No
- **Default**: `30`
- **Description**: Minimum confidence to include OCR text
- **Example**: `50` (higher = fewer false positives)
- **Tuning**:
  - Lower values: More text captured, more noise
  - Higher values: Less text captured, higher quality

#### OCR_TIMEOUT
- **Type**: Integer (seconds)
- **Required**: No
- **Default**: `10`
- **Description**: Maximum time for OCR processing per image
- **Example**: `15`
- **Notes**: Images timing out are marked as failed

## Brand Detection Configuration

#### BRAND_KEYWORDS
- **Type**: Comma-separated list
- **Required**: Yes
- **Description**: Brands/keywords to detect
- **Example**: `CloudWalk,InfinitePay,Visa,Mastercard,PayPal,Stripe,Pix`
- **Best practices**:
  - Use exact brand names
  - Include common misspellings if needed
  - Avoid generic terms
  - Test with fuzzy matching threshold

#### FUZZY_THRESHOLD
- **Type**: Integer (0-100)
- **Required**: No
- **Default**: `85`
- **Description**: Minimum similarity score for fuzzy matching
- **Example**: `90` (stricter matching)
- **Tuning guide**:
  - `95-100`: Only exact matches and minor typos
  - `85-94`: Reasonable typo tolerance
  - `70-84`: Loose matching, more false positives
  - `<70`: Very loose, high false positive rate

#### MATCH_MIN_LENGTH
- **Type**: Integer
- **Required**: No
- **Default**: `3`
- **Description**: Minimum word length for fuzzy matching
- **Example**: `4`
- **Notes**: Prevents matching very short words (like "it", "to")

## Processing Configuration

#### BATCH_SIZE
- **Type**: Integer
- **Required**: No
- **Default**: `10`
- **Description**: How many images to process per batch
- **Example**: `20`
- **Performance impact**:
  - Higher: Better throughput, more memory usage
  - Lower: Better latency, less memory usage

#### PROCESSING_INTERVAL
- **Type**: Integer (seconds)
- **Required**: No
- **Default**: `5`
- **Description**: How often to check for new work
- **Example**: `2` (more responsive)
- **Notes**: Balance between responsiveness and resource usage

#### RETRY_MAX_ATTEMPTS
- **Type**: Integer
- **Required**: No
- **Default**: `3`
- **Description**: Maximum retries for failed operations
- **Example**: `5`
- **Applies to**: OCR failures, API calls, database operations

#### RETRY_BACKOFF_FACTOR
- **Type**: Float
- **Required**: No
- **Default**: `2.0`
- **Description**: Exponential backoff multiplier
- **Example**: `1.5`
- **Behavior**: Delay = base_delay * (factor ^ attempt_number)

## Logging Configuration

#### LOG_LEVEL
- **Type**: String
- **Required**: No
- **Default**: `INFO`
- **Description**: Minimum log level to output
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Example**: `DEBUG` (verbose logging)
- **Production recommendation**: `INFO` or `WARNING`

#### LOG_FILE
- **Type**: File path
- **Required**: No
- **Default**: `./data/logs/app.log`
- **Description**: Where to write log files
- **Example**: `/var/log/fraud-monitor/app.log`
- **Notes**: Directory must exist and be writable

#### LOG_MAX_SIZE
- **Type**: Integer (bytes)
- **Required**: No
- **Default**: `104857600` (100MB)
- **Description**: Maximum log file size before rotation
- **Example**: `52428800` (50MB)

#### LOG_BACKUP_COUNT
- **Type**: Integer
- **Required**: No
- **Default**: `5`
- **Description**: Number of backup log files to keep
- **Example**: `10`
- **Result**: Keeps app.log, app.log.1, app.log.2, ..., app.log.N

## Security Configuration

#### SESSION_ENCRYPTION_KEY
- **Type**: String (32 characters)
- **Required**: No (but recommended)
- **Default**: None (session files unencrypted)
- **Description**: Key for encrypting session files
- **Example**: `your-32-char-encryption-key-here`
- **Generation**:
  ```bash
  # Generate secure key
  openssl rand -hex 16
  # Or
  python -c "import secrets; print(secrets.token_hex(16))"
  ```

#### ALERT_RATE_LIMIT
- **Type**: Integer (per minute)
- **Required**: No
- **Default**: `10`
- **Description**: Maximum alerts per minute to prevent spam
- **Example**: `20`
- **Behavior**: Additional alerts queued or dropped

#### API_TIMEOUT
- **Type**: Integer (seconds)
- **Required**: No
- **Default**: `30`
- **Description**: Timeout for external API calls
- **Example**: `60`
- **Applies to**: Telegram API, alert delivery

## Environment-Specific Configurations

### Development (.env.dev)
```bash
LOG_LEVEL=DEBUG
PROCESSING_INTERVAL=1
BATCH_SIZE=5
FUZZY_THRESHOLD=80
OCR_CONFIDENCE_THRESHOLD=20
RETRY_MAX_ATTEMPTS=1
```

### Production (.env.prod)
```bash
LOG_LEVEL=WARNING
PROCESSING_INTERVAL=5
BATCH_SIZE=20
FUZZY_THRESHOLD=90
OCR_CONFIDENCE_THRESHOLD=40
RETRY_MAX_ATTEMPTS=5
SESSION_ENCRYPTION_KEY=generate-secure-32-char-key
ALERT_RATE_LIMIT=20
```

### Testing (.env.test)
```bash
DB_URL=postgresql+psycopg2://test_user:test_pass@localhost:5432/fraud_test
MEDIA_DIR=./test_data/media
LOG_LEVEL=DEBUG
PROCESSING_INTERVAL=1
BATCH_SIZE=1
RETRY_MAX_ATTEMPTS=1
```

## Docker Environment Variables

### Docker Compose Override
```yaml
# docker-compose.override.yml for local development
version: '3.8'
services:
  app:
    environment:
      - LOG_LEVEL=DEBUG
      - PROCESSING_INTERVAL=2
    volumes:
      - ./dev_data:/app/data  # Separate dev data
```

### Production Docker Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    env_file:
      - .env.prod
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped

  db:
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

secrets:
  db_password:
    external: true
```

## Configuration Validation

### Startup Validation Script
```python
# scripts/validate_config.py
import os
import sys
from src.config import *

def validate_config():
    """Validate all required configuration is present and valid"""
    errors = []

    # Required fields
    if not TELEGRAM_API_ID:
        errors.append("TELEGRAM_API_ID is required")

    if not TELEGRAM_API_HASH:
        errors.append("TELEGRAM_API_HASH is required")

    if not TELEGRAM_GROUPS:
        errors.append("TELEGRAM_GROUPS is required")

    # Validate ranges
    if not (0 <= FUZZY_THRESHOLD <= 100):
        errors.append("FUZZY_THRESHOLD must be 0-100")

    if not (0 <= OCR_CONFIDENCE_THRESHOLD <= 100):
        errors.append("OCR_CONFIDENCE_THRESHOLD must be 0-100")

    # Validate paths
    media_parent = os.path.dirname(MEDIA_DIR)
    if not os.access(media_parent, os.W_OK):
        errors.append(f"MEDIA_DIR parent {media_parent} not writable")

    # Validate database URL
    if not DB_URL.startswith('postgresql'):
        errors.append("DB_URL must be PostgreSQL connection string")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("✅ Configuration validation passed")

if __name__ == "__main__":
    validate_config()
```

### Runtime Health Check
```python
# src/health_check.py
def health_check():
    """Runtime configuration and system health check"""
    checks = {
        'database': check_database_connection(),
        'telegram_auth': check_telegram_session(),
        'ocr_engine': check_tesseract_installed(),
        'disk_space': check_disk_space(MEDIA_DIR),
        'log_directory': check_log_directory_writable(),
    }

    healthy = all(checks.values())

    return {
        'status': 'healthy' if healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }
```