import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Telegram Configuration
TELEGRAM_API_ID = int(os.getenv('TELEGRAM_API_ID', '0'))
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH', '')
TELEGRAM_SESSION_PATH = os.getenv('TELEGRAM_SESSION_PATH', './data/telethon.session')
def _parse_groups(groups_str: str) -> List:
    """Parse groups string, converting numeric IDs to integers and keeping usernames as strings"""
    groups = []
    for g in groups_str.split(','):
        g = g.strip()
        if not g:
            continue
        # If it's a numeric ID (starts with - and rest are digits), convert to int
        if g.startswith('-') and g[1:].isdigit():
            groups.append(int(g))
        # If it starts with @ or contains letters, keep as string (username)
        else:
            groups.append(g)
    return groups

TELEGRAM_GROUPS = _parse_groups(os.getenv('TELEGRAM_GROUPS', ''))
TELEGRAM_ALERT_CHAT_ID = int(os.getenv('TELEGRAM_ALERT_CHAT_ID', '0'))

# Database Configuration
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg://fraud_user:fraud_pass@localhost:5432/fraud_db')

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
LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', '104857600'))  # 100MB
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

# Security
SESSION_ENCRYPTION_KEY = os.getenv('SESSION_ENCRYPTION_KEY', '')
ALERT_RATE_LIMIT = int(os.getenv('ALERT_RATE_LIMIT', '10'))
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))


def validate_config():
    """Validate required configuration values."""
    errors = []

    if not TELEGRAM_API_ID or TELEGRAM_API_ID == 0:
        errors.append("TELEGRAM_API_ID is required")

    if not TELEGRAM_API_HASH:
        errors.append("TELEGRAM_API_HASH is required")

    if not TELEGRAM_GROUPS:
        errors.append("TELEGRAM_GROUPS is required")

    if not TELEGRAM_ALERT_CHAT_ID or TELEGRAM_ALERT_CHAT_ID == 0:
        errors.append("TELEGRAM_ALERT_CHAT_ID is required")

    if not SESSION_ENCRYPTION_KEY:
        errors.append("SESSION_ENCRYPTION_KEY is required")

    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

    return True


def get_config_summary():
    """Return configuration summary for debugging."""
    return {
        'telegram_api_configured': bool(TELEGRAM_API_ID and TELEGRAM_API_HASH),
        'groups_count': len(TELEGRAM_GROUPS),
        'brand_keywords_count': len(BRAND_KEYWORDS),
        'media_dir': MEDIA_DIR,
        'db_url_configured': bool(DB_URL and not DB_URL.endswith('localhost:5432/fraud_db')),
        'log_level': LOG_LEVEL,
        'ocr_lang': OCR_LANG,
        'fuzzy_threshold': FUZZY_THRESHOLD
    }