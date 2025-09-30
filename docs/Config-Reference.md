# Configuration Reference

Essential configuration options for the Fraud Monitoring System.

## Required Settings

### Telegram API Credentials
Get these from https://my.telegram.org/apps

```bash
TELEGRAM_API_ID=12345                    # Your API ID
TELEGRAM_API_HASH=your_api_hash_here     # Your API Hash
```

### Groups and Alerts
```bash
TELEGRAM_GROUPS=-1001234567890           # Group IDs to monitor (comma-separated)
TELEGRAM_ALERT_CHAT_ID=-4034221236       # Where to send alerts
```

### Brand Detection
```bash
BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard   # Brands to detect
```

## Optional Settings

### OCR Configuration
```bash
OCR_LANG=eng                    # Tesseract language (default: eng)
OCR_CONFIDENCE_THRESHOLD=30     # Minimum OCR confidence (default: 30)
```

### Brand Matching
```bash
FUZZY_THRESHOLD=85              # Fuzzy matching threshold (default: 85)
MATCH_MIN_LENGTH=3              # Minimum match length (default: 3)
```

### Logging
```bash
LOG_LEVEL=INFO                  # Logging level (default: INFO)
LOG_FILE=./data/logs/app.log    # Log file path
```

### Database (Docker handles this)
```bash
DB_URL=postgresql+psycopg://fraud_user:fraud_pass@db:5432/fraud_db
```

## Getting Group IDs

Run this command to find group IDs you can access:

```bash
python debug_messages.py
```

This will show all groups with their IDs that you can monitor.

## Configuration Validation

The system validates configuration automatically at startup. If you have issues, check:

1. **Telegram credentials** - Valid API ID and Hash from my.telegram.org
2. **Group access** - Make sure you're a member of groups you want to monitor
3. **Chat permissions** - Alert chat must allow your account to send messages
4. **File paths** - Data directories must be writable

## Example Complete .env File

```bash
# Telegram API (required)
TELEGRAM_API_ID=21446602
TELEGRAM_API_HASH=bfc7b9df75183995854ec246c2240a17

# Groups (required)
TELEGRAM_GROUPS=-1002988913630
TELEGRAM_ALERT_CHAT_ID=-4034221236

# Brands to detect (required)
BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard

# Optional settings (defaults shown)
OCR_LANG=eng
OCR_CONFIDENCE_THRESHOLD=30
FUZZY_THRESHOLD=85
LOG_LEVEL=INFO

# Database (handled by Docker)
DB_URL=postgresql+psycopg://fraud_user:fraud_pass@db:5432/fraud_db
```