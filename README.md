# Fraud Monitoring System

Real-time fraud monitoring system that ingests Telegram group messages using user authentication, performs OCR on images for brand mentions, and sends instant alerts. Built for rapid deployment with minimal infrastructure.

## Quick Start (3-5 minutes)

### Prerequisites
- Docker & Docker Compose
- Telegram account
- 5 minutes to obtain Telegram API credentials

### 1. Get Telegram API Credentials

1. Visit https://my.telegram.org/apps
2. Create new application, note `API ID` and `API Hash`
3. Generate session encryption key (see below)

### 2. Setup Environment

```bash
git clone <your-repo>
cd t
cp .env.example .env
# Edit .env with your credentials (see below)
```

### 3. Configure Groups and Alerts

```bash
# Find accessible groups:
python debug_messages.py  # Shows groups you can access

# Use group usernames (preferred) or numeric IDs:
# - Username: @groupname
# - Numeric ID: -1001234567890

# For alert chat: use any chat/group where you want to receive alerts
```

### 4. Launch System

```bash
docker-compose up -d
docker-compose logs -f app  # Watch for "Connected to Telegram" message
```

### 5. Quick Smoke Test

1. **Text Test**: Send message with "CloudWalk" to monitored group
   - Should receive alert within 5-10 seconds

2. **Image Test**: Send clear image with "Visa" or "Mastercard" text
   - Use high-contrast, readable text (black on white background works best)
   - Should receive alert within 30 seconds

3. **Verify in logs**:
   ```bash
   docker-compose logs -f app
   # Look for: "Brand hit in text: CloudWalk (confidence: 100%)"
   # Or: "Alert sent for brand: Visa"
   ```

## Environment Configuration

Copy to `.env` and edit marked fields:

```bash
# EDIT ME: Get from https://my.telegram.org/apps
TELEGRAM_API_ID=12345
TELEGRAM_API_HASH=your_api_hash_here

TELEGRAM_SESSION_PATH=./data/telethon.session
# EDIT ME: Group usernames or IDs (comma-separated)
TELEGRAM_GROUPS=@suspicious_group,@another_group
# EDIT ME: Chat ID where alerts are sent
TELEGRAM_ALERT_CHAT_ID=-123456789

# EDIT ME: Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
SESSION_ENCRYPTION_KEY=your_encryption_key_here

DB_URL=postgresql+psycopg://fraud_user:fraud_pass@db:5432/fraud_db
MEDIA_DIR=./data/media
OCR_LANG=eng
# EDIT ME: Brands/keywords to detect (comma-separated)
BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard
FUZZY_THRESHOLD=85
LOG_LEVEL=INFO
```

## First Run Authentication

Authenticate with your Telegram account before first run:

```bash
# Generate encryption key
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Add the key to .env as SESSION_ENCRYPTION_KEY

# Authenticate your account
python scripts/auth.py
# Enter phone number and verification code when prompted

# Then start the system
docker-compose up -d
```

## Troubleshooting

- **"Session file not found"**: Run `python scripts/auth.py` to authenticate
- **"No brand hits detected"**: Check image has clear, readable text with target keywords
- **"Database is locked"**: Fixed in latest version - uses shared session
- **"Cannot find entity"**: Use group usernames (@group) instead of numeric IDs
- **"Database connection failed"**: Ensure PostgreSQL container is healthy

## Architecture

```
Telegram Groups → User Session → Message Collector → OCR Engine → Brand Matcher → Alert Sender
                       ↓               ↓              ↓            ↓             ↓
                  Authentication   PostgreSQL    Local Disk   Fuzzy Search   User Session
```

For detailed implementation, see `docs/Implementation-Guide.md`.