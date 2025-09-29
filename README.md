# Fraud Monitoring Telegram Bot

Real-time fraud monitoring system that ingests Telegram group messages, performs OCR on images for brand mentions, and sends instant alerts. Built for rapid deployment with minimal infrastructure.

## Quick Start (3-5 minutes)

### Prerequisites
- Docker & Docker Compose
- Telegram account
- 5 minutes to obtain Telegram API credentials

### 1. Get Telegram API Credentials

1. Visit https://my.telegram.org/apps
2. Create new application, note `API ID` and `API Hash`
3. Create a bot via @BotFather, get bot token for alerts

### 2. Setup Environment

```bash
git clone <your-repo>
cd fraud-monitor
cp .env.example .env
# Edit .env with your credentials (see below)
```

### 3. Configure Groups and Alerts

```bash
# Find group IDs using Telegram desktop:
# 1. Add bot to groups you want to monitor
# 2. Forward message from group to @userinfobot
# 3. Use the chat_id in TELEGRAM_GROUPS

# For alert chat: create group with your bot, get ID same way
```

### 4. Launch System

```bash
docker-compose up -d
docker-compose logs -f app  # Watch for "Connected to Telegram" message
```

### 5. Quick Smoke Test

1. Send image containing "CloudWalk" text to monitored group
2. Check alert chat for notification within 30 seconds
3. Verify in logs: `Brand hit detected: CloudWalk`

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

DB_URL=postgresql+psycopg2://fraud_user:fraud_pass@db:5432/fraud_db
MEDIA_DIR=./data/media
OCR_LANG=eng
# EDIT ME: Brands/keywords to detect (comma-separated)
BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard
FUZZY_THRESHOLD=85
LOG_LEVEL=INFO
```

## First Run Authentication

On first startup, you'll need to authenticate with Telegram:

```bash
docker-compose logs app
# Look for: "Please enter your phone (or bot token):"
docker-compose exec app python -c "
import asyncio
from src.auth import authenticate_session
asyncio.run(authenticate_session())
"
# Enter phone number and verification code when prompted
```

## Troubleshooting

- **"Session file not found"**: Normal on first run, follow authentication steps above
- **"No brand hits detected"**: Check OCR_LANG matches image text language
- **"Rate limit exceeded"**: Telegram API limits, wait 60 seconds
- **"Database connection failed"**: Ensure `docker-compose up db` started successfully

## Architecture

```
Telegram Groups → Telethon Client → OCR Engine → Brand Matcher → Alert Bot
                        ↓                ↓            ↓
                   PostgreSQL    Local Disk    Fuzzy Matching
```

For detailed implementation, see `docs/Implementation-Guide.md`.