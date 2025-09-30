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
# Edit .env with your API credentials first (see below)
# Don't worry about group IDs yet - we'll discover them in Step 3
```

### 3. Authenticate and Configure Groups

**⚠️ Critical**: You MUST authenticate first and use real groups you can access!

```bash
# Step 1: Authenticate with Telegram
python scripts/auth.py
# Enter your phone number and verification code when prompted

# Step 2: Discover your accessible groups
python debug_messages.py
# This will show a formatted table with:
# - Group IDs (copy these to .env)
# - Group names and types
# - Example .env configuration

# Step 3: Update .env with the IDs from above
# Choose groups you want to MONITOR for suspicious content:
TELEGRAM_GROUPS=-1001234567890,-1002345678901

# Choose where you want to RECEIVE alerts (can be same or different):
TELEGRAM_ALERT_CHAT_ID=-1003456789012
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

## Understanding Groups vs Alerts

**Key Concept**: You need to understand the difference between monitoring and alerting:

- **`TELEGRAM_GROUPS`**: Groups you want to **watch for suspicious messages** (input)
- **`TELEGRAM_ALERT_CHAT_ID`**: Where you want to **receive notifications** (output)

**Examples:**
```
✅ Good Setup (Different IDs):
TELEGRAM_GROUPS=-1001234567890        # Monitor public crypto group
TELEGRAM_ALERT_CHAT_ID=-1003456789012 # Send alerts to private team chat

❌ Problematic Setup (Same ID):
TELEGRAM_GROUPS=-1001234567890        # Monitor group
TELEGRAM_ALERT_CHAT_ID=-1001234567890 # Alert same group (may cause issues)
```

## Environment Configuration

Copy to `.env` and edit marked fields:

```bash
# EDIT ME: Get from https://my.telegram.org/apps
TELEGRAM_API_ID=12345
TELEGRAM_API_HASH=your_api_hash_here

TELEGRAM_SESSION_PATH=./data/telethon.session

# EDIT ME: Group/channel IDs to monitor (get from Step 3 above)
# Use NUMERIC IDs (recommended) or usernames - comma-separated for multiple
TELEGRAM_GROUPS=-1001234567890
# Alternative username format: TELEGRAM_GROUPS=@your_group_name

# EDIT ME: Chat ID where alerts are sent (get from Step 3 above)
# IMPORTANT: Use different ID from monitoring groups for best results
TELEGRAM_ALERT_CHAT_ID=-1003456789012

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

## Complete Setup Walkthrough

**If you followed the steps above, skip this section.** This is for reference:

```bash
# 1. Generate encryption key and add to .env
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Authenticate with Telegram
python scripts/auth.py

# 3. Find your groups (use the script from Step 2 in the main setup)
# Copy the IDs and update your .env file

# 4. Update .env with real IDs, then start
docker-compose up -d
```

## Troubleshooting

- **"Session file not found"**: Run `python scripts/auth.py` to authenticate
- **"UsernameInvalidError" or "Invalid Peer"**: You're using example/invalid group IDs. Run `python debug_messages.py` to get real group IDs you can access
- **"No accessible groups"**: Run `python debug_messages.py` to discover groups you can monitor
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