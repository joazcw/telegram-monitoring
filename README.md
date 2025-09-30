# Fraud Monitoring System

Real-time fraud monitoring that watches Telegram groups for suspicious messages, performs OCR on images, and sends instant alerts when target brands are mentioned.

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose
- Telegram account
- 5 minutes for setup

### Step 1: Get Telegram API Credentials

1. Visit https://my.telegram.org/apps
2. Create new application
3. Note your `API ID` and `API Hash`

### Step 2: Clone and Configure

```bash
git clone <your-repo>
cd t
cp .env.example .env
```

Edit `.env` and add your credentials:
```bash
TELEGRAM_API_ID=12345                    # From step 1
TELEGRAM_API_HASH=your_api_hash_here     # From step 1
```

### Step 3: Authenticate with Telegram

```bash
python scripts/auth.py
```
Enter your phone number and verification code when prompted.

### Step 4: Find Your Groups

```bash
python debug_messages.py
```

This shows all groups you can access with their IDs. Example output:
```
📋 Copy these IDs to your .env file:
------------------------------------------------------------
             ID | TYPE     | TITLE                          | USERNAME
------------------------------------------------------------
 -1001234567890 | Channel  | Portal da Fraude               | No username
 -1000987654321 | Channel  | Crypto Trading Tips            | @crypto_tips
    -4010293847 | Group    | My Private Alerts              | No username
```

### Step 5: Update .env with Real Group IDs

Edit `.env` with IDs from step 4:
```bash
# Groups to MONITOR for suspicious content:
TELEGRAM_GROUPS=-1001234567890

# Where to SEND alerts (use different ID for best results):
TELEGRAM_ALERT_CHAT_ID=-4010293847
```

### Step 6: Launch System

```bash
docker-compose up -d
docker-compose logs -f app  # Watch startup logs
```

Look for: `✅ Group accessible: -1001234567890` and `Listening for messages in 1 groups...`

### Step 7: Test It Works

1. **Text Test**: Send "CloudWalk" to monitored group → Should get alert in 5-10 seconds
2. **Image Test**: Send clear image with "Visa" text → Should get alert in 30 seconds

## 📋 What Groups to Use

**For Monitoring (`TELEGRAM_GROUPS`)**:
- Public crypto/trading channels
- Suspicious activity groups
- Any group where fraud might be discussed

**For Alerts (`TELEGRAM_ALERT_CHAT_ID`)**:
- Your private group/chat
- Team notification channel
- Different from monitoring groups (recommended)

## 🔧 Configuration

The system detects these brands by default:
- CloudWalk
- InfinitePay
- Visa
- Mastercard

To add more brands, edit `.env`:
```bash
BRAND_KEYWORDS=CloudWalk,InfinitePay,Visa,Mastercard,YourBrand
```

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| "Session file not found" | Run `python scripts/auth.py` |
| "UsernameInvalidError" | Run `python debug_messages.py` to get real group IDs |
| "No accessible groups" | Join some Telegram groups first |
| "No brand hits detected" | Use clear, readable text in images |
| "Database connection failed" | Check if Docker containers are running |

## 🏗️ Architecture

```
Telegram Groups → Message Collector → OCR + Brand Detection → Alert Sender
      ↓               ↓                        ↓                    ↓
  You monitor      PostgreSQL            Text Analysis        You get alerts
```

**How it works:**
1. Monitors your specified Telegram groups
2. Extracts text from images using OCR
3. Searches for brand mentions using fuzzy matching
4. Sends formatted alerts to your notification chat

**Features:**
- Real-time monitoring
- Image text extraction (OCR)
- Smart text matching (handles typos)
- Duplicate detection
- Rate limiting
- Docker containerized

## 📁 Project Structure

```
├── src/                    # Main application code
├── scripts/               # Setup scripts (auth.py)
├── tests/                 # Basic functionality tests
├── docker-compose.yml     # Container orchestration
├── debug_messages.py      # Group discovery tool
└── .env                  # Your configuration
```

## 🧪 Testing

Run basic tests to verify the system works:

```bash
python run_tests.py
```

Or run tests directly with pytest:
```bash
pytest tests/ -v
```

That's it! The system will now monitor your groups and send alerts when target brands are mentioned.