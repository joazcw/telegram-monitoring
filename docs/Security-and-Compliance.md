# Security and Compliance

## Secrets Management

### Environment Variables
All sensitive configuration must be stored in environment variables, never in code or configuration files.

```bash
# .env file structure (never commit to git)
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=a1b2c3d4e5f6789...
TELEGRAM_SESSION_PATH=./data/telethon.session
DB_URL=postgresql+psycopg2://fraud_user:secure_password@db:5432/fraud_db

# Verify no secrets in logs
grep -r "TELEGRAM_API" logs/ src/ --exclude="*.env*"  # Should return nothing
```

### Session File Security
```bash
# Secure session file permissions (user read-only)
chmod 600 data/telethon.session

# Verify ownership and permissions
ls -la data/telethon.session
# Expected: -rw------- 1 user user ... telethon.session

# Docker container permissions
docker-compose exec app ls -la /app/data/
# Ensure session files not world-readable
```

### Database Credentials
```bash
# Production database setup with minimum privileges
CREATE USER fraud_user WITH PASSWORD 'complex_random_password_here';
CREATE DATABASE fraud_db OWNER fraud_user;

# Grant only necessary permissions
GRANT CONNECT ON DATABASE fraud_db TO fraud_user;
GRANT CREATE, USAGE ON SCHEMA public TO fraud_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO fraud_user;

# Verify permissions are minimal
\du fraud_user  # Should NOT be superuser, createdb, createrole
```

## Access Control

### Telegram Permissions

#### API Application Settings
- **Application Type**: Desktop Application (not Web)
- **Domain**: Leave empty (not required for desktop apps)
- **Callback URL**: Leave empty
- **Description**: "Fraud monitoring system - internal use only"

#### Bot Configuration (for alerts)
```bash
# Create bot with @BotFather
# Commands to set:
/newbot
# Bot Name: Fraud Alert Bot
# Username: your_fraud_alert_bot

# Configure bot settings
/setprivacy  # Enable privacy mode (only sees direct messages)
/setjoingroups  # Disable (bot won't join groups)
/setcommands  # No commands needed
```

#### Session Permissions
```python
# Verify session has minimum required permissions
from telethon import TelegramClient
client = TelegramClient(session, api_id, api_hash)

# Check what groups the session can access
async def verify_permissions():
    await client.start()

    # Should only access explicitly added groups
    async for dialog in client.iter_dialogs():
        if dialog.is_group:
            print(f"Access to group: {dialog.name} (ID: {dialog.id})")

    # Verify no admin rights in groups (read-only access preferred)
    me = await client.get_me()
    print(f"Session user: {me.first_name} (@{me.username})")
```

### Database Security

#### Network Isolation
```yaml
# docker-compose.yml security configuration
version: '3.8'
services:
  db:
    image: postgres:15-alpine
    networks:
      - internal  # No external access
    ports: [] # Remove external port mapping in production

  app:
    networks:
      - internal
      - external  # Only app needs external access

networks:
  internal:
    internal: true  # No internet access
  external: {}
```

#### SSL/TLS Configuration
```bash
# Enable SSL for database connections (production)
DB_URL=postgresql+psycopg2://fraud_user:pass@db:5432/fraud_db?sslmode=require

# Generate certificates for internal communication
openssl req -new -x509 -days 365 -nodes -text \
  -out server.crt -keyout server.key \
  -subj "/CN=fraud-monitor-db"

# Mount certificates in database container
# volumes:
#   - ./certs/server.crt:/var/lib/postgresql/server.crt:ro
#   - ./certs/server.key:/var/lib/postgresql/server.key:ro
```

## Data Protection

### Personally Identifiable Information (PII)

#### What We Collect
- **Telegram User IDs**: Numeric identifiers (not personal names)
- **Chat IDs**: Group identifiers (not personal chats)
- **Message Content**: Text and media from monitored groups only
- **Timestamps**: When messages were posted

#### What We DO NOT Collect
- Phone numbers
- Real names
- Private messages
- User profiles
- Location data
- Contact lists

#### Data Minimization
```python
# Example: PII scrubbing before storage
import re

def scrub_pii(text):
    """Remove potential PII from text before storage"""
    if not text:
        return text

    # Remove phone numbers
    text = re.sub(r'\+?\d{10,15}', '[PHONE_REDACTED]', text)

    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                 '[EMAIL_REDACTED]', text)

    # Remove credit card numbers (basic pattern)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                 '[CARD_REDACTED]', text)

    return text

# Apply in message processing
message.text = scrub_pii(message.text or "")
```

### Data Retention

#### Automated Cleanup
```sql
-- Retention policy: 90 days for message content
DELETE FROM telegram_messages
WHERE timestamp < NOW() - INTERVAL '90 days';

-- Retention policy: 30 days for media files
DELETE FROM images
WHERE timestamp < NOW() - INTERVAL '30 days';

-- Keep brand hits longer for analysis (1 year)
DELETE FROM brand_hits
WHERE timestamp < NOW() - INTERVAL '365 days';

-- Keep OCR results same as images
DELETE FROM ocr_text
WHERE image_id NOT IN (SELECT id FROM images);
```

#### Automated cleanup script
```python
# scripts/cleanup_old_data.py
import os
from datetime import datetime, timedelta
from src.database import SessionLocal
from src.models import TelegramMessage, ImageRecord, OCRText

def cleanup_old_data():
    db = SessionLocal()

    # Delete old messages (90 days)
    cutoff_date = datetime.utcnow() - timedelta(days=90)
    old_messages = db.query(TelegramMessage).filter(
        TelegramMessage.timestamp < cutoff_date
    ).count()

    db.query(TelegramMessage).filter(
        TelegramMessage.timestamp < cutoff_date
    ).delete()

    # Clean orphaned media files
    media_cutoff = datetime.utcnow() - timedelta(days=30)
    old_images = db.query(ImageRecord).filter(
        ImageRecord.timestamp < media_cutoff
    ).all()

    for image in old_images:
        if os.path.exists(image.file_path):
            os.remove(image.file_path)

    db.query(ImageRecord).filter(
        ImageRecord.timestamp < media_cutoff
    ).delete()

    db.commit()
    print(f"Cleaned up {old_messages} messages and {len(old_images)} images")

# Crontab entry: Run daily at 2 AM
# 0 2 * * * cd /opt/fraud-monitor && docker-compose exec app python scripts/cleanup_old_data.py
```

### Right to Deletion

#### Data Subject Requests
```python
# scripts/delete_user_data.py
def delete_user_data(user_id: int):
    """Delete all data for a specific Telegram user ID"""
    db = SessionLocal()

    try:
        # Find all messages from this user
        messages = db.query(TelegramMessage).filter(
            TelegramMessage.sender_id == user_id
        ).all()

        message_ids = [m.id for m in messages]

        # Delete associated data
        db.query(BrandHit).filter(BrandHit.message_id.in_(message_ids)).delete()
        db.query(OCRText).filter(OCRText.image_id.in_(
            db.query(ImageRecord.id).filter(ImageRecord.message_id.in_(message_ids))
        )).delete()

        # Delete media files
        images = db.query(ImageRecord).filter(
            ImageRecord.message_id.in_(message_ids)
        ).all()

        for image in images:
            if os.path.exists(image.file_path):
                os.remove(image.file_path)

        db.query(ImageRecord).filter(
            ImageRecord.message_id.in_(message_ids)
        ).delete()

        # Delete messages
        db.query(TelegramMessage).filter(
            TelegramMessage.sender_id == user_id
        ).delete()

        db.commit()
        print(f"Deleted all data for user {user_id}")

    except Exception as e:
        db.rollback()
        print(f"Error deleting user data: {e}")
        raise
    finally:
        db.close()
```

## Log Security

### Sensitive Data Redaction
```python
# src/logging_config.py
import re
import logging

class PIIRedactingFormatter(logging.Formatter):
    """Custom formatter that redacts PII from log messages"""

    PII_PATTERNS = [
        (r'\b\d{10,15}\b', '[PHONE]'),  # Phone numbers
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Emails
        (r'TELEGRAM_API_HASH=\w+', 'TELEGRAM_API_HASH=[REDACTED]'),  # API hash
        (r'password[=:]\s*\S+', 'password=[REDACTED]'),  # Passwords
    ]

    def format(self, record):
        # Get original message
        msg = super().format(record)

        # Apply redaction patterns
        for pattern, replacement in self.PII_PATTERNS:
            msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)

        return msg

# Configure logging with PII redaction
def setup_logging():
    formatter = PIIRedactingFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = logging.FileHandler('data/logs/app.log')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

### Log Access Control
```bash
# Restrict log file permissions
chmod 640 data/logs/*.log
chown app:adm data/logs/*.log  # Allow admin group read access

# Log rotation with compression and retention
# /etc/logrotate.d/fraud-monitor
/opt/fraud-monitor/data/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        docker-compose restart app > /dev/null 2>&1 || true
    endscript
}
```

## Network Security

### Firewall Configuration
```bash
# UFW rules for production server
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH access (change port from 22)
sudo ufw allow 2222/tcp

# HTTPS for optional web interface
sudo ufw allow 443/tcp

# Block direct database access from external
sudo ufw deny 5432/tcp

# Apply rules
sudo ufw enable
sudo ufw status
```

### Container Network Isolation
```yaml
# docker-compose.yml with network security
services:
  app:
    networks:
      - backend
      - frontend
    # Only app can access external services

  db:
    networks:
      - backend  # Database isolated to backend network
    # No direct external access

networks:
  frontend:
    # External access for app
  backend:
    internal: true  # No internet access for backend services
```

## Compliance Considerations

### Telegram Terms of Service

#### Allowed Use Cases
- ✅ Monitoring public/semi-public groups where you have permission
- ✅ Fraud detection and prevention for business purposes
- ✅ Brand mention monitoring for reputation management
- ✅ Automated content analysis for security purposes

#### Prohibited Activities
- ❌ Spamming or sending unsolicited messages
- ❌ Collecting private user data without consent
- ❌ Accessing private chats or groups without authorization
- ❌ Circumventing Telegram's anti-spam measures
- ❌ Selling or sharing user data with third parties

#### Best Practices
```python
# Implement rate limiting to respect API limits
import time
from datetime import datetime, timedelta

class TelegramRateLimiter:
    def __init__(self, max_requests_per_minute=20):
        self.max_requests = max_requests_per_minute
        self.requests = []

    async def wait_if_needed(self):
        now = datetime.utcnow()
        # Remove requests older than 1 minute
        self.requests = [req for req in self.requests
                        if now - req < timedelta(minutes=1)]

        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.requests.append(now)
```

### GDPR Compliance (if applicable)

#### Legal Basis for Processing
- **Legitimate Interest**: Fraud prevention and brand protection
- **Consent**: Explicit consent from group administrators
- **Contract**: If monitoring is part of service agreement

#### Privacy Notice Template
```
PRIVACY NOTICE - Fraud Monitoring System

We collect and process messages from Telegram groups for fraud detection purposes.

Data Collected:
- Message content from monitored public/semi-public groups
- Timestamps and group identifiers
- Media files containing potential brand mentions

Legal Basis: Legitimate interest in fraud prevention

Retention: Messages deleted after 90 days, brand hits after 1 year

Your Rights:
- Access: Request copy of your data
- Deletion: Request removal of your data
- Objection: Opt out of monitoring

Contact: security@yourcompany.com
```

#### Data Processing Record
```python
# Maintain processing log for compliance
def log_processing_activity(activity_type, data_subject=None, legal_basis="legitimate_interest"):
    processing_log = {
        'timestamp': datetime.utcnow(),
        'activity': activity_type,  # 'collect', 'process', 'delete'
        'data_subject': data_subject,  # user_id if available
        'legal_basis': legal_basis,
        'retention_period': '90 days' if activity_type == 'collect' else None,
        'processing_purpose': 'fraud_detection_brand_monitoring'
    }

    # Store in separate compliance database or append to log file
    with open('data/logs/processing_log.jsonl', 'a') as f:
        f.write(json.dumps(processing_log) + '\n')
```

## Security Monitoring

### Intrusion Detection
```bash
# Monitor for suspicious activity
tail -f data/logs/app.log | grep -E "(WARN|ERROR|Exception)" &

# Check for failed authentication attempts
grep -i "unauthorized\|forbidden\|failed" data/logs/app.log | tail -20

# Monitor for unusual API usage
grep -i "rate limit\|too many requests" data/logs/app.log | tail -10

# Database connection monitoring
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT client_addr, state, COUNT(*)
FROM pg_stat_activity
WHERE datname = 'fraud_db'
GROUP BY client_addr, state;"
```

### Security Alerts
```python
# scripts/security_monitor.py
import smtplib
from email.mime.text import MIMEText

def send_security_alert(severity, message):
    """Send security alert via email"""
    if severity in ['HIGH', 'CRITICAL']:
        subject = f"[FRAUD-MONITOR] {severity} Security Alert"

        msg = MIMEText(f"""
Security Alert: {severity}

Message: {message}
Time: {datetime.utcnow()}
System: Fraud Monitoring System

Please investigate immediately.
        """)

        msg['Subject'] = subject
        msg['From'] = 'fraud-monitor@yourcompany.com'
        msg['To'] = 'security@yourcompany.com'

        # Configure SMTP settings
        smtp = smtplib.SMTP('localhost')
        smtp.send_message(msg)
        smtp.quit()

# Example usage in monitoring script
def check_security_indicators():
    # Check for excessive failed requests
    with open('data/logs/app.log') as f:
        recent_errors = [line for line in f if 'ERROR' in line and
                        datetime_from_log(line) > datetime.utcnow() - timedelta(hours=1)]

    if len(recent_errors) > 10:
        send_security_alert('HIGH', f'{len(recent_errors)} errors in last hour')
```