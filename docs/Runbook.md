# Operations Runbook

## Local Development Setup

### Prerequisites
```bash
# System dependencies
sudo apt-get install tesseract-ocr tesseract-ocr-eng postgresql-client

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Development Database
```bash
# Start PostgreSQL container
docker run -d --name fraud-db \
  -e POSTGRES_USER=fraud_user \
  -e POSTGRES_PASSWORD=fraud_pass \
  -e POSTGRES_DB=fraud_db \
  -p 5432:5432 postgres:15-alpine

# Run migrations
export DB_URL="postgresql+psycopg2://fraud_user:fraud_pass@localhost:5432/fraud_db"
alembic upgrade head

# Create test data
python -m src.test_data_generator
```

### Running Services
```bash
# Start message collector
python -m src.main &

# Start processing in separate terminal
python -m src.processor &

# Monitor logs
tail -f data/logs/app.log
```

## Production Deployment

### VM Setup (Ubuntu 22.04)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Create application directory
sudo mkdir -p /opt/fraud-monitor
sudo chown $USER:$USER /opt/fraud-monitor
cd /opt/fraud-monitor

# Clone repository
git clone <repo-url> .
```

### Environment Configuration
```bash
# Create production environment file
cp .env.example .env.prod

# Edit with production values
vim .env.prod

# CRITICAL: Set these values
# TELEGRAM_API_ID=<from my.telegram.org>
# TELEGRAM_API_HASH=<from my.telegram.org>
# TELEGRAM_ALERT_CHAT_ID=<your alert chat>
# BRAND_KEYWORDS=<your monitored brands>
```

### Production Startup
```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# Verify health
docker-compose ps
docker-compose logs app | grep "Connected to Telegram"

# First-time authentication
docker-compose exec app python -m src.auth
# Follow prompts to authenticate Telegram session
```

### SSL/TLS Setup (Optional)
```bash
# Install Caddy for automatic HTTPS
docker run -d --name caddy \
  -p 80:80 -p 443:443 \
  -v /opt/fraud-monitor/Caddyfile:/etc/caddy/Caddyfile \
  -v caddy_data:/data \
  caddy:2-alpine

# Caddyfile content:
# fraud-monitor.yourdomain.com {
#   reverse_proxy app:8080
# }
```

## Health Checks and Monitoring

### System Health Verification
```bash
# Check all services running
docker-compose ps

# Expected output:
# fraud-monitor-app  Up    healthy
# fraud-monitor-db   Up    healthy

# Verify Telegram connection
docker-compose logs app | tail -20 | grep -E "(Connected|Error|Exception)"

# Check database connectivity
docker-compose exec db psql -U fraud_user -d fraud_db -c "SELECT COUNT(*) FROM telegram_messages;"

# Verify disk space
df -h /opt/fraud-monitor/data
# Should have >1GB free for media storage
```

### Application Metrics
```bash
# Message processing rate (last hour)
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT COUNT(*) as messages_last_hour
FROM telegram_messages
WHERE timestamp > NOW() - INTERVAL '1 hour';"

# OCR processing status
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT
  COUNT(*) as total_images,
  COUNT(*) FILTER (WHERE processed = true) as processed,
  COUNT(*) FILTER (WHERE processed = false) as pending
FROM images;"

# Brand hit summary (last 24 hours)
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT brand_name, COUNT(*) as hits, COUNT(*) FILTER (WHERE alert_sent = true) as alerted
FROM brand_hits
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY brand_name;"
```

### Log Analysis
```bash
# Application logs
docker-compose logs -f app

# Key log patterns to watch:
# ✅ "Connected to Telegram" - Client healthy
# ✅ "Processed message 12345" - Message ingestion working
# ✅ "OCR completed in 2341ms" - OCR processing active
# ✅ "Brand hit detected: CloudWalk" - Detection working
# ✅ "Alert sent for brand: CloudWalk" - Alerting functional

# ❌ "Rate limit exceeded" - Telegram API throttling
# ❌ "Session file not found" - Authentication issue
# ❌ "Database connection failed" - Database issue
# ❌ "OCR failed for" - Processing error
# ❌ "Failed to send alert" - Alerting issue
```

## Common Issues and Resolutions

### Telegram Session Issues

#### Symptom: "Session file not found" or "Unauthorized"
```bash
# Solution: Reauthorize session
docker-compose exec app python -m src.auth

# If container can't start, use temporary container:
docker run -it --rm -v $(pwd)/data:/app/data \
  -e TELEGRAM_API_ID=$TELEGRAM_API_ID \
  -e TELEGRAM_API_HASH=$TELEGRAM_API_HASH \
  fraud-monitor python -m src.auth
```

#### Symptom: "Rate limit exceeded"
```bash
# Check current limits
docker-compose logs app | grep -i "rate limit" | tail -5

# Solution: Wait and verify backoff working
# Application should automatically retry with exponential backoff
# If persistent, check for message flood in monitored groups

# Temporary fix: Reduce group list
docker-compose exec app python -c "
from src.config import TELEGRAM_GROUPS
print('Current groups:', TELEGRAM_GROUPS)
"
# Edit .env to reduce TELEGRAM_GROUPS, restart container
```

### Database Issues

#### Symptom: "Database connection failed"
```bash
# Check database container status
docker-compose ps db

# If unhealthy, check logs
docker-compose logs db

# Common fixes:
# 1. Disk space full
df -h
# 2. Port conflict
sudo netstat -tlnp | grep 5432
# 3. Memory issues
free -h

# Reset database (CAUTION: Data loss)
docker-compose down
docker volume rm fraud-monitor_postgres_data
docker-compose up -d
```

#### Symptom: Slow queries or high CPU
```bash
# Check for long-running queries
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"

# Check database stats
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT schemaname,tablename,n_tup_ins,n_tup_upd,n_tup_del,n_live_tup,n_dead_tup
FROM pg_stat_user_tables ORDER BY n_live_tup DESC;"

# Run maintenance
docker-compose exec db psql -U fraud_user -d fraud_db -c "VACUUM ANALYZE;"
```

### Processing Issues

#### Symptom: Images not being processed (OCR stuck)
```bash
# Check processing queue
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT COUNT(*) as pending_images, MIN(timestamp) as oldest_pending
FROM images WHERE processed = false;"

# Check Tesseract installation
docker-compose exec app tesseract --version
docker-compose exec app tesseract --list-langs

# Test OCR manually
docker-compose exec app python -c "
from src.ocr_engine import OCREngine
ocr = OCREngine()
result = ocr.extract_text('/app/test-image.png')
print('OCR Result:', result)
"

# Clear stuck processing (CAUTION: Reprocesses images)
docker-compose exec db psql -U fraud_user -d fraud_db -c "
UPDATE images SET processed = false
WHERE processed = true AND id NOT IN (SELECT image_id FROM ocr_text);"
```

#### Symptom: No brand hits detected
```bash
# Test brand matcher
docker-compose exec app python -c "
from src.brand_matcher import BrandMatcher
matcher = BrandMatcher()
test_text = 'Check out CloudWalk payment system'
matches = matcher.find_matches(test_text)
print('Matches found:', matches)
"

# Check current brand keywords
docker-compose exec app python -c "
from src.config import BRAND_KEYWORDS, FUZZY_THRESHOLD
print('Keywords:', BRAND_KEYWORDS)
print('Threshold:', FUZZY_THRESHOLD)
"

# Verify OCR text extraction
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT id, LEFT(extracted_text, 100) as text_sample, confidence
FROM ocr_text
ORDER BY timestamp DESC LIMIT 10;"
```

### Alert Issues

#### Symptom: Alerts not being sent
```bash
# Check unsent alerts
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT COUNT(*) as unsent_alerts, MIN(timestamp) as oldest
FROM brand_hits WHERE alert_sent = false;"

# Test alert sender
docker-compose exec app python -c "
import asyncio
from src.alerting import AlertSender
sender = AlertSender()
test_data = {
    'brand': 'Test',
    'matched_text': 'test message',
    'confidence': 95,
    'chat_id': 'test',
    'message_id': 'test'
}
asyncio.run(sender.send_brand_alert(test_data))
"

# Check Telegram bot token
docker-compose exec app python -c "
from src.config import TELEGRAM_ALERT_CHAT_ID
print('Alert chat ID:', TELEGRAM_ALERT_CHAT_ID)
"
```

## Maintenance Procedures

### Daily Operations
```bash
# Check system health
./scripts/health-check.sh

# Monitor disk usage
du -sh data/media/* | sort -h | tail -10

# Review error logs
docker-compose logs app | grep -i error | tail -20

# Verify alert delivery
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT
  DATE(timestamp) as date,
  COUNT(*) as total_hits,
  COUNT(*) FILTER (WHERE alert_sent = true) as alerts_sent
FROM brand_hits
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;"
```

### Weekly Maintenance
```bash
# Database maintenance
docker-compose exec db psql -U fraud_user -d fraud_db -c "VACUUM ANALYZE;"

# Clean old media files (>30 days)
find data/media -type f -mtime +30 -exec rm {} \;

# Update Docker images
docker-compose pull
docker-compose up -d

# Check for application updates
git fetch
git log --oneline HEAD..origin/main

# Backup database
docker-compose exec db pg_dump -U fraud_user fraud_db | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Monthly Operations
```bash
# Archive old data (>90 days)
docker-compose exec db psql -U fraud_user -d fraud_db -c "
DELETE FROM telegram_messages WHERE timestamp < NOW() - INTERVAL '90 days';"

# Analyze database performance
docker-compose exec db psql -U fraud_user -d fraud_db -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC LIMIT 10;"

# Review and rotate logs
docker-compose logs app > logs/app_$(date +%Y%m).log
docker-compose restart app

# Security updates
sudo apt update && sudo apt upgrade
docker system prune -f
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker-compose exec db pg_dump -U fraud_user fraud_db | gzip > fraud_backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Automated daily backup (crontab)
0 2 * * * /opt/fraud-monitor/scripts/backup-db.sh

# backup-db.sh content:
#!/bin/bash
cd /opt/fraud-monitor
docker-compose exec -T db pg_dump -U fraud_user fraud_db | gzip > "backups/fraud_$(date +%Y%m%d).sql.gz"
find backups/ -name "fraud_*.sql.gz" -mtime +7 -delete
```

### Recovery Procedure
```bash
# Stop application
docker-compose down

# Restore database
gunzip -c fraud_backup_20250927_020000.sql.gz | docker-compose exec -T db psql -U fraud_user fraud_db

# Restore media files (if backed up separately)
tar -xzf media_backup_20250927.tar.gz -C data/

# Restart services
docker-compose up -d

# Verify recovery
docker-compose logs app | grep "Connected to Telegram"
```

### Disaster Recovery
```bash
# Complete system rebuild
# 1. Provision new VM
# 2. Install Docker
# 3. Clone repository
# 4. Restore .env.prod
# 5. Restore database backup
# 6. Restore media files
# 7. Start services
# 8. Reauthorize Telegram session if needed

# Recovery time objective: <30 minutes
# Recovery point objective: <24 hours (daily backups)
```