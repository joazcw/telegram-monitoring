# Docker Deployment Guide

This guide explains how to deploy the Fraud Monitoring System using Docker.

## Prerequisites

- Docker (version 20.10+)
- Docker Compose (version 2.0+)
- At least 2GB RAM available
- 5GB free disk space

## Quick Start

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd fraud-monitor
   cp .env.example .env
   ```

2. **Configure environment variables**:
   Edit `.env` file with your Telegram API credentials:
   ```bash
   nano .env
   ```

3. **Deploy the system**:
   ```bash
   ./scripts/deploy.sh
   ```

## Manual Deployment

### 1. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

**Required Variables:**
- `TELEGRAM_API_ID`: Your Telegram API ID from https://my.telegram.org/apps
- `TELEGRAM_API_HASH`: Your Telegram API Hash
- `TELEGRAM_GROUPS`: Comma-separated list of groups to monitor
- `TELEGRAM_ALERT_CHAT_ID`: Chat ID where alerts will be sent
- `SESSION_ENCRYPTION_KEY`: 32-character encryption key for sessions

**Optional Variables:**
- `BRAND_KEYWORDS`: Brands to detect (default: CloudWalk,InfinitePay,Visa,Mastercard)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: Processing batch size (default: 10)

### 2. Build and Start Services

```bash
# Build the application image
docker-compose build

# Start database first
docker-compose up -d db

# Wait for database to be ready
sleep 10

# Run database migrations
docker-compose run --rm app python scripts/migrate.py upgrade

# Start all services
docker-compose up -d
```

### 3. Verify Deployment

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Check health
docker-compose exec app python -c "from src.database import check_db_connection; print('DB:', check_db_connection())"
```

## Services

The system includes the following services:

### Application (`fraud-monitor-app`)
- Main fraud monitoring application
- Processes Telegram messages and images
- Sends alerts when brand mentions are detected
- Health check endpoint available

### Database (`fraud-monitor-db`)
- PostgreSQL 15 database
- Stores messages, images, OCR results, and brand hits
- Persistent data with Docker volumes
- Health checks with `pg_isready`

### Redis (`fraud-monitor-redis`)
- Optional caching layer
- Used for rate limiting and temporary data
- Improves performance for high-volume processing

## Data Persistence

The system uses Docker volumes for data persistence:

- `postgres_data`: Database files
- `redis_data`: Redis data
- `./data:/app/data`: Application data (mounted from host)
- `./logs:/app/data/logs`: Log files (mounted from host)

## Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f db
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart app
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (DANGER: destroys data)
docker-compose down -v
```

### Update Application
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Database Operations
```bash
# Run migrations
docker-compose run --rm app python scripts/migrate.py upgrade

# Access database
docker-compose exec db psql -U fraud_user -d fraud_db

# Backup database
docker-compose exec db pg_dump -U fraud_user fraud_db > backup.sql

# Restore database
docker-compose exec -T db psql -U fraud_user -d fraud_db < backup.sql
```

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check if database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Restart database
docker-compose restart db
```

**Application Won't Start**
```bash
# Check application logs
docker-compose logs app

# Check environment variables
docker-compose exec app env | grep TELEGRAM

# Restart application
docker-compose restart app
```

**Health Check Failures**
```bash
# Check all service health
docker-compose ps

# Run manual health check
docker-compose exec app python -c "
from src.processor import FraudProcessor
import asyncio
async def check():
    p = FraudProcessor()
    return await p._health_check()
print(asyncio.run(check()))
"
```

### Performance Tuning

**High Memory Usage**
- Reduce `BATCH_SIZE` in `.env`
- Increase `PROCESSING_INTERVAL`
- Monitor with `docker stats`

**Slow Processing**
- Increase `BATCH_SIZE`
- Decrease `PROCESSING_INTERVAL`
- Consider adding more CPU cores

**Database Performance**
- Monitor with `docker-compose exec db pg_stat_activity`
- Consider connection pooling
- Optimize queries if needed

## Security Considerations

1. **Environment Variables**: Never commit `.env` to version control
2. **Database**: Change default passwords in production
3. **Network**: Use Docker networks for service isolation
4. **Updates**: Regularly update base images for security patches
5. **Backup**: Regular database backups with encryption

## Production Deployment

For production deployment, consider:

1. **Orchestration**: Use Docker Swarm or Kubernetes
2. **Load Balancing**: Multiple application instances
3. **Monitoring**: Add Prometheus/Grafana monitoring
4. **Logging**: Centralized logging with ELK stack
5. **Backup**: Automated backup strategies
6. **SSL/TLS**: Secure communications
7. **Secrets Management**: Use Docker secrets or external vault

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f`
2. Review health checks: `docker-compose ps`
3. Verify configuration: Check `.env` file
4. Monitor resources: `docker stats`