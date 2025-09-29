#!/bin/bash

# Fraud Monitoring System Deployment Script
set -e

echo "🚀 Deploying Fraud Monitoring System..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Please create one based on .env.example"
    print_status "Copying .env.example to .env..."
    cp .env.example .env
    print_warning "Please edit .env file with your Telegram credentials before continuing."
    print_warning "Run: nano .env"
    exit 1
fi

# Create necessary directories
print_status "Creating data directories..."
mkdir -p data/media data/logs

# Build the application
print_status "Building Docker images..."
docker-compose build

# Run database migrations
print_status "Setting up database..."
docker-compose up -d db
sleep 10  # Wait for database to be ready

print_status "Running database migrations..."
docker-compose run --rm app python scripts/migrate.py upgrade

# Start all services
print_status "Starting all services..."
docker-compose up -d

# Wait a bit for services to start
sleep 5

# Check service health
print_status "Checking service health..."
docker-compose ps

# Show logs
print_status "Showing recent logs..."
docker-compose logs --tail=20

print_status "✅ Deployment complete!"
print_status "Services are running at:"
print_status "  - Application: fraud-monitor-app"
print_status "  - Database: fraud-monitor-db (port 5432)"
print_status "  - Redis: fraud-monitor-redis (port 6379)"
print_status ""
print_status "To view logs: docker-compose logs -f"
print_status "To stop services: docker-compose down"
print_status "To restart: docker-compose restart"