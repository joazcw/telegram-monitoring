#!/bin/bash

# Simple Fraud Monitoring System Deployment
set -e

echo "🚀 Deploying Fraud Monitoring System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "❌ Please edit .env file with your Telegram credentials before continuing."
    echo "   Run: nano .env"
    exit 1
fi

# Create data directory
echo "📁 Creating data directories..."
mkdir -p data/media data/logs

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up -d --build

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 15

# Initialize database
echo "🗄️  Initializing database..."
docker-compose exec app python -c "from src.database import init_db; init_db(); print('✅ Database initialized')"

echo "✅ Deployment complete!"
echo "📋 Services running:"
echo "   • Application: fraud monitoring"
echo "   • Database: PostgreSQL"
echo ""
echo "📝 Useful commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop: docker-compose down"
echo "   • Restart: docker-compose restart"