#!/usr/bin/env python3
"""
Simple Database Initialization Script
Creates all required tables for the Fraud Monitoring System.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import init_db, check_db_connection
from src.config import DB_URL


def main():
    """Initialize the database with all required tables"""
    print("🗄️  Fraud Monitoring System - Database Initialization")
    print("=" * 60)

    # Display database URL (without sensitive info)
    if 'localhost' in DB_URL:
        print(f"Database: {DB_URL}")
    else:
        print(f"Database: {DB_URL.split('@')[1] if '@' in DB_URL else 'configured'}")

    print("\n1️⃣  Checking database connection...")
    if not check_db_connection():
        print("❌ Cannot connect to database")
        print("💡 Make sure PostgreSQL is running and credentials are correct")
        return 1

    print("✅ Database connection successful")

    print("\n2️⃣  Initializing database schema...")
    try:
        init_db()
        print("✅ All tables created successfully")

        print("\n📋 Created tables:")
        print("   • telegram_messages - Stores Telegram message data")
        print("   • images - Image file records and hashes")
        print("   • ocr_text - OCR extraction results")
        print("   • brand_hits - Fraud detection matches")

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return 1

    print("\n✅ Database initialization complete!")
    print("🚀 Your fraud monitoring system is ready to use")
    return 0


if __name__ == "__main__":
    sys.exit(main())