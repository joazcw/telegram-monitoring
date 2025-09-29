#!/usr/bin/env python3
"""
Telegram Authentication Helper

This script helps authenticate with Telegram API and create session files
for the fraud monitoring system.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.telegram_collector import TelegramCollector
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_PATH
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print authentication banner"""
    print("=" * 60)
    print("🔐 Telegram Authentication Helper")
    print("   Fraud Monitoring System")
    print("=" * 60)
    print()


def validate_config():
    """Validate configuration before authentication"""
    print("📋 Checking configuration...")

    if not TELEGRAM_API_ID or TELEGRAM_API_ID == 0:
        print("❌ TELEGRAM_API_ID is missing or invalid")
        print("   Get it from: https://my.telegram.org/apps")
        return False

    if not TELEGRAM_API_HASH:
        print("❌ TELEGRAM_API_HASH is missing")
        print("   Get it from: https://my.telegram.org/apps")
        return False

    print(f"✅ API ID: {TELEGRAM_API_ID}")
    print(f"✅ API Hash: {TELEGRAM_API_HASH[:8]}...")
    print(f"✅ Session Path: {TELEGRAM_SESSION_PATH}")
    print()

    return True


async def authenticate():
    """Perform Telegram authentication"""
    print("🚀 Starting authentication process...")
    print()

    try:
        collector = TelegramCollector()

        if not collector.credentials_available:
            print("❌ Telegram credentials not configured properly")
            return False

        print("📱 Connecting to Telegram...")
        await collector.authenticate_session()

        print()
        print("✅ Authentication successful!")
        print(f"   Session saved to: {TELEGRAM_SESSION_PATH}")

        # Get account info
        print("\n📞 Getting account information...")
        if collector.client:
            me = await collector.client.get_me()
            print(f"   Account: {me.first_name} {me.last_name or ''}")
            print(f"   Username: @{me.username or 'N/A'}")
            print(f"   Phone: {me.phone or 'N/A'}")
            print(f"   ID: {me.id}")

        # Get group information if configured
        group_info = await collector.get_group_info()
        if group_info:
            print(f"\n📋 Found {len(group_info)} configured groups:")
            for info in group_info:
                print(f"   - {info['title']} (ID: {info['id']})")
                if info['username']:
                    print(f"     Username: @{info['username']}")
        else:
            print("\n⚠️  No groups configured or accessible")
            print("   Check TELEGRAM_GROUPS in your .env file")

        await collector.stop()
        return True

    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False


def check_existing_session():
    """Check if session already exists"""
    session_file = Path(TELEGRAM_SESSION_PATH)

    if session_file.exists():
        print(f"🔍 Found existing session: {session_file}")
        response = input("   Do you want to recreate it? (y/N): ").strip().lower()

        if response in ['y', 'yes']:
            print("   Removing existing session...")
            session_file.unlink()
            # Also remove journal file if it exists
            journal_file = session_file.with_suffix('.session-journal')
            if journal_file.exists():
                journal_file.unlink()
            print("   ✅ Existing session removed")
            return False
        else:
            print("   Keeping existing session")
            return True

    return False


async def test_session():
    """Test existing session"""
    print("🧪 Testing existing session...")

    try:
        collector = TelegramCollector()

        if not collector.credentials_available:
            print("❌ Credentials not available")
            return False

        # Try to connect with existing session
        await collector.client.start()

        if await collector.client.is_user_authorized():
            me = await collector.client.get_me()
            print(f"✅ Session valid - Logged in as: {me.first_name}")

            # Test health check
            health_ok = await collector.health_check()
            print(f"✅ Health check: {'PASSED' if health_ok else 'FAILED'}")

            await collector.stop()
            return True
        else:
            print("❌ Session exists but not authorized")
            await collector.stop()
            return False

    except Exception as e:
        logger.error(f"Session test failed: {e}")
        return False


def print_help():
    """Print help information"""
    print("📚 Help Information:")
    print()
    print("Before running this script, make sure you have:")
    print("1. Created a Telegram application at https://my.telegram.org/apps")
    print("2. Added TELEGRAM_API_ID and TELEGRAM_API_HASH to your .env file")
    print("3. Added the groups you want to monitor to TELEGRAM_GROUPS")
    print("4. Set the TELEGRAM_ALERT_CHAT_ID for receiving alerts")
    print()
    print("During authentication, you will need:")
    print("- Your phone number (with country code)")
    print("- SMS verification code")
    print("- Two-factor password (if enabled)")
    print()
    print("Session files will be stored at:")
    print(f"  {TELEGRAM_SESSION_PATH}")
    print()


def main():
    """Main authentication flow"""
    print_banner()

    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_help()
        return

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 Testing mode - checking existing session")
        result = asyncio.run(test_session())
        sys.exit(0 if result else 1)

    # Validate configuration
    if not validate_config():
        print("\n❌ Configuration validation failed")
        print("Please check your .env file and try again")
        print("\nFor help, run: python scripts/auth.py --help")
        sys.exit(1)

    # Check for existing session
    if check_existing_session():
        print("\n🧪 Testing existing session...")
        result = asyncio.run(test_session())
        if result:
            print("\n✅ Authentication already complete!")
            print("Your session is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Existing session is invalid, recreating...")

    # Perform authentication
    print("🔐 Starting interactive authentication...")
    print("Please have your phone ready to receive SMS codes.")
    print()

    result = asyncio.run(authenticate())

    if result:
        print("\n🎉 Authentication completed successfully!")
        print("You can now start the fraud monitoring system.")
        print()
        print("Next steps:")
        print("1. docker-compose up -d")
        print("2. Check logs: docker-compose logs -f")
    else:
        print("\n❌ Authentication failed")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()