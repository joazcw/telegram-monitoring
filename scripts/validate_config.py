#!/usr/bin/env python3
"""
Configuration Validator
Quick check if environment is properly configured before authentication
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🔍 Validating Configuration...")
    print("=" * 50)

    try:
        from src.config import (
            TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_PATH,
            TELEGRAM_GROUPS, TELEGRAM_ALERT_CHAT_ID, BRAND_KEYWORDS
        )

        issues = []

        # Check Telegram API configuration
        if not TELEGRAM_API_ID or TELEGRAM_API_ID == 0:
            issues.append("❌ TELEGRAM_API_ID is missing or invalid")
        else:
            print(f"✅ API ID: {TELEGRAM_API_ID}")

        if not TELEGRAM_API_HASH:
            issues.append("❌ TELEGRAM_API_HASH is missing")
        else:
            print(f"✅ API Hash: {TELEGRAM_API_HASH[:8]}...")

        # Check groups configuration
        if not TELEGRAM_GROUPS:
            issues.append("❌ TELEGRAM_GROUPS is empty")
        else:
            print(f"✅ Groups: {len(TELEGRAM_GROUPS)} configured ({', '.join(TELEGRAM_GROUPS)})")

        # Check alert chat
        if not TELEGRAM_ALERT_CHAT_ID:
            issues.append("❌ TELEGRAM_ALERT_CHAT_ID is missing")
        else:
            print(f"✅ Alert Chat ID: {TELEGRAM_ALERT_CHAT_ID}")

        # Check brand keywords
        if not BRAND_KEYWORDS:
            issues.append("❌ BRAND_KEYWORDS is empty")
        else:
            print(f"✅ Brand Keywords: {len(BRAND_KEYWORDS)} configured ({', '.join(BRAND_KEYWORDS)})")

        # Check session path directory
        session_dir = Path(TELEGRAM_SESSION_PATH).parent
        if not session_dir.exists():
            print(f"📁 Creating session directory: {session_dir}")
            session_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Session directory created")
        else:
            print(f"✅ Session directory exists: {session_dir}")

        print("\n" + "=" * 50)

        if issues:
            print("❌ Configuration Issues Found:")
            for issue in issues:
                print(f"   {issue}")
            print("\nPlease fix these issues in your .env file")
            return False
        else:
            print("✅ Configuration looks good!")
            print("\nYou can now run the authentication:")
            print("   python scripts/auth.py")
            print("\nNote: Authentication requires interactive input")
            print("Run it directly in your terminal (not through AI assistant)")
            return True

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)