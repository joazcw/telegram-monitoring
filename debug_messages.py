#!/usr/bin/env python3
"""
Group Discovery Script - Find Telegram groups/channels you can access
Run this after authentication to get group IDs for your .env file
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telethon import TelegramClient
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_PATH

async def main():
    print("🔍 Discovering your accessible Telegram groups...")
    print("=" * 60)

    try:
        # Initialize Telegram client
        client = TelegramClient(TELEGRAM_SESSION_PATH, TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()

        # Get user info
        me = await client.get_me()
        print(f"👤 Logged in as: {me.first_name} (@{me.username or 'no_username'})")
        print("=" * 60)

        # Find accessible groups/channels
        groups = []
        async for dialog in client.iter_dialogs():
            if dialog.is_group or dialog.is_channel:
                groups.append(dialog)

        if not groups:
            print("❌ No groups/channels found.")
            print("\n💡 Troubleshooting:")
            print("   - Make sure you're a member of some Telegram groups")
            print("   - Join groups you want to monitor before running this script")
            return

        print(f"✅ Found {len(groups)} accessible groups/channels:")
        print()
        print("📋 Copy these IDs to your .env file:")
        print("-" * 60)
        print(f"{'ID':>15} | {'TYPE':8} | {'TITLE':30} | {'USERNAME':20}")
        print("-" * 60)

        for dialog in groups[:20]:  # Show first 20 groups
            group_type = "Channel" if dialog.is_channel else "Group"
            username = f"@{dialog.entity.username}" if hasattr(dialog.entity, 'username') and dialog.entity.username else "No username"
            title = dialog.title[:28] + "..." if len(dialog.title) > 28 else dialog.title
            print(f"{dialog.id:>15} | {group_type:8} | {title:30} | {username:20}")

        if len(groups) > 20:
            print(f"... and {len(groups) - 20} more groups")

        print()
        print("📝 Example .env configuration:")
        print("-" * 40)
        if groups:
            # Show first group as monitoring example
            monitor_id = groups[0].id
            # Show second group as alert example, or same if only one
            alert_id = groups[1].id if len(groups) > 1 else groups[0].id

            print(f"# Monitor this group for suspicious messages:")
            print(f"TELEGRAM_GROUPS={monitor_id}")
            print(f"# Send alerts to this chat:")
            print(f"TELEGRAM_ALERT_CHAT_ID={alert_id}")

            if monitor_id == alert_id and len(groups) == 1:
                print("# ⚠️  Using same group for monitoring and alerts")
                print("# Consider creating a separate private group for alerts")

        print()
        print("✅ Setup complete! Update your .env file with the IDs above.")

    except FileNotFoundError:
        print("❌ Session file not found!")
        print("💡 Run 'python scripts/auth.py' first to authenticate with Telegram")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await client.disconnect()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())