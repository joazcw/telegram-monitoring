#!/usr/bin/env python3
"""
Debug script to check messages in database
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database import SessionLocal
from src.models import TelegramMessage

def main():
    db = SessionLocal()
    try:
        messages = db.query(TelegramMessage).filter(
            TelegramMessage.processed == False
        ).limit(5).all()

        print(f"Found {len(messages)} unprocessed messages:")
        for msg in messages:
            print(f"ID: {msg.id}")
            print(f"Message ID: {msg.message_id}")
            print(f"Chat ID: {msg.chat_id}")
            print(f"Text: {repr(msg.text)}")  # Using repr to show exact content
            print(f"Text type: {type(msg.text)}")
            print(f"Has media: {msg.has_media}")
            print(f"Processed: {msg.processed}")
            print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()