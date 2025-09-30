import os
import hashlib
from telethon import TelegramClient, events
from sqlalchemy.exc import IntegrityError
from src.config import (
    TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_PATH,
    TELEGRAM_GROUPS, MEDIA_DIR, MAX_MEDIA_SIZE
)
from src.database import SessionLocal
from src.models import TelegramMessage, ImageRecord
import logging
import aiofiles

logger = logging.getLogger(__name__)


class TelegramCollector:
    def __init__(self):
        self.groups = TELEGRAM_GROUPS
        self.media_dir = MEDIA_DIR
        self.max_media_size = MAX_MEDIA_SIZE
        self.session_authenticated = False
        self.client = None

        # Check if credentials are configured
        self.credentials_available = bool(TELEGRAM_API_ID and TELEGRAM_API_HASH)

        if self.credentials_available:
            self.client = TelegramClient(
                TELEGRAM_SESSION_PATH,
                TELEGRAM_API_ID,
                TELEGRAM_API_HASH
            )
        else:
            logger.warning("Telegram API credentials not configured. Client functionality will be limited.")

        # Ensure media directory exists
        os.makedirs(self.media_dir, exist_ok=True)
        logger.info(f"Initialized TelegramCollector for {len(self.groups)} groups")

    async def start(self):
        """Start the Telegram client and message collection"""
        if not self.client:
            logger.error("Cannot start: Telegram client not initialized (missing credentials)")
            raise Exception("Telegram API credentials not configured")

        try:
            await self.client.start()
            self.session_authenticated = await self.client.is_user_authorized()

            if not self.session_authenticated:
                logger.error("Telegram session not authenticated. Run authentication first.")
                raise Exception("Not authenticated with Telegram")

            me = await self.client.get_me()
            logger.info(f"Connected to Telegram as: {me.first_name} (@{me.username or 'no_username'})")

            # Validate and resolve group access
            valid_groups = await self._validate_groups()
            if not valid_groups:
                logger.error("No accessible groups found. Check your TELEGRAM_GROUPS configuration.")
                raise Exception("No accessible groups configured")

            # Register message handler with validated groups
            @self.client.on(events.NewMessage(chats=valid_groups))
            async def message_handler(event):
                await self._process_message(event)

            logger.info(f"Listening for messages in {len(valid_groups)} groups...")
            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"Failed to start Telegram client: {e}")
            raise

    async def _validate_groups(self):
        """Validate and return list of accessible groups"""
        valid_groups = []

        for group in self.groups:
            try:
                # Try to resolve the group/chat
                entity = await self.client.get_input_entity(group)
                valid_groups.append(group)
                logger.info(f"✅ Group accessible: {group}")

            except Exception as e:
                logger.warning(f"❌ Cannot access group '{group}': {e}")
                logger.warning(f"   Skipping {group}. Make sure:")
                logger.warning(f"   - Group exists and username is correct")
                logger.warning(f"   - You are a member of the group")
                logger.warning(f"   - For private groups, use numeric ID instead of username")

        if not valid_groups:
            logger.error("No groups are accessible. Common issues:")
            logger.error("1. Group usernames don't exist (check for typos)")
            logger.error("2. You're not a member of the groups")
            logger.error("3. Groups are private (need numeric IDs instead)")
            logger.error("4. Using example values from README.md")
            logger.error("Run 'python debug_messages.py' to see accessible groups")

        return valid_groups

    async def _process_message(self, event):
        """Process incoming message and store in database"""
        db = SessionLocal()
        try:
            # Extract message data
            message_data = {
                'chat_id': event.chat_id,
                'message_id': event.message.id,
                'sender_id': event.sender_id,
                'text': event.message.text or "",
                'has_media': bool(event.message.media),
                'processed': False
            }

            # Create message record
            message = TelegramMessage(**message_data)
            db.add(message)
            db.flush()  # Get the ID

            logger.info(f"Processing message {event.message.id} from chat {event.chat_id}")

            # Download and store media if present
            if event.message.media:
                await self._download_media(event.message, message.id, db)

            db.commit()
            logger.debug(f"Successfully stored message {event.message.id}")

        except IntegrityError:
            logger.debug(f"Message {event.message.id} already exists, skipping")
            db.rollback()
        except Exception as e:
            logger.error(f"Error processing message {event.message.id}: {e}")
            db.rollback()
            # Don't re-raise to keep collector running
        finally:
            db.close()

    async def _download_media(self, message, db_message_id: int, db):
        """Download media file and create image record"""
        file_path = None
        try:
            # Check file size before download
            if hasattr(message.media, 'document') and message.media.document:
                if message.media.document.size > self.max_media_size:
                    logger.warning(f"Media file too large ({message.media.document.size} bytes), skipping")
                    return

            # Download media file
            file_path = await message.download_media(file=self.media_dir)
            if not file_path:
                logger.warning("Failed to download media file")
                return

            # Calculate file hash for deduplication
            sha256_hash = await self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)

            # Create image record
            image_record = ImageRecord(
                message_id=db_message_id,
                file_path=file_path,
                sha256_hash=sha256_hash,
                file_size=file_size,
                processed=False
            )
            db.add(image_record)
            logger.info(f"Downloaded media: {file_path} ({file_size} bytes)")

        except IntegrityError:
            logger.debug(f"Media file with hash already exists, removing duplicate")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            db.rollback()
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            raise

    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()

        try:
            # Use synchronous file reading for compatibility
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Telegram client is healthy"""
        try:
            if not self.client:
                logger.warning("Health check: Telegram client not initialized")
                return False

            if not self.session_authenticated:
                return False

            if not self.client.is_connected():
                return False

            # Test API call
            me = await self.client.get_me()
            return me is not None

        except Exception as e:
            logger.error(f"Telegram health check failed: {e}")
            return False

    async def stop(self):
        """Gracefully stop the collector"""
        if self.client and self.client.is_connected():
            await self.client.disconnect()
        logger.info("Telegram collector stopped")

    async def authenticate_session(self):
        """Interactive authentication for Telegram session"""
        try:
            await self.client.start()

            if not await self.client.is_user_authorized():
                phone = input("Please enter your phone number: ")
                await self.client.send_code_request(phone)

                code = input("Please enter the code you received: ")
                await self.client.sign_in(phone, code)

                logger.info("Successfully authenticated with Telegram")
            else:
                logger.info("Already authenticated with Telegram")

            me = await self.client.get_me()
            logger.info(f"Connected as: {me.first_name} (@{me.username or 'no_username'})")

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    async def get_group_info(self):
        """Get information about configured groups"""
        if not self.session_authenticated:
            logger.error("Not authenticated. Cannot get group info.")
            return []

        group_info = []
        for group_identifier in self.groups:
            try:
                entity = await self.client.get_entity(group_identifier)
                info = {
                    'id': entity.id,
                    'title': getattr(entity, 'title', 'Unknown'),
                    'username': getattr(entity, 'username', None),
                    'type': entity.__class__.__name__
                }
                group_info.append(info)
                logger.info(f"Group: {info['title']} (ID: {info['id']})")
            except Exception as e:
                logger.error(f"Could not get info for group {group_identifier}: {e}")

        return group_info

    def get_stats(self) -> dict:
        """Get collector statistics"""
        base_stats = {
            'configured_groups': len(self.groups),
            'authenticated': self.session_authenticated,
            'connected': (self.client.is_connected() if self.client and hasattr(self.client, 'is_connected') else False),
            'credentials_configured': self.credentials_available
        }

        try:
            db = SessionLocal()
            message_count = db.query(TelegramMessage).count()
            image_count = db.query(ImageRecord).count()
            processed_messages = db.query(TelegramMessage).filter(TelegramMessage.processed == True).count()

            base_stats.update({
                'total_messages': message_count,
                'total_images': image_count,
                'processed_messages': processed_messages,
                'unprocessed_messages': message_count - processed_messages,
                'database_available': True
            })
            db.close()
        except Exception as e:
            logger.warning(f"Cannot get database stats: {e}")
            base_stats.update({
                'total_messages': 'N/A (DB unavailable)',
                'total_images': 'N/A (DB unavailable)',
                'processed_messages': 'N/A (DB unavailable)',
                'unprocessed_messages': 'N/A (DB unavailable)',
                'database_available': False
            })

        return base_stats