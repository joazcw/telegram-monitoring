import asyncio
import time
from datetime import datetime
from src.ocr_engine import OCREngine
from src.brand_matcher import BrandMatcher
from src.alert_sender import AlertSender
from src.database import SessionLocal
from src.models import ImageRecord, OCRText, BrandHit, TelegramMessage
from src.config import BATCH_SIZE, PROCESSING_INTERVAL, RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_FACTOR
import logging

logger = logging.getLogger(__name__)


class FraudProcessor:
    def __init__(self, telegram_client=None):
        self.ocr_engine = OCREngine()
        self.brand_matcher = BrandMatcher()
        self.alert_sender = AlertSender(telegram_client=telegram_client)
        self.batch_size = BATCH_SIZE
        self.processing_interval = PROCESSING_INTERVAL
        self.max_retries = RETRY_MAX_ATTEMPTS
        self.backoff_factor = RETRY_BACKOFF_FACTOR
        self.running = False
        self.stats = {
            'processed_images': 0,
            'brand_hits': 0,
            'alerts_sent': 0,
            'processing_errors': 0,
            'start_time': None
        }
        logger.info("Initialized FraudProcessor")

    async def start(self):
        """Start the main processing loop"""
        self.running = True
        self.stats['start_time'] = datetime.now()
        logger.info("Starting fraud processor...")

        # Run health checks
        if not await self._health_check():
            raise Exception("Health checks failed, cannot start processor")

        try:
            while self.running:
                await self._process_batch()
                await asyncio.sleep(self.processing_interval)
        except KeyboardInterrupt:
            logger.info("Received stop signal")
        except Exception as e:
            logger.error(f"Fatal error in processor: {e}")
            raise
        finally:
            await self.stop()

    async def _process_batch(self):
        """Process a batch of unprocessed images and text messages"""
        db = SessionLocal()
        try:
            # Process unprocessed images
            pending_images = db.query(ImageRecord).filter(
                ImageRecord.processed == False
            ).order_by(ImageRecord.timestamp).limit(self.batch_size).all()

            if pending_images:
                logger.info(f"Processing batch of {len(pending_images)} images")
                for image in pending_images:
                    try:
                        await self._process_single_image(image, db)
                    except Exception as e:
                        logger.error(f"Error processing image {image.id}: {e}")
                        self.stats['processing_errors'] += 1

            # Process unprocessed text messages
            pending_messages = db.query(TelegramMessage).filter(
                TelegramMessage.processed == False,
                TelegramMessage.text.isnot(None),
                TelegramMessage.text != ''
            ).order_by(TelegramMessage.timestamp).limit(self.batch_size).all()

            if pending_messages:
                logger.info(f"Processing batch of {len(pending_messages)} text messages")
                for message in pending_messages:
                    try:
                        await self._process_single_text_message(message, db)
                    except Exception as e:
                        logger.error(f"Error processing message {message.id}: {e}")
                        self.stats['processing_errors'] += 1

            if not pending_images and not pending_messages:
                logger.debug("No pending items to process")

            db.commit()

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            db.rollback()
        finally:
            db.close()

    async def _process_single_image(self, image: ImageRecord, db):
        """Process single image: OCR -> Brand Matching -> Alerting"""
        start_time = time.time()
        try:
            logger.debug(f"Processing image {image.id}: {image.file_path}")

            # Step 1: OCR Text Extraction
            extracted_text, confidence, ocr_time = self.ocr_engine.extract_text(image.file_path)

            # Store OCR result
            ocr_record = OCRText(
                image_id=image.id,
                extracted_text=extracted_text,
                confidence=confidence,
                processing_time=ocr_time
            )
            db.add(ocr_record)
            db.flush()

            logger.debug(f"OCR extracted {len(extracted_text)} characters with {confidence}% confidence")

            # Step 2: Brand Matching (only if we have text)
            brand_matches = []
            if extracted_text.strip():
                brand_matches = self.brand_matcher.find_matches(extracted_text)
                logger.debug(f"Found {len(brand_matches)} potential brand matches")

            # Step 3: Store Brand Hits and Send Alerts
            for match in brand_matches:
                brand_hit = BrandHit(
                    message_id=image.message_id,
                    image_id=image.id,
                    brand_name=match['brand'],
                    matched_text=match['matched_text'],
                    confidence_score=match['confidence'],
                    match_type=match['match_type'],
                    alert_sent=False
                )
                db.add(brand_hit)
                db.flush()

                # Send alert
                alert_sent = await self._send_alert(brand_hit, image, extracted_text, db)
                brand_hit.alert_sent = alert_sent

                if alert_sent:
                    self.stats['alerts_sent'] += 1

                self.stats['brand_hits'] += 1
                logger.info(f"Brand hit: {match['brand']} (confidence: {match['confidence']}%)")

            # Mark image as processed
            image.processed = True
            self.stats['processed_images'] += 1

            processing_time = time.time() - start_time
            logger.debug(f"Completed processing image {image.id} in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to process image {image.id}: {e}")
            # Don't mark as processed so it can be retried
            raise

    async def _process_single_text_message(self, message: TelegramMessage, db):
        """Process single text message: Brand Matching -> Alerting"""
        start_time = time.time()
        try:
            # Validate message text
            if not message.text or message.text.strip() == '':
                logger.debug(f"Skipping empty text message {message.id}")
                message.processed = True
                return

            logger.debug(f"Processing text message {message.id}: {message.text[:50]}...")

            # Step 1: Brand Matching on message text
            matches = self.brand_matcher.find_matches(message.text)

            for match in matches:
                # Create brand hit record
                brand_hit = BrandHit(
                    message_id=message.id,
                    brand_name=match['brand'],
                    matched_text=match['matched_text'],
                    confidence_score=match['confidence'],
                    match_type=match['match_type']
                )
                db.add(brand_hit)
                db.flush()

                # Send alert
                alert_sent = await self._send_text_alert(brand_hit, message, db)
                brand_hit.alert_sent = alert_sent

                if alert_sent:
                    self.stats['alerts_sent'] += 1

                self.stats['brand_hits'] += 1
                logger.info(f"Brand hit in text: {match['brand']} (confidence: {match['confidence']}%)")

            # Mark message as processed
            message.processed = True
            self.stats['processed_messages'] = self.stats.get('processed_messages', 0) + 1

            processing_time = time.time() - start_time
            logger.debug(f"Completed processing text message {message.id} in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to process text message {message.id}: {e}")
            # Don't mark as processed so it can be retried
            raise

    async def _send_alert(self, hit: BrandHit, image: ImageRecord, extracted_text: str, db) -> bool:
        """Send alert for brand hit"""
        try:
            # Get message context
            message = db.query(TelegramMessage).filter(
                TelegramMessage.id == hit.message_id
            ).first()

            alert_data = {
                'brand': hit.brand_name,
                'matched_text': hit.matched_text,
                'confidence': hit.confidence_score,
                'match_type': hit.match_type,
                'chat_id': message.chat_id if message else 'Unknown',
                'message_id': message.message_id if message else 'Unknown',
                'extracted_text': extracted_text,
                'image_path': image.file_path
            }

            success = await self.alert_sender.send_brand_alert(alert_data)
            if success:
                logger.info(f"Alert sent for brand hit: {hit.brand_name}")
            else:
                logger.warning(f"Failed to send alert for brand hit: {hit.brand_name}")

            return success

        except Exception as e:
            logger.error(f"Error sending alert for brand hit {hit.id}: {e}")
            return False

    async def _send_text_alert(self, hit: BrandHit, message: TelegramMessage, db) -> bool:
        """Send alert for text message brand hit"""
        try:
            alert_data = {
                'brand': hit.brand_name,
                'matched_text': hit.matched_text,
                'confidence': hit.confidence_score,
                'match_type': hit.match_type,
                'chat_id': message.chat_id,
                'message_id': message.message_id,
                'extracted_text': message.text[:200] + '...' if len(message.text) > 200 else message.text,
                'message_type': 'text'
            }

            success = await self.alert_sender.send_brand_alert(alert_data)
            if success:
                logger.info(f"Text alert sent for brand hit: {hit.brand_name}")
            else:
                logger.warning(f"Failed to send text alert for brand hit: {hit.brand_name}")

            return success

        except Exception as e:
            logger.error(f"Error sending text alert for brand hit {hit.id}: {e}")
            return False

    async def _health_check(self) -> bool:
        """Comprehensive health check of all components"""
        logger.info("Running health checks...")

        checks = {
            'ocr_engine': self.ocr_engine.health_check(),
            'brand_matcher': self.brand_matcher.health_check(),
            'alert_sender': await self.alert_sender.health_check()
        }

        all_healthy = all(checks.values())

        for component, healthy in checks.items():
            status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
            logger.info(f"{component}: {status}")

        return all_healthy

    def get_stats(self) -> dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime'] = str(datetime.now() - stats['start_time'])

        return stats

    async def stop(self):
        """Gracefully stop the processor"""
        self.running = False
        await self.alert_sender.stop()

        # Send final stats
        stats = self.get_stats()
        logger.info(f"Processor stopped. Final stats: {stats}")

    async def process_single_image_by_id(self, image_id: int) -> bool:
        """Process a single image by ID (useful for testing)"""
        db = SessionLocal()
        try:
            image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            if not image:
                logger.error(f"Image with ID {image_id} not found")
                return False

            await self._process_single_image(image, db)
            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def get_pending_images_count(self) -> int:
        """Get count of pending images"""
        try:
            db = SessionLocal()
            count = db.query(ImageRecord).filter(ImageRecord.processed == False).count()
            db.close()
            return count
        except Exception as e:
            logger.error(f"Error getting pending images count: {e}")
            return 0

    async def send_system_alert(self, severity: str, message: str) -> bool:
        """Send system alert (proxy to alert sender)"""
        return await self.alert_sender.send_system_alert(severity, message)