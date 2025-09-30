import os
from datetime import datetime, timedelta
from telethon import TelegramClient
from src.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_ALERT_CHAT_ID, ALERT_RATE_LIMIT, TELEGRAM_SESSION_PATH
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AlertSender:
    def __init__(self, telegram_client=None):
        self.alert_chat_id = TELEGRAM_ALERT_CHAT_ID
        self.rate_limit = ALERT_RATE_LIMIT
        self.alert_history = []  # Track recent alerts for rate limiting
        self.client = telegram_client  # Use shared client if provided
        self.own_client = False  # Track if we own the client
        self.credentials_available = bool(TELEGRAM_API_ID and TELEGRAM_API_HASH)

        # If no client provided, create our own (for testing/standalone usage)
        if not self.client:
            if self.credentials_available:
                # Use a separate session file to avoid SQLite locking conflicts
                alert_session_path = TELEGRAM_SESSION_PATH.replace('.session', '_alerts.session')
                self.client = TelegramClient(alert_session_path, TELEGRAM_API_ID, TELEGRAM_API_HASH)
                self.own_client = True
            else:
                logger.warning("Telegram API credentials not configured. Alert functionality will be limited.")

        logger.info(f"Initialized AlertSender for chat {self.alert_chat_id}")

    async def send_brand_alert(self, alert_data: Dict) -> bool:
        """Send formatted alert about brand mention"""
        if not self.client:
            logger.warning("Cannot send brand alert: Telegram client not initialized")
            return False

        try:
            # Check rate limiting
            if not self._check_rate_limit():
                logger.warning("Alert rate limit exceeded, queuing alert")
                return False

            # Connect if not already connected (only if we own the client)
            if not self.client.is_connected() and self.own_client:
                await self.client.start()

            # Format and send alert
            alert_text = self._format_alert(alert_data)
            await self.client.send_message(
                self.alert_chat_id,
                alert_text,
                parse_mode='html'
            )

            # Track alert for rate limiting
            self._record_alert()
            logger.info(f"Alert sent for brand: {alert_data.get('brand', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    async def send_system_alert(self, severity: str, message: str) -> bool:
        """Send system status alert"""
        if not self.client:
            logger.warning("Cannot send system alert: Telegram client not initialized")
            return False

        try:
            if not self._check_rate_limit():
                logger.warning("System alert rate limit exceeded")
                return False

            if not self.client.is_connected() and self.own_client:
                await self.client.start()

            severity_emoji = {
                'INFO': '💡',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }

            alert_text = f"""
{severity_emoji.get(severity, '📢')} <b>System Alert - {severity}</b>

<b>Message:</b> {message}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
<b>System:</b> Fraud Monitoring Bot

<i>Automated system notification</i>
            """.strip()

            await self.client.send_message(
                self.alert_chat_id,
                alert_text,
                parse_mode='html'
            )

            self._record_alert()
            logger.info(f"System alert sent: {severity} - {message}")
            return True

        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
            return False

    def _format_alert(self, data: Dict) -> str:
        """Format brand detection alert message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        # Build context information
        context_parts = []
        if data.get('extracted_text'):
            text_preview = data['extracted_text'][:100]
            if len(data['extracted_text']) > 100:
                text_preview += "..."
            context_parts.append(f"<b>OCR Text:</b> \"{text_preview}\"")

        if data.get('image_path'):
            context_parts.append(f"<b>Image:</b> {os.path.basename(data['image_path'])}")

        context_section = '\n'.join(context_parts) if context_parts else ""

        alert_text = f"""
🚨 <b>Brand Mention Detected</b>

<b>Brand:</b> {data.get('brand', 'Unknown')}
<b>Matched Text:</b> "{data.get('matched_text', '')}"
<b>Confidence:</b> {data.get('confidence', 0)}%
<b>Match Type:</b> {data.get('match_type', 'unknown').title()}
<b>Source Chat:</b> {data.get('chat_id', 'Unknown')}
<b>Message ID:</b> {data.get('message_id', 'Unknown')}
<b>Detection Time:</b> {timestamp}

{context_section}

<i>🤖 Automated fraud detection alert</i>
        """.strip()

        return alert_text

    def _check_rate_limit(self) -> bool:
        """Check if we can send another alert based on rate limiting"""
        now = datetime.now()

        # Remove alerts older than 1 minute
        self.alert_history = [
            alert_time for alert_time in self.alert_history
            if now - alert_time < timedelta(minutes=1)
        ]

        return len(self.alert_history) < self.rate_limit

    def _record_alert(self):
        """Record that an alert was sent"""
        self.alert_history.append(datetime.now())

    async def health_check(self) -> bool:
        """Check if alert system is working"""
        if not self.client:
            logger.warning("Health check: Telegram client not initialized")
            return False

        try:
            # Just check if client exists and credentials are available
            # Don't try to connect during health check to avoid interactive prompts
            test_data = {
                'brand': 'TEST',
                'matched_text': 'Health Check',
                'confidence': 100,
                'match_type': 'test',
                'chat_id': 'HEALTH_CHECK',
                'message_id': 'TEST_MSG_001'
            }

            # Just validate formatting without connecting
            test_alert = self._format_alert(test_data)
            health_ok = len(test_alert) > 0 and self.credentials_available

            if health_ok:
                logger.info("Alert system health check: formatting OK, credentials available")

            return health_ok

        except Exception as e:
            logger.error(f"Alert system health check failed: {e}")
            return False

    async def stop(self):
        """Gracefully stop the alert sender"""
        if self.client and self.client.is_connected() and self.own_client:
            await self.client.disconnect()
        logger.info("Alert sender stopped")

    def get_stats(self) -> dict:
        """Get basic alert system status"""
        return {
            'credentials_configured': self.credentials_available,
            'chat_configured': bool(self.alert_chat_id and self.alert_chat_id != 0),
            'connected': (self.client.is_connected() if self.client and hasattr(self.client, 'is_connected') else False)
        }

    async def send_test_alert(self) -> bool:
        """Send a test alert for validation"""
        test_data = {
            'brand': 'CloudWalk',
            'matched_text': 'CloudWalk Test Alert',
            'confidence': 95,
            'match_type': 'test',
            'chat_id': 'TEST_CHAT',
            'message_id': 'TEST_001',
            'extracted_text': 'This is a test alert to verify the fraud monitoring system is working correctly.'
        }

        return await self.send_brand_alert(test_data)