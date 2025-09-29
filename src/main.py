import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from src.telegram_collector import TelegramCollector
from src.processor import FraudProcessor
from src.database import init_db, check_db_connection
from src.config import LOG_LEVEL, LOG_FILE

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce SQLAlchemy log noise
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class FraudMonitorApp:
    def __init__(self):
        self.collector = None
        self.processor = None
        self.shutdown_event = asyncio.Event()

    async def startup(self):
        """Initialize application components"""
        logger.info("🚀 Starting Fraud Monitoring System")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")

        # Initialize database
        logger.info("Initializing database...")
        try:
            if not check_db_connection():
                logger.warning("Database connection failed, continuing in limited mode")
            else:
                init_db()
                logger.info("✅ Database initialized")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            logger.warning("Continuing in limited mode without database")

        # Initialize components
        self.collector = TelegramCollector()
        self.processor = FraudProcessor(telegram_client=self.collector.client)
        logger.info("✅ Components initialized")

    async def run(self):
        """Run the main application"""
        try:
            await self.startup()

            # Setup signal handlers for graceful shutdown
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)

            logger.info("Starting concurrent services...")

            # Run collector and processor concurrently
            tasks = await asyncio.gather(
                self.collector.start(),
                self.processor.start(),
                self._monitor_health(),
                return_exceptions=True
            )

            # Check if any task failed
            for i, task in enumerate(tasks):
                if isinstance(task, Exception):
                    service_name = ['collector', 'processor', 'monitor'][i]
                    logger.error(f"Service {service_name} failed: {task}")

        except Exception as e:
            logger.error(f"Application failed to start: {e}")
            raise
        finally:
            await self.shutdown()

    async def _monitor_health(self):
        """Monitor system health and log statistics"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if self.processor:
                    stats = self.processor.get_stats()
                    logger.info(f"📊 Processing Stats: {stats}")

                # Check component health
                if self.collector and not await self.collector.health_check():
                    logger.error("❌ Telegram collector health check failed")

                if self.processor and not await self.processor._health_check():
                    logger.error("❌ Processor health check failed")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🛑 Shutting down Fraud Monitoring System...")

        # Signal shutdown
        self.shutdown_event.set()

        # Stop components
        if self.processor:
            await self.processor.stop()

        if self.collector:
            await self.collector.stop()

        logger.info("✅ Shutdown complete")


async def main():
    """Main application entry point"""
    setup_logging()
    app = FraudMonitorApp()

    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())