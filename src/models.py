from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class TelegramMessage(Base):
    __tablename__ = 'telegram_messages'

    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, nullable=False)
    message_id = Column(BigInteger, nullable=False)
    sender_id = Column(BigInteger)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    has_media = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)

    __table_args__ = (
        Index('ix_chat_message', 'chat_id', 'message_id', unique=True),
        Index('ix_timestamp', 'timestamp'),
        Index('ix_processed_media', 'processed', 'has_media'),
        Index('ix_chat_recent', 'chat_id', 'timestamp'),
    )

    def __repr__(self):
        return f"<TelegramMessage(id={self.id}, chat_id={self.chat_id}, message_id={self.message_id})>"


class ImageRecord(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    sha256_hash = Column(String(64), nullable=False, unique=True)
    file_size = Column(Integer)
    processed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_unprocessed', 'processed', 'timestamp'),
        Index('ix_message_images', 'message_id'),
        Index('ix_file_size', 'file_size'),
    )

    def __repr__(self):
        return f"<ImageRecord(id={self.id}, message_id={self.message_id}, file_path='{self.file_path}')>"


class OCRText(Base):
    __tablename__ = 'ocr_text'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    extracted_text = Column(Text)
    confidence = Column(Integer)  # 0-100
    processing_time = Column(Integer)  # milliseconds
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_image_ocr', 'image_id'),
        Index('ix_confidence', 'confidence'),
    )

    def __repr__(self):
        return f"<OCRText(id={self.id}, image_id={self.image_id}, confidence={self.confidence})>"


class BrandHit(Base):
    __tablename__ = 'brand_hits'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=False)
    image_id = Column(Integer)
    brand_name = Column(String(100), nullable=False)
    matched_text = Column(String(500))
    confidence_score = Column(Integer)  # 0-100
    match_type = Column(String(20), default='fuzzy')  # 'exact', 'fuzzy'
    alert_sent = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_brand_recent', 'brand_name', 'timestamp'),
        Index('ix_unsent_alerts', 'alert_sent', 'timestamp'),
        Index('ix_confidence_score', 'confidence_score'),
    )

    def __repr__(self):
        return f"<BrandHit(id={self.id}, brand_name='{self.brand_name}', confidence_score={self.confidence_score})>"