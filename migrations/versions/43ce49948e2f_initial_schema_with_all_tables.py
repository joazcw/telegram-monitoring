"""Initial schema with all tables

Revision ID: 43ce49948e2f
Revises: 
Create Date: 2025-09-27 10:19:30.837754

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '43ce49948e2f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create telegram_messages table
    op.create_table('telegram_messages',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('chat_id', sa.BigInteger(), nullable=False),
        sa.Column('message_id', sa.BigInteger(), nullable=False),
        sa.Column('sender_id', sa.BigInteger()),
        sa.Column('text', sa.Text()),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('has_media', sa.Boolean(), nullable=True),
        sa.Column('processed', sa.Boolean(), nullable=True)
    )

    # Create indexes for telegram_messages
    op.create_index('ix_chat_message', 'telegram_messages', ['chat_id', 'message_id'], unique=True)
    op.create_index('ix_timestamp', 'telegram_messages', ['timestamp'])
    op.create_index('ix_processed_media', 'telegram_messages', ['processed', 'has_media'])
    op.create_index('ix_chat_recent', 'telegram_messages', ['chat_id', 'timestamp'])

    # Create images table
    op.create_table('images',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('message_id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(512), nullable=False),
        sa.Column('sha256_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('file_size', sa.Integer()),
        sa.Column('processed', sa.Boolean(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True)
    )

    # Create indexes for images
    op.create_index('ix_unprocessed', 'images', ['processed', 'timestamp'])
    op.create_index('ix_message_images', 'images', ['message_id'])
    op.create_index('ix_file_size', 'images', ['file_size'])

    # Create ocr_text table
    op.create_table('ocr_text',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('extracted_text', sa.Text()),
        sa.Column('confidence', sa.Integer()),
        sa.Column('processing_time', sa.Integer()),
        sa.Column('timestamp', sa.DateTime(), nullable=True)
    )

    # Create indexes for ocr_text
    op.create_index('ix_image_ocr', 'ocr_text', ['image_id'])
    op.create_index('ix_confidence', 'ocr_text', ['confidence'])

    # Create brand_hits table
    op.create_table('brand_hits',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('message_id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer()),
        sa.Column('brand_name', sa.String(100), nullable=False),
        sa.Column('matched_text', sa.String(500)),
        sa.Column('confidence_score', sa.Integer()),
        sa.Column('match_type', sa.String(20), nullable=True),
        sa.Column('alert_sent', sa.Boolean(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True)
    )

    # Create indexes for brand_hits
    op.create_index('ix_brand_recent', 'brand_hits', ['brand_name', 'timestamp'])
    op.create_index('ix_unsent_alerts', 'brand_hits', ['alert_sent', 'timestamp'])
    op.create_index('ix_confidence_score', 'brand_hits', ['confidence_score'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('brand_hits')
    op.drop_table('ocr_text')
    op.drop_table('images')
    op.drop_table('telegram_messages')
