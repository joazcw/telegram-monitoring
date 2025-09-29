# ADR-0002: Data Model and Indexes

**Status**: Accepted
**Date**: 2025-09-27
**Decision Makers**: Platform Engineering Team
**Stakeholders**: Security, Operations, Data Engineering

## Context

The fraud monitoring system needs a database schema that supports:

- **High-frequency writes**: 1000+ messages/day with media attachments
- **Fast lookups**: Query unprocessed images, brand hits by time range
- **Deduplication**: Prevent duplicate message and media processing
- **Audit trail**: Track processing status and alert delivery
- **Data retention**: Support automated cleanup of old records

Key performance requirements:
- Message ingestion: <100ms per record
- Image processing queries: <1 second
- Brand hit searches: <2 seconds for 30-day range
- Dashboard queries: <5 seconds for aggregations

## Decision

We will implement a **4-table normalized schema** with strategic indexing:

### Core Tables

#### telegram_messages
Primary entity for all Telegram message data.

```sql
CREATE TABLE telegram_messages (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    sender_id BIGINT,
    text TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    has_media BOOLEAN DEFAULT FALSE,
    processed BOOLEAN DEFAULT FALSE,

    -- Indexes
    CONSTRAINT uk_chat_message UNIQUE (chat_id, message_id),
    INDEX idx_timestamp (timestamp DESC),
    INDEX idx_processed_media (processed, has_media) WHERE has_media = TRUE,
    INDEX idx_chat_recent (chat_id, timestamp DESC)
);
```

#### images
Media file metadata with deduplication support.

```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES telegram_messages(id) ON DELETE CASCADE,
    file_path VARCHAR(512) NOT NULL,
    sha256_hash CHAR(64) NOT NULL,
    file_size INTEGER,
    processed BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Deduplication and processing indexes
    CONSTRAINT uk_sha256 UNIQUE (sha256_hash),
    INDEX idx_unprocessed (processed, timestamp) WHERE processed = FALSE,
    INDEX idx_message_images (message_id),
    INDEX idx_file_size (file_size) -- For cleanup queries
);
```

#### ocr_text
OCR extraction results with confidence scoring.

```sql
CREATE TABLE ocr_text (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    extracted_text TEXT,
    confidence INTEGER, -- 0-100 scale
    processing_time INTEGER, -- milliseconds
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Full-text search and quality filtering
    INDEX idx_image_ocr (image_id),
    INDEX idx_confidence (confidence DESC),
    INDEX gin_fulltext ON extracted_text USING GIN (to_tsvector('english', extracted_text))
);
```

#### brand_hits
Detected brand mentions with fuzzy matching details.

```sql
CREATE TABLE brand_hits (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES telegram_messages(id) ON DELETE CASCADE,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    brand_name VARCHAR(100) NOT NULL,
    matched_text VARCHAR(500),
    confidence_score INTEGER, -- 0-100 scale
    match_type VARCHAR(20) DEFAULT 'fuzzy', -- 'exact', 'fuzzy'
    alert_sent BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Analytics and alerting indexes
    INDEX idx_brand_recent (brand_name, timestamp DESC),
    INDEX idx_unsent_alerts (alert_sent, timestamp) WHERE alert_sent = FALSE,
    INDEX idx_confidence (confidence_score DESC),
    INDEX idx_match_type (match_type, brand_name)
);
```

## Options Considered

### Alternative Schema Designs

#### Single Denormalized Table
- **Pros**: Simple queries, no joins, faster writes
- **Cons**: Data duplication, large row size, harder to maintain
- **Verdict**: Rejected - Storage inefficient for images with multiple OCR attempts

#### Document Store Approach (JSONB columns)
- **Pros**: Flexible schema, good PostgreSQL JSONB support
- **Cons**: Harder to index specific fields, complex queries
- **Verdict**: Rejected - Structured data fits relational model better

#### Event Sourcing Pattern
- **Pros**: Complete audit trail, replay capability, immutable records
- **Cons**: Query complexity, storage overhead, harder debugging
- **Verdict**: Rejected - Overkill for current requirements

### Alternative Indexing Strategies

#### Composite Indexes vs Single-Column
**Decision**: Use composite indexes for common query patterns
```sql
-- Chosen: Composite for processing queries
INDEX idx_processed_media (processed, has_media) WHERE has_media = TRUE

-- Rejected: Separate indexes (less efficient)
INDEX idx_processed (processed)
INDEX idx_has_media (has_media)
```

#### B-Tree vs Hash Indexes
**Decision**: B-Tree for all indexes (PostgreSQL default)
- Hash indexes don't support range queries
- B-tree handles equality efficiently enough
- Better WAL logging and replication support

#### Partial vs Full Indexes
**Decision**: Use partial indexes for boolean filters
```sql
-- Efficient: Only indexes unprocessed records
INDEX idx_unprocessed (timestamp) WHERE processed = FALSE

-- Wasteful: Indexes all records including processed
INDEX idx_timestamp_full (processed, timestamp)
```

### Alternative Primary Key Strategies

#### UUID vs Serial Integer
**Decision**: Serial integer for performance
- **UUID Pros**: Globally unique, no collision risk
- **UUID Cons**: 36 bytes vs 4 bytes, slower joins, random insertion
- **Verdict**: Serial chosen - simpler, faster, sufficient for single DB

#### Composite Natural Keys
**Decision**: Surrogate keys with unique constraints
```sql
-- Chosen: Surrogate PK + unique constraint
id SERIAL PRIMARY KEY,
CONSTRAINT uk_chat_message UNIQUE (chat_id, message_id)

-- Rejected: Composite natural PK (complex foreign keys)
PRIMARY KEY (chat_id, message_id)
```

## Data Integrity Constraints

### Deduplication Strategy
- **Messages**: Unique constraint on `(chat_id, message_id)`
- **Images**: Unique constraint on `sha256_hash`
- **Processing**: Boolean flags prevent reprocessing

### Referential Integrity
- **CASCADE DELETE**: Child records deleted when parent removed
- **Foreign Keys**: Enforce relationships, prevent orphaned records
- **NOT NULL**: Required fields identified and enforced

### Data Validation
```sql
-- Ensure valid confidence scores
ALTER TABLE ocr_text ADD CONSTRAINT chk_confidence
    CHECK (confidence >= 0 AND confidence <= 100);

-- Ensure positive file sizes
ALTER TABLE images ADD CONSTRAINT chk_file_size
    CHECK (file_size > 0);

-- Ensure valid timestamps
ALTER TABLE telegram_messages ADD CONSTRAINT chk_timestamp
    CHECK (timestamp <= NOW() + INTERVAL '1 hour');
```

## Index Maintenance Strategy

### Monitoring Index Usage
```sql
-- Query to identify unused indexes
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_tup_read = 0 AND idx_tup_fetch = 0;

-- Query to identify missing indexes (slow queries)
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
WHERE mean_time > 1000  -- >1 second average
ORDER BY mean_time DESC;
```

### Index Maintenance Schedule
- **Daily**: Update table statistics (`ANALYZE`)
- **Weekly**: Reindex frequently updated tables
- **Monthly**: Review index usage and remove unused indexes
- **Quarterly**: Full database vacuum and analyze

### Growth Management
```sql
-- Partition large tables by month (future consideration)
CREATE TABLE brand_hits_202509 PARTITION OF brand_hits
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- Archive old data before partitioning becomes necessary
DELETE FROM telegram_messages
WHERE timestamp < NOW() - INTERVAL '90 days';
```

## Migration and Evolution Strategy

### Schema Versioning
- **Alembic migrations**: All schema changes tracked
- **Backward compatibility**: Support rolling deployments
- **Data migrations**: Separate from schema changes

### Example Migration Pattern
```python
# Migration: Add perceptual hash for image deduplication
def upgrade():
    # Add column with default
    op.add_column('images', sa.Column('phash', sa.String(16), nullable=True))

    # Create index
    op.create_index('idx_phash', 'images', ['phash'])

    # Populate existing records (data migration)
    connection = op.get_bind()
    connection.execute("""
        UPDATE images SET phash = calculate_phash(file_path)
        WHERE phash IS NULL AND processed = TRUE
    """)

def downgrade():
    op.drop_index('idx_phash')
    op.drop_column('images', 'phash')
```

## Performance Projections

### Storage Estimates (1 year)
- **Messages**: 365k records × 200 bytes ≈ 75MB
- **Images**: 50k records × 300 bytes ≈ 15MB
- **OCR Text**: 50k records × 1KB ≈ 50MB
- **Brand Hits**: 5k records × 150 bytes ≈ 1MB
- **Indexes**: ~40MB
- **Total**: ~180MB (manageable for PostgreSQL)

### Query Performance Targets
```sql
-- Message ingestion (<100ms)
INSERT INTO telegram_messages (...) VALUES (...);

-- Find unprocessed images (<1s, expect ~10-50 records)
SELECT * FROM images WHERE processed = FALSE ORDER BY timestamp LIMIT 50;

-- Brand hit analytics (<2s, expect ~100-1000 records)
SELECT brand_name, COUNT(*) FROM brand_hits
WHERE timestamp > NOW() - INTERVAL '30 days' GROUP BY brand_name;

-- Recent alerts (<1s, expect ~10-20 records)
SELECT * FROM brand_hits
WHERE alert_sent = FALSE ORDER BY timestamp DESC LIMIT 20;
```

## Consequences

### Positive
- **Fast Writes**: Simple schema supports high-frequency inserts
- **Efficient Queries**: Strategic indexes optimize common operations
- **Data Integrity**: Constraints prevent corruption and duplicates
- **Scalable**: Clear partitioning and archiving path
- **Debuggable**: Normalized structure aids troubleshooting

### Negative
- **Join Complexity**: Some queries require multiple table joins
- **Index Maintenance**: Multiple indexes require monitoring and upkeep
- **Storage Growth**: Media files and text will accumulate over time
- **Migration Risk**: Schema changes require careful planning

### Monitoring Requirements
- **Disk Usage**: Alert when database >80% full
- **Query Performance**: Monitor slow query log (>1 second)
- **Index Bloat**: Track index size vs table size ratios
- **Constraint Violations**: Alert on duplicate key errors