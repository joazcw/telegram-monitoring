# Architecture Overview

## System Requirements

**Primary Goal**: Real-time detection of brand/keyword mentions in Telegram group messages, with focus on OCR-based image analysis and immediate alerting.

**Success Criteria**:
- Detect brand mentions within 30 seconds of message posting
- 95%+ accuracy for clear text in images (OCR confidence >70)
- Zero message loss during normal operation
- Support 10+ concurrent group monitoring
- Alert delivery within 5 seconds of detection

**Non-Requirements (v1)**:
- Real-time audio/video processing
- Machine learning model training
- Multi-tenant architecture
- Advanced analytics dashboard

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram      │    │     OCR         │    │    Brand        │
│   Groups        │    │   Engine        │    │   Matcher       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Group A   │ │    │ │ Tesseract   │ │    │ │ RapidFuzz   │ │
│ │   Group B   │◄┼────┼►│ + OpenCV    │◄┼────┼►│ Fuzzy Match │ │
│ │   Group C   │ │    │ │ Preprocess  │ │    │ │ + Normalize │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telethon      │    │   PostgreSQL    │    │   Alert Bot     │
│   Collector     │    │   Database      │    │  (Telegram)     │
│                 │    │                 │    │                 │
│ • Session mgmt  │    │ • Messages      │    │ • Format alert  │
│ • Media download│◄──►│ • Images        │◄──►│ • Send to chat  │
│ • Event routing │    │ • OCR results   │    │ • Track delivery│
└─────────────────┘    │ • Brand hits    │    └─────────────────┘
                       │ • Deduplication │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Local Disk     │
                       │   Storage       │
                       │                 │
                       │ • Media files   │
                       │ • Session data  │
                       │ • Application   │
                       │   logs          │
                       └─────────────────┘
```

## Data Flow

### 1. Message Ingestion
```
Telegram API → Telethon Client → Message Parser → Database Insert
                      ↓
                Media Download → SHA-256 Hash → Disk Storage → Image Record
```

### 2. Processing Pipeline
```
Database Poll → Unprocessed Images → OCR Engine → Text Extraction
                                          ↓
Brand Matcher ← Fuzzy Matching ← Text Normalization ← OCR Result
     ↓
Match Found? → Brand Hit Record → Alert Formatter → Telegram Bot API
```

### 3. Error Handling Flow
```
Processing Error → Log + Retry Counter → Max Retries? → Dead Letter
                                              ↓
                                         Mark Failed
```

## Core Components

### TelegramCollector
- **Purpose**: Maintain persistent connection to Telegram, collect messages
- **Key Operations**:
  - Session management and reconnection
  - Message event filtering by group
  - Media download with deduplication
  - Database persistence with idempotency

### OCREngine
- **Purpose**: Extract text from images with preprocessing
- **Key Operations**:
  - Image preprocessing (noise reduction, thresholding)
  - Tesseract text extraction with confidence scoring
  - Error handling for corrupted/unsupported formats

### BrandMatcher
- **Purpose**: Detect brand mentions using fuzzy string matching
- **Key Operations**:
  - Text normalization (remove punctuation, lowercase)
  - Exact substring matching (priority)
  - Fuzzy matching with configurable threshold
  - Deduplication of multiple matches per brand

### AlertSender
- **Purpose**: Format and deliver notifications
- **Key Operations**:
  - Alert templating with context
  - Telegram bot API interaction
  - Delivery confirmation and retry logic

## Data Boundaries

### Stored Data
- **telegram_messages**: Full message content + metadata
- **images**: File path, hash, processing status
- **ocr_text**: Extracted text + confidence scores
- **brand_hits**: Match details + alert status

### Derived Data
- OCR confidence aggregations
- Processing time metrics
- Alert delivery statistics

### External Dependencies
- Telegram API session (persistent, renewable)
- Tesseract OCR engine (local installation)
- PostgreSQL connection (with connection pooling)

## Error Handling & Recovery

### Telegram API Errors
- **Rate Limits**: Exponential backoff with max 5-minute delay
- **Session Expiry**: Automatic reauthorization prompt
- **Network Failures**: Reconnection with circuit breaker

### Processing Errors
- **OCR Failures**: Log and skip, mark image as failed
- **Database Errors**: Transaction rollback, retry up to 3 times
- **Alert Failures**: Log, mark unsent, manual intervention required

### Data Integrity
- **Message Deduplication**: Unique constraint on (chat_id, message_id)
- **Media Deduplication**: SHA-256 hash uniqueness
- **Processing Idempotency**: Boolean flags prevent reprocessing

## Scale-Up Considerations

### Current Bottlenecks
- Single-threaded OCR processing (~2-3 seconds per image)
- Sequential message processing
- Local disk storage limits

### Horizontal Scaling Path
```
Current: Monolith Process
             ↓
Phase 2: Message Queue + Worker Pool
         (Redis/RabbitMQ + multiple OCR workers)
             ↓
Phase 3: Microservices
         (Collector, OCR Service, Matcher Service, Alert Service)
             ↓
Phase 4: Container Orchestration
         (Kubernetes + auto-scaling)
```

### Storage Scaling
```
Local Disk → MinIO/S3 → CDN for media serving
PostgreSQL → Read replicas → Sharding by chat_id
```

### Performance Targets (v2+)
- Support 1000+ concurrent groups
- <10 second end-to-end processing
- 99.9% uptime
- Horizontal auto-scaling

## Security Considerations

### API Keys & Sessions
- Telegram session encrypted at rest
- Environment variable injection only
- No credentials in logs or code

### Data Access
- Database user with minimal permissions
- No external API endpoints exposed
- Local network communication only

### Privacy
- Message content stored temporarily (configurable retention)
- No user PII collection beyond Telegram IDs
- Optional data anonymization for analytics

## Monitoring & Observability

### Key Metrics
- Messages processed per minute
- OCR processing time percentiles
- Brand hit detection rate
- Alert delivery success rate
- Database query performance

### Health Checks
- Telegram connection status
- Database connectivity
- Disk space availability
- OCR engine responsiveness

### Alerting Triggers
- Message processing backlog >100
- OCR failure rate >10%
- Alert delivery failure rate >5%
- Disk space <1GB remaining