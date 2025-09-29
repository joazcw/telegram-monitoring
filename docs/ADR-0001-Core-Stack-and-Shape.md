# ADR-0001: Core Stack and Shape

**Status**: Accepted
**Date**: 2025-09-27
**Decision Makers**: Platform Engineering Team
**Stakeholders**: Security, Operations, Development

## Context

We need to implement a fraud monitoring system that ingests Telegram messages, performs OCR on images to detect brand mentions, and sends real-time alerts. The system must be:

- **Fast to deploy**: Operational within 1 week
- **Low maintenance**: Minimal infrastructure complexity
- **Reliable**: Handle message ingestion without loss
- **Scalable**: Clear upgrade path for higher volumes

Key constraints:
- Small team (2-3 developers)
- Low initial message volume (~1000 messages/day)
- Budget-conscious infrastructure
- Compliance with basic security requirements

## Decision

We will implement a **Python monolith** with the following core stack:

### Language & Runtime
- **Python 3.11**: Mature ecosystem for OCR, fuzzy matching, async I/O
- **asyncio**: Handle concurrent Telegram connections and processing

### Telegram Integration
- **Telethon**: User session-based client for group message ingestion
- **Telegram Bot API**: For alert delivery (simpler than user session for outbound)

### Database & Storage
- **PostgreSQL 15**: ACID compliance, JSON support, full-text search capabilities
- **SQLAlchemy + Alembic**: ORM with migrations for schema evolution
- **Local filesystem**: Media storage with SHA-256 deduplication

### OCR & Text Processing
- **Tesseract 5.x**: Industry standard, supports 100+ languages
- **OpenCV**: Image preprocessing for OCR accuracy
- **pytesseract**: Python wrapper with confidence scoring

### Brand Detection
- **RapidFuzz**: Fast fuzzy string matching for typo tolerance
- **Custom normalization**: Handle punctuation, spacing, case variations

### Deployment & Orchestration
- **Docker + Docker Compose**: Reproducible environments
- **Single container**: Monolith approach for simplicity
- **PostgreSQL container**: Official image with health checks

## Options Considered

### Alternative Language Choices

#### Node.js
- **Pros**: Good Telegram library ecosystem, async by default
- **Cons**: Limited OCR options, weaker ML/text processing libraries
- **Verdict**: Rejected - OCR ecosystem not mature enough

#### Go
- **Pros**: Excellent concurrency, fast deployment, single binary
- **Cons**: Limited OCR bindings, smaller fuzzy matching libraries
- **Verdict**: Rejected - Would require more custom OCR integration

#### Java/Spring Boot
- **Pros**: Mature ecosystem, good Tesseract bindings, enterprise features
- **Cons**: Higher memory usage, slower startup, over-engineered for use case
- **Verdict**: Rejected - Too heavy for our requirements

### Alternative Telegram Integrations

#### Official Bot API Only
- **Pros**: Simple, well-documented, stable
- **Cons**: Cannot read messages in groups where bot isn't admin
- **Verdict**: Rejected - Insufficient for passive monitoring

#### MTProto Direct Implementation
- **Pros**: Maximum control and performance
- **Cons**: Complex protocol, high maintenance, security risks
- **Verdict**: Rejected - Too complex for timeline

#### Pyrogram
- **Pros**: Modern Python Telegram client, active development
- **Cons**: Smaller community, fewer examples for our use case
- **Verdict**: Considered but Telethon chosen for maturity

### Alternative Database Options

#### SQLite
- **Pros**: Zero configuration, embedded, perfect for development
- **Cons**: No concurrent writes, limited full-text search, scaling concerns
- **Verdict**: Rejected - Will hit limits quickly

#### MongoDB
- **Pros**: Schema flexibility, good for unstructured message data
- **Cons**: Additional operational complexity, less mature full-text search
- **Verdict**: Rejected - RDBMS better fit for our structured queries

#### Redis + PostgreSQL
- **Pros**: Fast caching layer, pub/sub for real-time alerts
- **Cons**: Added complexity, two systems to maintain
- **Verdict**: Deferred to v2 - Keep it simple for now

### Alternative OCR Solutions

#### Cloud OCR (Google Vision, AWS Textract)
- **Pros**: Higher accuracy, handles complex layouts, no local dependencies
- **Cons**: Cost per request, latency, data privacy concerns, vendor lock-in
- **Verdict**: Rejected - Cost and privacy concerns for fraud monitoring

#### PaddleOCR
- **Pros**: Modern neural OCR, excellent accuracy, supports Asian languages
- **Cons**: Larger Docker images, GPU requirements for best performance
- **Verdict**: Planned for v2 - Tesseract sufficient for initial deployment

### Alternative Deployment Options

#### Kubernetes
- **Pros**: Auto-scaling, service discovery, rolling deployments
- **Cons**: Operational complexity, overkill for single service
- **Verdict**: Rejected - Docker Compose sufficient for v1

#### Serverless (AWS Lambda, Cloud Functions)
- **Pros**: Auto-scaling, pay-per-use, no server management
- **Cons**: Cold starts, complex state management, vendor lock-in
- **Verdict**: Rejected - Persistent connections needed for Telegram

#### VM with systemd
- **Pros**: Simple, direct control, minimal overhead
- **Cons**: Manual deployment, no container benefits, harder to reproduce
- **Verdict**: Rejected - Docker provides better reproducibility

## Consequences

### Positive
- **Fast Development**: Python ecosystem allows rapid prototyping
- **Simple Operations**: Single container deployment, familiar database
- **Cost Effective**: No cloud OCR charges, minimal infrastructure
- **Flexible**: Easy to swap components as requirements evolve
- **Debuggable**: Monolith makes troubleshooting straightforward

### Negative
- **Performance Bottlenecks**: Single-threaded OCR processing
- **Storage Limitations**: Local disk will need monitoring and rotation
- **Scaling Ceiling**: Will need architectural changes for high volume
- **Telegram Dependency**: Rate limits and API changes are risks

### Risk Mitigation
- **Performance**: Implement processing queues in v2 if needed
- **Storage**: Add S3/MinIO integration path, implement retention policies
- **Scaling**: Document clear microservices migration path
- **Telegram**: Implement robust retry logic, plan for API changes

## Future Decision Points

### When to Revisit (Triggers)
- Message volume >10,000/day (performance)
- Storage usage >100GB (storage costs)
- OCR processing backlog >5 minutes (accuracy/latency)
- Team size >5 developers (code complexity)
- Compliance requirements change (security/audit)

### Planned Evolution
- **v2**: Add Redis for queuing, PaddleOCR option, S3 storage
- **v3**: Microservices split, Kubernetes deployment, ML-based matching
- **v4**: Multi-tenant support, advanced analytics, real-time dashboard

## Implementation Notes

### Critical Path Dependencies
1. Telegram API credentials and initial session setup
2. PostgreSQL schema with proper indexing
3. Tesseract language pack installation
4. Docker image with all system dependencies

### Success Metrics
- Deployment time: <2 hours from git clone to running system
- Message processing latency: <30 seconds end-to-end
- OCR accuracy: >90% for clear text images
- System uptime: >99% during business hours