# Architecture

The FastAPI service owns persistence and exposes `/api/v1`. Async SQLAlchemy models retain raw GitHub payloads where useful. Ingestion is designed around stable GitHub IDs, unique constraints, delivery deduplication, and checkpoints so retries are safe. Celery workers are deliberately thin orchestration boundaries. The frontend is a read-only Phase 1 explorer. Authentication secrets are environment-only; no tokens or sample/fake data are checked in. Phase 2 ML and mining are not included.
