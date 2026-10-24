# Architecture

The FastAPI service owns persistence and exposes `/api/v1`. Async SQLAlchemy models retain raw GitHub payloads where useful. Celery workers are deliberately thin orchestration boundaries around the services; `LOCAL_DEMO_MODE` or `?inline=true` runs supported operations synchronously instead. Authentication secrets are environment-only; no tokens are checked in, and the only sample data is the explicitly marked synthetic demo seed.

The pipeline has five stages, each in its own service module under `backend/app/services`:

1. **Ingestion** (`ingestion.py`, `github.py`, `auth.py`, `public_import.py`, `webhooks.py`): GitHub App or public REST import of pull requests, reviews, comments, files, commits, and review threads. Writes are idempotent upserts keyed on stable GitHub IDs, retries resume from a persisted checkpoint, and webhook deliveries are verified and deduplicated. `preprocess.py` marks bot, empty, and low-information comments as filtered.
2. **Semantic mining** (`phase2.py`): Sentence Transformers embeddings (cached by model and text hash) clustered with HDBSCAN. Each cluster becomes a candidate norm with evidence links to its comments and pull requests.
3. **Norm knowledge base** (`norms.py`): classification, lifecycle status, approval, rejection, and conflicts. Every mutation writes an immutable norm version.
4. **PR analysis** (`analysis.py`): changed-file context is compared with approved, active norms. Findings carry an explainable risk score and are always provisional; nothing is written to GitHub.
5. **Evaluation** (`evaluation.py`): precision, recall, F1, and false-positive rate from human feedback labels.

The React frontend (`frontend/src/main.tsx`) provides Dashboard, Norm Explorer, PR Analyzer, and Evaluation views over the same API.
