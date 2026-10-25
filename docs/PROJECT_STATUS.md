# Project status

Last verified: 2026-10-25 on branch `fix/e2e-verification` (base `ad4d048`).
Only items that were actually executed are marked verified.

## Verified working

| Area | How it was verified |
|---|---|
| Migrations (SQLite, Postgres 16) | `alembic upgrade head` on a fresh SQLite file, a local Postgres 16, and the `postgres:16` image |
| Persistence and read APIs | Demo seed, then `/repositories`, `/stats`, `/pull-requests`, `/evaluation` |
| Mining → approve → analyze → feedback → evaluation | `scripts/verify_workflow.py --stand-in-embedder` through the HTTP routes, with real HDBSCAN (see limits below) |
| Celery task bodies on Postgres | Three consecutive tasks in one process (`tests/test_tasks.py`) |
| Missing models or worker | `/mine` and `/analyze` return 503 with a readable message; `/sync` returns 503 when Redis is down |
| Backend Docker image | Built with `INSTALL_PHASE2=false`; container ran migrations, served `/health`, seeded data on `postgres:16`, passed `pytest` |
| Frontend | `npm ci`, `tsc -b`, `npm run build` on the host |

## Not verified

- **Real Sentence Transformers model:** this environment's network policy blocks
  `huggingface.co` and `download.pytorch.org`, so the default model never loaded.
  Mining and analysis quality with the real model are unknown.
- **GitHub API:** `api.github.com` is blocked; GitHub App install, ingestion, and
  webhooks are covered only by mocked tests.
- **Full `docker compose up`:** the backend image was smoke-tested alone with a
  build-only CA tweak for this sandbox's TLS proxy. The Phase 2 image, worker,
  Redis and the frontend container were not run.

## Changes in this branch

1. Anonymous `GitHubClient("")` no longer inherits `GITHUB_TOKEN`. Public
   repository syncs and public imports pass the optional server token
   explicitly, and GraphQL runs only when the client has a token.
2. Celery tasks dispose the async engine before their event loop closes. Before
   this, the second task in a worker process failed on Postgres with an
   asyncpg "attached to a different loop" error (reproduced).
3. `MiningResult.clustered` counted every comment, including HDBSCAN noise.
4. An embedding model that fails to load raises `RuntimeError`, so routes return
   503 instead of an unhandled 500.
5. `pythonpath = ["."]` for pytest, so `pytest` runs inside the Docker image
   (the image installs dependencies but not the `app` package).
6. Added `scripts/verify_workflow.py` and tests for each fix.

## Tests (latest run)

- Local, Python 3.13, with `[dev,phase2]` and Postgres: **37 passed**, both with
  and without `GITHUB_TOKEN` set. After the review fix: **38 passed, 1 skipped**
  (Postgres test; Postgres was not running for that final run).
- Backend image (no Phase 2), `./backend` mounted as in compose: **35 passed,
  2 skipped** (the Postgres-only and Phase 2-only tests).
- `ruff check .`: clean. `tsc -b` and `npm run build`: clean.
- Verification script: stand-in run completes. A real-model run stops at
  `/mine` with 503 because the model download is blocked.

## Known bugs and limitations

- HDBSCAN defaults (`min_samples` = `min_cluster_size` = 3) mark every comment as
  noise in very small repositories. The verification fixture sets
  `PHASE2_MIN_SAMPLES=1`.
- Analysis relevance compares embeddings of raw diffs with embeddings of English
  review comments (threshold 0.55). It is unverified whether the real model
  reaches that threshold for typical code-only diffs.
- The violation signal is lexical: a norm counts as followed when its key words
  appear in the added lines, so a line like `# use print instead of the logger`
  reads as compliant.
- Mining stores embeddings under the configured model name, while analysis looks
  them up by `embedder.model_name`. They match in normal use but would diverge
  if the two names differed.
- The UI has no **Connect GitHub** flow, and its mine and analyze buttons always
  run inline rather than through the worker.

## Reproduce

```bash
cd backend
pip install -e ".[dev]"            # add ".[phase2]" for mining and analysis
pytest -q                          # PRNORMS_TEST_POSTGRES_URL=postgresql+asyncpg://... enables the Postgres test

# End-to-end workflow on a new SQLite file (never a real database):
python -m scripts.verify_workflow --database /tmp/verify.db                       # real model
python -m scripts.verify_workflow --database /tmp/verify2.db --stand-in-embedder  # offline pipeline check

# Backend image without Phase 2:
docker build --build-arg INSTALL_PHASE2=false -t prnorms-backend backend
docker run --rm -v "$PWD/backend:/app" prnorms-backend pytest -q
```

## Remaining priorities

1. Run `verify_workflow.py` with the real model on a host that can reach
   Hugging Face, and check the analysis relevance threshold on real diffs.
2. Run `docker compose up --build` end to end, including the worker and a queued
   (non-inline) mine and analyze.
3. Import a real public repository and confirm ingestion and mining on real
   review data.
