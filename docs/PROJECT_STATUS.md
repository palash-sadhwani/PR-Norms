# Project status

Last verified: 2026-10-25 on branch `fix/e2e-verification` (base `ad4d048`).
Only items that were actually executed are marked verified. The presentation
walkthrough is in `docs/DEMO_GUIDE.md`.

## Verified working

| Area | How it was verified |
|---|---|
| Migrations (SQLite, Postgres 16) | `alembic upgrade head` on fresh SQLite files, a local Postgres 16, and the `postgres:16` image |
| Real GitHub import | `POST /public/import` of `palash-sadhwani/PR-Norms` through the app: 1 PR, 14 files, 6 commits stored; re-import is idempotent by GitHub ID |
| Import errors | Inaccessible repository → readable 403; malformed identifier → 400 |
| Mining → approve → analyze → feedback → evaluation | `scripts/verify_workflow.py --demo-fallback` through the HTTP routes, with real HDBSCAN and the labelled `demo-hashing` embedder |
| Demo scenarios through mining and analysis | Scratch dry run of the 18 scripted demo comments through `mine_repository` and `analyze_pull_request` (fallback embedder): 6 norms with 3 comments each; the open shipping PR gets findings for the hardcoded-credential, missing-test and query-in-loop norms |
| Demo repository code | `demo/build_demo_repo.py git` builds every branch; each corrected branch passes its own `pytest` |
| Celery task bodies on Postgres | Three consecutive tasks in one process (`tests/test_tasks.py`) |
| Missing models or worker | `/mine` and `/analyze` return 503 with a readable message; `/sync` returns 503 when Redis is down |
| Backend Docker image | Built with `INSTALL_PHASE2=false`; ran migrations, served `/health`, seeded data on `postgres:16`, passed `pytest` |
| Frontend | `tsc -b`, `npm run build`; Playwright against the running backend showed the fallback banner, stats and the no-review-comments empty state with no console errors |

## Not verified

- **Real Sentence Transformers model:** this environment's network policy blocks
  `huggingface.co` and `download.pytorch.org`. Mining and analysis quality with
  the real model are unknown, including whether real-model relevance reaches 0.55.
- **Import of PR-Norms-Demo:** the repository does not exist yet. The GitHub App
  in this environment may not create repositories (403), so it has to be
  created by its owner (see `docs/DEMO_GUIDE.md`).
- **Public repositories other than session-attached ones:** this environment's
  proxy only allows GitHub API calls for attached repositories.
- **GraphQL review threads:** GraphQL is blocked here; resolved/outdated thread
  state is covered only by mocked tests.
- **Full `docker compose up`:** the Phase 2 image, worker, Redis and frontend
  container were not run together.

## Changes in this branch

1. Anonymous `GitHubClient("")` no longer inherits `GITHUB_TOKEN`; public syncs
   and imports pass the optional server token explicitly; GraphQL runs only with
   a token.
2. Celery tasks dispose the async engine before their event loop closes (fixes a
   reproduced asyncpg "attached to a different loop" failure).
3. `MiningResult.clustered` excludes HDBSCAN noise.
4. An embedding model that fails to load returns 503 instead of 500.
5. `pythonpath = ["."]` for pytest, so tests run inside the Docker image.
6. Embedder factory with an opt-in, labelled fallback
   (`PHASE2_EMBEDDING_BACKEND=demo-hashing`), stored under its own model name;
   mining now stores vectors under the embedder's model name, fixing the
   mining/analysis model-name mismatch. `GET /embedding/config` exposes it.
7. Mining uses line-level review comments only; conversation (issue) comments are
   stored, counted and shown separately (`/pull-requests/{id}/issue-comments`).
8. Small datasets (< 50 comments) use HDBSCAN `min_samples=1` when unset.
9. Analysis compares 3-line windows of added code with the lines each supporting
   review comment was attached to, and points findings at the matching lines;
   findings return the norm statement and evidence (PR number, file, line, link).
10. Public import can include open PRs; access errors map to a readable 403.
11. Comment preprocessing drops trailing signature footers (raw text kept).
12. UI: fallback banner, open-PR import, issue-comment and feedback counts, empty
    state for repositories without review comments, evidence with GitHub links.
13. `demo/build_demo_repo.py` and `docs/DEMO_GUIDE.md`.

## Tests (latest run)

- Backend `pytest`: **41 passed, 1 skipped** (the Postgres-only Celery test; it
  passed earlier against Postgres 16).
- `ruff check` (backend and `demo/`): clean. Frontend `tsc -b` and
  `npm run build`: clean.
- `verify_workflow.py --demo-fallback`: completes. A real-model run stops at
  `/mine` with 503 because the model download is blocked.

## Known bugs and limitations

- The violation signal is heuristic (key norm terms absent from added lines), so
  nearly every relevant norm is marked suspected. Findings are provisional.
- With the fallback embedder, the demo shipping PR's logging and validation
  issues score below the 0.55 relevance threshold and are missed.
- Norm confidence is cluster cohesion plus share of comments, not a probability.
- Formal review summaries are stored and shown but not mined.
- Issue comments and review comments share a unique `github_id` column; GitHub
  allocates those IDs from different sequences, so a collision is possible but
  has not been observed.
- The UI's mine and analyze buttons run inline rather than through the worker.

## Reproduce

```bash
cd backend
pip install -e ".[dev]"            # add ".[phase2]" for mining and analysis
pytest -q                          # PRNORMS_TEST_POSTGRES_URL=postgresql+asyncpg://... enables the Postgres test

# End-to-end workflow on a new SQLite file (never a real database):
python -m scripts.verify_workflow --database /tmp/verify.db                  # real model
python -m scripts.verify_workflow --database /tmp/verify2.db --demo-fallback # labelled fallback

# Backend image without Phase 2:
docker build --build-arg INSTALL_PHASE2=false -t prnorms-backend backend
docker run --rm -v "$PWD/backend:/app" prnorms-backend pytest -q
```

## Remaining priorities

1. Create PR-Norms-Demo, populate it, and run the full demo through the UI.
2. Run the demo once with the real model on a machine that can reach Hugging Face.
3. Run `docker compose up --build` end to end, including a queued mine and analyze.
