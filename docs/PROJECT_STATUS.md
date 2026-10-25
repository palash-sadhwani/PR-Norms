# Project status

Last verified: 2026-10-25 on branch `fix/e2e-verification` (base `ad4d048`).
Only items that were actually executed are marked verified. The presentation
walkthrough is in `docs/DEMO_GUIDE.md`.

## Verified working

| Area | How it was verified |
|---|---|
| Migrations (SQLite, Postgres 16) | `alembic upgrade head` on fresh SQLite files, a local Postgres 16, and the `postgres:16` image |
| Real GitHub import | `POST /public/import` of `palash-sadhwani/PR-Norms` (1 PR, 14 files, 6 commits) and of `palash-sadhwani/PR-Norms-Demo` (7 PRs, 6 reviews, 18 line-level review comments, 6 conversation comments, 10 files, 13 commits) |
| Demo repository published | `demo/build_demo_repo.py populate`: PRs #1–#6 each have one COMMENT review with 3 inline comments, a correction commit, a reply, and are merged; #7 is open and unreviewed (checked through the GitHub API) |
| Demo workflow, **fallback embedder only** | Through the app on a fresh SQLite file: mining gave 6 norms (3 comments from 3 different PRs each, no noise); all 6 approved; analysis of #7 gave 3 findings (hardcoded credential at `shipping.py:5-7`, query in a loop at `12-14`, missing test at `12-14`); all 9 evidence links match GitHub's PR number, file, original line and text; feedback and evaluation submitted in the browser (3 labels: precision 0.67, recall 1.0, F1 0.80) |
| Import errors | Inaccessible repository → readable 403; malformed identifier → 400 |
| Mining → approve → analyze → feedback → evaluation | `scripts/verify_workflow.py --demo-fallback` through the HTTP routes, with real HDBSCAN and the labelled `demo-hashing` embedder |
| Demo repository code | `demo/build_demo_repo.py git` builds every branch; each corrected branch passes its own `pytest` |
| Celery task bodies on Postgres | Three consecutive tasks in one process (`tests/test_tasks.py`) |
| Missing models or worker | `/mine` and `/analyze` return 503 with a readable message; `/sync` returns 503 when Redis is down |
| Backend Docker image | Built with `INSTALL_PHASE2=false`; ran migrations, served `/health`, seeded data on `postgres:16`, passed `pytest` |
| Frontend | `tsc -b`, `npm run build`; Playwright against the running backend showed the fallback banner, stats and the no-review-comments empty state with no console errors |

## Not verified

- **Real Sentence Transformers model:** this environment's network policy blocks
  `huggingface.co` and `download.pytorch.org`. Mining and analysis quality with
  the real model are unknown, including whether real-model relevance reaches 0.55.
- **Real-model run on the demo repository:** importing worked, but `/mine` with the
  default backend returned 503 (`403 Forbidden` from Hugging Face).
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
14. Evidence reports `original_line` for review comments GitHub marks outdated
    (its `line` is null after later commits), flagged `outdated`.
15. Submitting feedback refreshes the dashboard counts.

## Tests (latest run)

- Backend `pytest`: **42 passed, 1 skipped** (the Postgres-only Celery test; it
  passed earlier against Postgres 16).
- `ruff check` (backend and `demo/`): clean. Frontend `tsc -b` and
  `npm run build`: clean.
- `verify_workflow.py --demo-fallback`: completes. A real-model run stops at
  `/mine` with 503 because the model download is blocked.

## Known bugs and limitations

- The violation signal is heuristic (key norm terms absent from added lines), so
  nearly every relevant norm is marked suspected. Findings are provisional.
- With the fallback embedder, the demo shipping PR's logging and validation
  issues score below the 0.55 relevance threshold and are missed, and the
  missing-test finding is anchored to the loop lines by word overlap.
- Feedback is an append-only history; evaluation uses each finding's latest label.
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

1. Run the demo once with the real model on a machine that can reach Hugging Face.
2. Decide whether formal review summaries should also be mined.
3. Run `docker compose up --build` end to end, including a queued mine and analyze.
