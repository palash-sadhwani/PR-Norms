# PR-Norms

PR-Norms collects GitHub pull-request review data and mines reproducible Phase 2
candidate engineering norms from semantically similar comments. Phase 3 violation
detection is not included.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Backend API: `http://localhost:8000/docs`, frontend: `http://localhost:5173`.
Run backend checks with `cd backend && pip install -e ".[dev]" && pytest`.
Install local Phase 2 models with `cd backend && pip install -e ".[phase2]"`.

For a Redis-free localhost demonstration, set `LOCAL_DEMO_MODE=true`. This
explicit local-only switch runs supported sync, mining, and analysis operations
synchronously. Celery remains the default when the switch is unset or false.
An optional `GITHUB_TOKEN` may be set in the backend terminal only; it is never
sent to the browser.

See `docs/architecture.md` for the boundaries and ingestion guarantees.

Configure a GitHub App with pull-request read access, contents read access,
metadata read access, and repository webhook delivery. Set the App slug,
private key, webhook secret, callback URL, and a random `GITHUB_STATE_SECRET`
in `.env`. The browser only receives the GitHub installation redirect and
repository data; installation tokens remain server-side.

The import flow is:

1. Open **Connect GitHub**.
2. Complete the GitHub App installation.
3. Select a discovered repository.
4. Start historical import and monitor `/api/v1/ingestion/{job_id}`.

The worker resumes from the persisted pull-request checkpoint after a retry.
Webhook deliveries are verified, deduplicated, and enqueue incremental syncs.

## Public repository demonstration

For a localhost-only demonstration without GitHub App credentials, import a
bounded set of recent closed pull requests from the public GitHub REST API:

```powershell
cd backend
$env:DATABASE_URL="sqlite+aiosqlite:///./prnorms.db"
..\.venv\Scripts\python.exe -m alembic upgrade head
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal, import up to ten recent closed PRs:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/public/import `
  -ContentType "application/json" `
  -Body '{"repository":"fastapi/fastapi","max_pull_requests":10}'
```

The response includes `repository_id`, `job_id`, completion counts, and
errors. Public imports are marked `github_public_api` and are never mixed with
the explicitly marked synthetic demo repository. Repeating the import is
idempotent for PRs, comments, files, and commits; each explicit import request
gets its own ingestion job while reusing those records.

Run Phase 2 mining synchronously for a local demonstration (Redis/Celery is
not required):

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/repositories/{repository_id}/mine?inline=true
```

This requires the optional Sentence Transformers and HDBSCAN dependencies and
access to the configured model. If they are unavailable, the API returns an
explicit `503`; no candidate norms are fabricated. Inspect genuine evidence
with:

```text
GET /api/v1/repositories/{repository_id}/candidate-norms
GET /api/v1/repositories/{repository_id}/clusters
```

After an import, start semantic mining with:

```text
POST /api/v1/repositories/{repository_id}/mine
GET  /api/v1/repositories/{repository_id}/clusters
GET  /api/v1/repositories/{repository_id}/candidate-norms
```

Phase 2 uses the configured Sentence Transformers model and HDBSCAN. The
default model is `sentence-transformers/all-MiniLM-L6-v2`; set
`PHASE2_EMBEDDING_MODEL`, `PHASE2_MIN_CLUSTER_SIZE`, and
`PHASE2_MIN_SAMPLES` to tune local runs. Original comment text remains in the
Phase 1 records, while candidate norms retain comment and pull-request evidence.

Phase 4 analyzes an imported pull request without writing to GitHub:

```text
POST /api/v1/pull-requests/{pull_request_id}/analyze
GET  /api/v1/pull-requests/{pull_request_id}/analysis
POST /api/v1/norm-findings/{finding_id}/feedback
```

Only approved, active norms are considered. Findings are provisional. The
current explainable score is:

```text
risk_score = 0.45 * relevance_score
           + 0.35 * evidence_strength
           + 0.20 * norm_confidence
```

Risk levels are `high` at 0.80 or above, `medium` at 0.60 or above, and
`low` below 0.60 when a deterministic contradiction signal is present.
Relevant norms with no contradiction are returned with risk level `none`.
Missing diff context produces no findings.

## Phase 5 web platform and evaluation

The React application provides repository-scoped Dashboard, Norm Explorer,
PR Analyzer, and Evaluation views. Norms can be searched by statement,
classification, lifecycle status, and approval state. Detail views show
supporting review-comment/historical-PR links and immutable norm versions.
PR findings remain provisional; human `confirmed`, `accepted`, `dismissed`,
and `needs_review` feedback is recorded through the existing feedback API.

Evaluation uses the latest feedback label for each finding in the selected
repository. `confirmed` and `accepted` are positive labels; `dismissed` and
`needs_review` are negative labels. Precision, recall, F1, and false-positive
rate are returned with sample counts. Metrics whose denominator is zero are
returned as `null` and rendered as `Undefined`.

The web platform does not fabricate repositories, progress, findings, or
metrics. GitHub, Redis/Celery, and Sentence Transformer availability is
reported as an unavailable/error state when those services are not running.

## Local demonstration dataset

When GitHub credentials or Docker services are unavailable, create the
explicitly synthetic demonstration repository in a separate `manual-demo.db`:

```powershell
cd backend
$env:DATABASE_URL="sqlite+aiosqlite:///./manual-demo.db"
$env:LOCAL_DEMO_MODE="true"
.\.venv312\Scripts\python.exe -m alembic upgrade head
.\.venv312\Scripts\python.exe -m scripts.seed_demo
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

The Vite proxy targets `http://127.0.0.1:8000`. The frontend labels the seeded
repository as `Synthetic demo data`.

The dataset is labeled `demo/pr-norms-showcase` and its repository metadata
contains `pr-norms-synthetic-demo-v1`. The seed is repeatable: rerunning it
reports that the demo already exists, and it refuses to use that name if an
unmarked repository already exists. It never modifies real repositories or
sends GitHub requests. Use a new database filename for a fresh disposable demo;
never reset the real imported databases.

Phase 3 keeps candidate norms provisional until an explicit decision:

```text
GET   /api/v1/candidate-norms/{norm_id}
PATCH /api/v1/candidate-norms/{norm_id}
POST  /api/v1/candidate-norms/{norm_id}/classify
POST  /api/v1/candidate-norms/{norm_id}/approve
POST  /api/v1/candidate-norms/{norm_id}/reject
POST  /api/v1/candidate-norms/{norm_id}/conflicts
```

Classification values are `hard_rule`, `team_convention`,
`reviewer_preference`, and `one_off`. Lifecycle statuses are `emerging`,
`active`, `weakening`, `deprecated`, and `conflicting`. Every mutation adds an
immutable norm version with its reason, actor, confidence, and evidence count.
