# Demo guide

A five-minute walkthrough of PR-Norms on real GitHub data. Status of each step is in
`docs/PROJECT_STATUS.md`; nothing here is claimed beyond what that file records.

## 1. Start the application

Local run without Docker (SQLite, no Redis or Celery needed):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,phase2]"                           # phase2 = Sentence Transformers + HDBSCAN
export DATABASE_URL="sqlite+aiosqlite:///./demo.db"      # PowerShell: $env:DATABASE_URL="..."
export LOCAL_DEMO_MODE=true                              # run import, mining, analysis inline
python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend && npm ci && npm run dev -- --host 127.0.0.1 --port 5173
```

Open http://127.0.0.1:5173. The first mining run downloads
`sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) from Hugging Face.

Docker alternative: `cp .env.example .env && docker compose up --build`.

### Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLite or Postgres URL. Use a new SQLite file for the demo. |
| `LOCAL_DEMO_MODE` | `true` runs sync, mining and analysis inline (no Redis/Celery). |
| `GITHUB_TOKEN` | Optional, backend only. Raises the GitHub rate limit from 60 to 5,000 requests/hour. Placeholder: `ghp_your_token_here`. Never commit it. |
| `PHASE2_EMBEDDING_BACKEND` | `sentence-transformers` (default, the real model) or `demo-hashing` (labelled lexical fallback for machines that cannot download the model). |
| `PHASE2_MIN_CLUSTER_SIZE` | Minimum comments per norm (default 3). |

When `demo-hashing` is active the UI shows a red banner, and mining and analysis
responses report `embedding_fallback: true`. Fallback vectors are stored under
their own model name and never mix with real-model results.

## 2. Connect a public repository

In **Public GitHub import**, paste `https://github.com/owner/repo` or
`owner/repo`, keep **Include open PRs** ticked, choose how many recent PRs to
fetch, and select **Import public repository**. The importer stores PR metadata,
formal reviews, line-level review comments, conversation (issue) comments,
changed files with diffs, and commits, keyed by GitHub IDs so re-imports update
rather than duplicate. Private repositories need `GITHUB_TOKEN` or the GitHub
App; access errors are shown as a readable 403, never replaced with other data.

Only line-level review comments are mined. If a repository has none, the
dashboard says so.

## 3. Prepare the demo repository

`demo/build_demo_repo.py` builds **PR-Norms-Demo**: a small Python inventory
service with six reviewed pull requests and one open, unreviewed PR. All review
discussion is scripted demonstration content posted from the repository owner's
account and labelled as such.

| PR | Scenario | Review comments on |
|---|---|---|
| 1 | Add payment gateway client | hardcoded credential, secret in logs, missing test |
| 2 | Add product search | missing input validation, query in a loop, missing test |
| 3 | Import supplier stock CSV | hardcoded credential, swallowed exception, missing validation |
| 4 | Add order summary report | query in a loop, sensitive data in logs, swallowed exception |
| 5 | Prevent stock from going negative | missing regression test, missing validation, query in a loop |
| 6 | Record login attempts | password in logs, hardcoded secret, swallowed exception |
| 7 | Add shipping quotes (open) | not reviewed: the PR to analyze |

Each of PRs 1 to 6 gets a review with three inline comments, a correction commit,
a follow-up reply, and is merged. Each norm therefore has three supporting
comments from three different PRs.

1. Create an **empty public** repository named `PR-Norms-Demo` on GitHub (no
   README, licence or .gitignore).
2. Populate it from a machine with git push access:

   ```bash
   python demo/build_demo_repo.py git ../PR-Norms-Demo
   cd ../PR-Norms-Demo
   git remote add origin https://github.com/<owner>/PR-Norms-Demo.git
   export GITHUB_TOKEN=<token with repo scope>    # read from the environment, never printed
   python ../PR-Norms/demo/build_demo_repo.py populate <owner>/PR-Norms-Demo
   ```

   The script skips PRs, reviews and merges that already exist, so a rerun is safe.

3. In PR-Norms, import `<owner>/PR-Norms-Demo` with **Include open PRs** ticked.

## 4. Five-minute sequence

1. **Problem (30 s).** Review comments hold a team's real engineering rules, but
   they are scattered across hundreds of PR threads and never written down.
2. **Import (45 s).** Import PR-Norms-Demo. Dashboard: 7 PRs, 18 line-level review
   comments, 6 conversation comments. Open PR 1 to show a real review thread.
3. **Mine (60 s).** Norm Explorer → **Mine Norms**. Each candidate norm is a
   cluster of similar reviewer comments, with links to the original comments on
   GitHub. Point out that norms are provisional until a person approves them.
4. **Approve (30 s).** Approve the hardcoded-credential norm and two or three
   others; reject one to show the decision is recorded with a version history.
5. **Analyze (75 s).** PR Analyzer → select **#7 Add shipping quotes** →
   **Analyze PR**. Open a finding: it names the norm, the matching lines in
   `inventory/shipping.py`, the relevance scores, and, under *Supporting review
   evidence*, the earlier comments (PR, file, line, GitHub link) that justify it.
6. **Feedback and evaluation (45 s).** Confirm a correct finding, dismiss a weak
   one, then open Evaluation: precision, recall and F1 come only from these stored
   labels.
7. **Close (15 s).** Every finding traces back to real review comments; humans
   approve norms and label findings, so the system is explainable and auditable.

### Presentation script

> Code review is where a team's engineering standards actually live: "don't
> hardcode credentials", "add a regression test", "don't log customer emails".
> Those rules are repeated in comment after comment but rarely written down, so
> new contributors keep tripping over them.
>
> PR-Norms imports a repository's pull requests and review comments from GitHub,
> groups similar review comments into candidate norms, and asks a person to
> approve them. Approved norms are then applied to new pull requests.
>
> Here is the demo repository. In three separate pull requests a reviewer flagged
> a hardcoded secret: a payment API key, a supplier password and a session secret.
> PR-Norms mined those three comments into one norm. When I analyze the new
> shipping PR, it flags the hardcoded `SHIPPING_API_KEY`, highlights those lines,
> and shows the three earlier review comments, each linked to GitHub, as the
> evidence. The finding is provisional: I confirm or dismiss it, and that
> feedback drives the evaluation metrics.
>
> The workflow is GitHub ingestion → norm mining → human approval → PR analysis →
> feedback and evaluation, and every step is traceable to real review evidence.

## 5. What is implemented and verified

See `docs/PROJECT_STATUS.md` for the exact runs. In short:

- GitHub REST ingestion, norm mining, approval, analysis, feedback and evaluation
  run through the real API routes.
- Verified in the development environment with the labelled `demo-hashing`
  fallback, because Hugging Face downloads were blocked there.

## 6. Limitations to state honestly

- The real Sentence Transformers model has not been run end to end in the
  development environment. Run the demo once with it before presenting. If the
  model cannot be downloaded, set `PHASE2_EMBEDDING_BACKEND=demo-hashing` and say
  that the banner means lexical matching, not semantic embeddings.
- The violation signal is heuristic: a finding is a relevant norm plus a check
  that the norm's key terms are absent from the added lines. Findings are
  provisional by design.
- With the fallback embedder, the shipping PR's logging and validation issues
  are missed (relevance below 0.55). These are genuine misses, not hidden.
- Norm confidence is cluster cohesion and share of comments, not a probability.
- GraphQL review-thread state needs a token; without one, threads are not
  marked resolved or outdated.
