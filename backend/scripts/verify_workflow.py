"""End-to-end check of mining and analysis through the HTTP API.

Runs against a fresh SQLite database (never a real one):

    python -m scripts.verify_workflow --database ./verify.db

Steps: synthetic fixture -> mine -> approve -> analyze -> feedback -> evaluation.
The default uses the configured Sentence Transformers model. When the model
cannot be downloaded, `--demo-fallback` sets PHASE2_EMBEDDING_BACKEND=demo-hashing,
the app's labelled lexical fallback, so the pipeline (HDBSCAN, persistence,
routes) can still be exercised; that run does not verify embedding quality.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

FIXTURE_REPOSITORY = "verify/pr-norms-e2e"
FIXTURE_MARKER = "pr-norms-verification-fixture"
HISTORICAL_COMMENTS = [
    ("app/services/accounts.py", "Please move database access into the repository layer."),
    ("app/services/orders.py", "Move this database access into the repository layer please."),
    ("app/services/billing.py", "Database access belongs in the repository layer, move it."),
    ("app/services/users.py", "Please keep database access in the repository layer."),
    ("app/api/accounts.py", "Use the logger instead of print statements here."),
    ("app/api/orders.py", "Please use the logger instead of print statements."),
    ("app/api/billing.py", "Replace print statements with the logger please."),
    ("app/api/users.py", "Use the logger, not print statements, in this module."),
]
TARGET_PATCH = (
    "@@ -1,2 +1,4 @@\n import os\n"
    "+# Use print statements instead of the logger here for now.\n"
    '+print("loaded accounts")\n'
)


async def create_fixture() -> tuple[int, int]:
    from app.db.session import SessionLocal
    from app.models import (
        Installation,
        PullRequest,
        PullRequestFile,
        Repository,
        Review,
        ReviewComment,
        User,
    )
    from app.services.preprocess import preprocess_comment

    marker = {"fixture": FIXTURE_MARKER}
    async with SessionLocal() as db:
        installation = Installation(
            github_id=8_000_000, account_login="verify", account_type="User", raw=marker
        )
        db.add(installation)
        await db.flush()
        repository = Repository(
            installation_id=installation.id,
            github_id=8_000_001,
            full_name=FIXTURE_REPOSITORY,
            default_branch="main",
            raw=marker,
        )
        reviewer = User(github_id=8_000_002, login="fixture-reviewer")
        db.add_all([repository, reviewer])
        await db.flush()
        for index, (path, body) in enumerate(HISTORICAL_COMMENTS):
            pull_request = PullRequest(
                repository_id=repository.id,
                github_id=8_000_100 + index,
                number=100 + index,
                title=f"Historical change {index}",
                state="closed",
                merged=True,
                base_branch="main",
                head_branch=f"change-{index}",
                raw=marker,
            )
            db.add(pull_request)
            await db.flush()
            review = Review(
                pull_request_id=pull_request.id,
                github_id=8_000_200 + index,
                author_id=reviewer.id,
                state="commented",
                raw=marker,
            )
            db.add(review)
            await db.flush()
            db.add(
                ReviewComment(
                    pull_request_id=pull_request.id,
                    review_id=review.id,
                    github_id=8_000_300 + index,
                    author_id=reviewer.id,
                    body=body,
                    normalized_body=preprocess_comment(body).normalized_text,
                    path=path,
                    line=2,
                    diff_hunk='@@ -1,1 +1,2 @@\n+print("x")',
                    raw=marker,
                )
            )
        target = PullRequest(
            repository_id=repository.id,
            github_id=8_000_999,
            number=999,
            title="Add account loading",
            state="open",
            merged=False,
            base_branch="main",
            head_branch="feature/accounts",
            raw=marker,
        )
        db.add(target)
        await db.flush()
        db.add(
            PullRequestFile(
                pull_request_id=target.id,
                filename="app/api/accounts.py",
                status="modified",
                additions=2,
                deletions=0,
                patch=TARGET_PATCH,
                raw=marker,
            )
        )
        await db.commit()
        return repository.id, target.id


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


async def run() -> None:
    import httpx

    from app.main import app
    repository_id, target_id = await create_fixture()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as api:
        config = (await api.get("/embedding/config")).json()
        print(f"embedder: {config['model_name']} (fallback={config['fallback']})")
        mined = await api.post(f"/repositories/{repository_id}/mine?inline=true")
        check(mined.status_code == 202, f"mine -> {mined.status_code} {mined.text[:300]}")
        check(
            mined.json()["embedding_fallback"] == config["fallback"],
            "mining reports the embedder it used",
        )
        print("  mining:", {k: mined.json().get(k) for k in ("embedded", "clusters", "norms")})

        norms = (await api.get(f"/repositories/{repository_id}/candidate-norms")).json()
        check(len(norms) > 0, f"candidate norms mined: {len(norms)}")
        for norm in norms:
            print(f"  norm {norm['id']}: {norm['statement']!r} support={norm['support_count']}")
            approved = await api.post(
                f"/candidate-norms/{norm['id']}/approve",
                json={"changed_by": "verify-script", "reason": "verification run"},
            )
            check(approved.status_code == 200, f"approve norm {norm['id']}")

        analyzed = await api.post(f"/pull-requests/{target_id}/analyze?inline=true")
        check(analyzed.status_code == 202, f"analyze -> {analyzed.status_code} {analyzed.text[:300]}")
        analysis = (await api.get(f"/pull-requests/{target_id}/analysis")).json()
        findings = analysis.get("findings", [])
        check(len(findings) > 0, f"findings returned: {len(findings)}")
        for finding in findings:
            print(
                f"  finding {finding['id']}: norm={finding['candidate_norm_id']} "
                f"risk={finding['risk_level']} score={finding['risk_score']}"
            )

        feedback = await api.post(
            f"/norm-findings/{findings[0]['id']}/feedback",
            json={"decision": "confirmed", "changed_by": "verify-script"},
        )
        check(feedback.status_code == 200, f"feedback -> {feedback.status_code}")
        evaluation = (await api.get(f"/repositories/{repository_id}/evaluation")).json()
        check(evaluation["labeled_count"] >= 1, f"evaluation labeled_count={evaluation['labeled_count']}")
        print("  evaluation:", evaluation)
    print("workflow completed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, help="path to a new SQLite file")
    parser.add_argument("--demo-fallback", action="store_true")
    args = parser.parse_args()
    database = Path(args.database).resolve()
    if database.exists():
        raise SystemExit(f"{database} exists; pass a new file so no real data is touched")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{database}"
    # HDBSCAN's default min_samples marks every point of an 8-comment fixture as noise.
    os.environ.setdefault("PHASE2_MIN_SAMPLES", "1")
    if args.demo_fallback:
        os.environ["PHASE2_EMBEDDING_BACKEND"] = "demo-hashing"
    backend = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=backend, check=True
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
