"""Create a repeatable, explicitly synthetic dataset for local demonstrations."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentCluster,
    CommentEmbedding,
    Installation,
    NormFinding,
    NormFindingFeedback,
    NormVersion,
    PullRequest,
    PullRequestAnalysis,
    PullRequestFile,
    Repository,
    Review,
    ReviewComment,
    User,
)

DEMO_REPOSITORY = "demo/pr-norms-showcase"
DEMO_MARKER = "pr-norms-synthetic-demo-v1"


async def seed() -> None:
    async with SessionLocal() as db:
        repository = await db.scalar(
            select(Repository).where(Repository.full_name == DEMO_REPOSITORY)
        )
        if repository is not None:
            if repository.raw.get("synthetic_demo") != DEMO_MARKER:
                raise RuntimeError(
                    f"{DEMO_REPOSITORY} already exists and is not a PR-Norms demo repository"
                )
            print(f"Demo repository already exists: id={repository.id}")
            return

        installation = Installation(
            github_id=9_000_001,
            account_login="pr-norms-demo",
            account_type="Organization",
            raw={"synthetic_demo": DEMO_MARKER},
        )
        db.add(installation)
        await db.flush()
        repository = Repository(
            installation_id=installation.id,
            github_id=9_000_002,
            full_name=DEMO_REPOSITORY,
            default_branch="main",
            raw={"synthetic_demo": DEMO_MARKER, "display_label": "Synthetic demo data"},
        )
        db.add(repository)
        await db.flush()

        reviewer = User(
            github_id=9_000_003,
            login="demo-reviewer",
            name="Demo Reviewer",
        )
        author = User(github_id=9_000_004, login="demo-author", name="Demo Author")
        db.add_all([reviewer, author])
        await db.flush()

        historical_pr = PullRequest(
            repository_id=repository.id,
            github_id=9_000_010,
            number=10,
            title="Move persistence behind the repository layer",
            body="Historical example used as norm evidence.",
            state="closed",
            merged=True,
            author_id=author.id,
            base_branch="main",
            head_branch="refactor/repository-layer",
            raw={"synthetic_demo": DEMO_MARKER},
        )
        target_pr = PullRequest(
            repository_id=repository.id,
            github_id=9_000_011,
            number=11,
            title="Add account lookup",
            body="Demonstration PR with a provisional finding.",
            state="open",
            merged=False,
            author_id=author.id,
            base_branch="main",
            head_branch="feature/account-lookup",
            raw={"synthetic_demo": DEMO_MARKER},
        )
        db.add_all([historical_pr, target_pr])
        await db.flush()

        review = Review(
            pull_request_id=historical_pr.id,
            github_id=9_000_020,
            author_id=reviewer.id,
            state="changes_requested",
            body="Please keep database access behind the repository layer.",
            raw={"synthetic_demo": DEMO_MARKER},
        )
        db.add(review)
        await db.flush()
        comment = ReviewComment(
            pull_request_id=historical_pr.id,
            review_id=review.id,
            github_id=9_000_030,
            author_id=reviewer.id,
            body="Move DB access to the repository layer.",
            normalized_body="move db access repository layer",
            path="app/services/accounts.py",
            line=24,
            diff_hunk="@@ -20,4 +20,5 @@\n+db.session.execute(query)",
            raw={"synthetic_demo": DEMO_MARKER},
        )
        db.add(comment)
        await db.flush()

        cluster = CommentCluster(
            repository_id=repository.id,
            model_name="synthetic-demo",
            algorithm="synthetic",
            algorithm_config={"synthetic_demo": DEMO_MARKER},
            label=1,
            size=1,
            confidence=0.95,
        )
        db.add(cluster)
        await db.flush()
        norm = CandidateNorm(
            repository_id=repository.id,
            cluster_id=cluster.id,
            statement="Keep database access behind the repository layer.",
            confidence=0.92,
            classification="team_convention",
            status="active",
            approved=True,
            decision="approved",
            reviewed_by="demo-reviewer",
            reviewed_at=datetime.now(UTC).replace(tzinfo=None),
            support_count=1,
        )
        db.add(norm)
        await db.flush()
        db.add(
            CandidateNormEvidence(
                candidate_norm_id=norm.id,
                review_comment_id=comment.id,
                pull_request_id=historical_pr.id,
            )
        )
        db.add(
            NormVersion(
                candidate_norm_id=norm.id,
                version=1,
                statement=norm.statement,
                classification=norm.classification,
                confidence=norm.confidence,
                status=norm.status,
                approved=norm.approved,
                decision=norm.decision,
                reason="Synthetic demonstration approval",
                changed_by="demo-reviewer",
                evidence_count=1,
            )
        )
        db.add(
            CommentEmbedding(
                review_comment_id=comment.id,
                model_name="synthetic-demo",
                source_text_hash="synthetic-demo-comment",
                dimensions=2,
                vector=[1.0, 0.0],
            )
        )
        target_file = PullRequestFile(
            pull_request_id=target_pr.id,
            filename="app/services/accounts.py",
            status="modified",
            additions=1,
            deletions=0,
            patch="@@ -20,4 +20,5 @@\n+db.session.execute(query)",
            raw={"synthetic_demo": DEMO_MARKER},
        )
        db.add(target_file)
        await db.flush()
        analysis = PullRequestAnalysis(
            pull_request_id=target_pr.id,
            status="completed",
            formula_version="v1",
            findings_count=1,
            highest_risk="medium",
            context_available=True,
        )
        db.add(analysis)
        await db.flush()
        finding = NormFinding(
            analysis_id=analysis.id,
            candidate_norm_id=norm.id,
            pull_request_file_id=target_file.id,
            file_path=target_file.filename,
            line_start=21,
            line_end=21,
            relevance_score=0.88,
            evidence_strength=0.20,
            risk_score=0.75,
            risk_level="medium",
            provisional=True,
            violation_suspected=True,
            explanation=(
                "Synthetic demonstration finding: database access appears outside "
                "the repository layer."
            ),
            evidence_comment_ids=[comment.id],
            evidence_pull_request_ids=[historical_pr.id],
            status="confirmed",
        )
        db.add(finding)
        await db.flush()
        db.add(
            NormFindingFeedback(
                finding_id=finding.id,
                decision="confirmed",
                note="Synthetic demonstration feedback.",
                changed_by="demo-reviewer",
            )
        )
        await db.commit()
        print(
            f"Created synthetic demo repository id={repository.id}, "
            f"historical_pr_id={historical_pr.id}, target_pr_id={target_pr.id}, "
            f"norm_id={norm.id}, finding_id={finding.id}"
        )


if __name__ == "__main__":
    asyncio.run(seed())
