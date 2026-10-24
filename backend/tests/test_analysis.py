import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentCluster,
    CommentEmbedding,
    Installation,
    NormConflict,
    NormFinding,
    NormFindingFeedback,
    PullRequest,
    PullRequestFile,
    Repository,
    ReviewComment,
)
from app.services.analysis import analyze_pull_request, record_feedback


class FakeEmbedder:
    model_name = "test"

    def encode(self, texts, batch_size):
        return [
            [1.0, 0.0] if "repository" in text or "db.session" in text else [0.0, 1.0]
            for text in texts
        ]


async def database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, factory


async def fixture(
    db, *, status="active", approved=True, patch="@@ -1 +1 @@\n+repository layer"
):
    installation = Installation(github_id=1, account_login="acme")
    db.add(installation)
    await db.flush()
    repository = Repository(
        installation_id=installation.id, github_id=2, full_name="acme/demo"
    )
    db.add(repository)
    await db.flush()
    pr = PullRequest(
        repository_id=repository.id, github_id=3, number=1, title="Change", state="open"
    )
    db.add(pr)
    await db.flush()
    file = PullRequestFile(pull_request_id=pr.id, filename="app/db.py", patch=patch)
    db.add(file)
    cluster = CommentCluster(
        repository_id=repository.id,
        model_name="test",
        algorithm="hdbscan",
        algorithm_config={},
        label=1,
        size=1,
        confidence=0.9,
    )
    db.add(cluster)
    await db.flush()
    norm = CandidateNorm(
        repository_id=repository.id,
        cluster_id=cluster.id,
        statement="Use the repository layer.",
        confidence=0.9,
        support_count=1,
        status=status,
        approved=approved,
        decision="approved" if approved else "provisional",
    )
    db.add(norm)
    comment = ReviewComment(
        pull_request_id=pr.id,
        github_id=10,
        body="Use the repository layer.",
        normalized_body="Use the repository layer.",
        is_filtered=False,
    )
    db.add(comment)
    await db.flush()
    db.add(
        CandidateNormEvidence(
            candidate_norm_id=norm.id,
            review_comment_id=comment.id,
            pull_request_id=pr.id,
        )
    )
    db.add(
        CommentEmbedding(
            review_comment_id=comment.id,
            model_name="test",
            source_text_hash="hash",
            dimensions=2,
            vector=[1.0, 0.0],
        )
    )
    await db.commit()
    return repository, pr, norm


@pytest.mark.asyncio
async def test_analysis_missing_diff_context_has_no_findings():
    engine, factory = await database()
    async with factory() as db:
        _, pr, _ = await fixture(db, patch=None)
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        assert result.context_available is False
        assert result.findings == 0
    await engine.dispose()


@pytest.mark.asyncio
async def test_analysis_no_relevant_norms_and_relevant_non_violation():
    engine, factory = await database()
    async with factory() as db:
        _, pr, norm = await fixture(db)
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        finding = (await db.scalars(select(NormFinding))).first()
        assert result.findings == 1
        assert finding.violation_suspected is False
        assert finding.evidence_pull_request_ids == [pr.id]
        norm.status = "deprecated"
        await db.commit()
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        assert result.findings == 0
    await engine.dispose()


@pytest.mark.asyncio
async def test_analysis_no_relevant_norm():
    engine, factory = await database()
    async with factory() as db:
        _, pr, _ = await fixture(db, patch="@@ -1 +1 @@\n+unrelated change")
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        assert result.findings == 0
    await engine.dispose()


@pytest.mark.asyncio
async def test_analysis_suspected_violation_and_feedback():
    engine, factory = await database()
    async with factory() as db:
        _, pr, _ = await fixture(db, patch="@@ -1 +1 @@\n+db.session.execute(query)")
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        finding = (await db.scalars(select(NormFinding))).first()
        assert result.highest_risk in {"low", "medium", "high"}
        assert finding.violation_suspected is True
        feedback = await record_feedback(
            db,
            finding.id,
            decision="dismissed",
            note="Reviewed manually",
            changed_by="dev",
        )
        assert feedback.decision == "dismissed"
    await engine.dispose()


@pytest.mark.asyncio
async def test_reanalysis_keeps_labeled_findings():
    engine, factory = await database()
    async with factory() as db:
        # Enforce foreign keys like Postgres so feedback cannot be orphaned.
        await db.execute(text("PRAGMA foreign_keys=ON"))
        _, pr, norm = await fixture(db, patch="@@ -1 +1 @@\n+db.session.execute(query)")
        await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        finding = (await db.scalars(select(NormFinding))).one()
        await record_feedback(
            db, finding.id, decision="confirmed", note=None, changed_by="dev"
        )
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        assert result.findings == 1
        rerun = (await db.scalars(select(NormFinding))).one()
        assert rerun.id == finding.id
        assert rerun.status == "confirmed"
        assert (await db.scalars(select(NormFindingFeedback))).one().finding_id == finding.id
        norm.status = "deprecated"
        await db.commit()
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        assert result.findings == 0
        kept = (await db.scalars(select(NormFinding))).one()
        assert kept.id == finding.id
        assert kept.current is False
    await engine.dispose()


@pytest.mark.asyncio
async def test_analysis_ignores_norm_terms_only_in_removed_lines():
    engine, factory = await database()
    async with factory() as db:
        _, pr, _ = await fixture(
            db,
            patch="@@ -1,2 +1,2 @@\n-repository.layer.fetch()\n+db.session.execute(query)",
        )
        await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        finding = (await db.scalars(select(NormFinding))).one()
        assert finding.violation_suspected is True
        assert (finding.line_start, finding.line_end) == (1, 2)
    await engine.dispose()


@pytest.mark.asyncio
async def test_conflicting_norm_is_not_used():
    engine, factory = await database()
    async with factory() as db:
        _, pr, _norm = await fixture(db, status="conflicting")
        result = await analyze_pull_request(db, pr.id, embedder=FakeEmbedder())
        assert result.findings == 0
        assert await db.get(NormConflict, 1) is None
    await engine.dispose()
