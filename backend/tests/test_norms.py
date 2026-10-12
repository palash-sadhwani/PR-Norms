import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentCluster,
    Installation,
    NormConflict,
    NormVersion,
    PullRequest,
    Repository,
    ReviewComment,
)
from app.services.norms import (
    approve_norm,
    mark_conflict,
    reject_norm,
    update_norm,
)
from app.services.phase2 import mine_repository


async def _database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, factory


async def _fixture(db):
    installation = Installation(github_id=1, account_login="acme")
    db.add(installation)
    await db.flush()
    repository = Repository(
        installation_id=installation.id, github_id=2, full_name="acme/demo"
    )
    db.add(repository)
    await db.flush()
    pr = PullRequest(
        repository_id=repository.id,
        github_id=3,
        number=1,
        title="Review",
        state="open",
    )
    db.add(pr)
    await db.flush()
    cluster = CommentCluster(
        repository_id=repository.id,
        model_name="test",
        algorithm="hdbscan",
        algorithm_config={},
        label=1,
        size=1,
        confidence=0.8,
    )
    db.add(cluster)
    await db.flush()
    norm = CandidateNorm(
        repository_id=repository.id,
        cluster_id=cluster.id,
        statement="Use the repository layer.",
        confidence=0.8,
        support_count=1,
    )
    db.add(norm)
    await db.flush()
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
    await db.commit()
    return repository, pr, norm


@pytest.mark.asyncio
async def test_norm_lifecycle_preserves_versions_and_evidence():
    engine, factory = await _database()
    async with factory() as db:
        _, pr, norm = await _fixture(db)
        await update_norm(
            db,
            norm.id,
            classification="team_convention",
            status="active",
            reason="Repeated review guidance",
            changed_by="maintainer",
        )
        await approve_norm(
            db, norm.id, changed_by="maintainer", reason="Approved after review"
        )
        await reject_norm(
            db, norm.id, changed_by="maintainer", reason="Superseded later"
        )

        refreshed = await db.get(CandidateNorm, norm.id)
        versions = list(
            (
                await db.scalars(
                    select(NormVersion)
                    .where(NormVersion.candidate_norm_id == norm.id)
                    .order_by(NormVersion.version)
                )
            ).all()
        )
        evidence = list(
            (
                await db.scalars(
                    select(CandidateNormEvidence).where(
                        CandidateNormEvidence.candidate_norm_id == norm.id
                    )
                )
            ).all()
        )
        assert refreshed.decision == "rejected"
        assert refreshed.status == "deprecated"
        assert [item.version for item in versions] == [1, 2, 3, 4]
        assert versions[1].classification == "team_convention"
        assert versions[2].approved is True
        assert versions[3].decision == "rejected"
        assert len(evidence) == 1
        assert evidence[0].pull_request_id == pr.id

    await engine.dispose()


@pytest.mark.asyncio
async def test_phase2_refresh_keeps_norm_history():
    engine, factory = await _database()
    async with factory() as db:
        repository, _, norm = await _fixture(db)
        await approve_norm(db, norm.id, changed_by="maintainer", reason="approved")

        class Embedder:
            model_name = "test"

            def encode(self, texts, batch_size):
                return [[1.0, 0.0] for _ in texts]

        class Clusterer:
            def fit_predict(self, vectors):
                return [1]

        await mine_repository(
            db,
            repository.id,
            embedder=Embedder(),
            clusterer=Clusterer(),
            model_name="test",
            batch_size=1,
        )
        versions = list((await db.scalars(select(NormVersion))).all())
        assert len(versions) == 2
        assert await db.get(CandidateNorm, norm.id) is not None

    await engine.dispose()


@pytest.mark.asyncio
async def test_conflicting_norms_require_evidence_and_preserve_relation():
    engine, factory = await _database()
    async with factory() as db:
        repository, _, first = await _fixture(db)
        cluster = CommentCluster(
            repository_id=repository.id,
            model_name="test",
            algorithm="hdbscan",
            algorithm_config={},
            label=2,
            size=1,
            confidence=0.7,
        )
        db.add(cluster)
        await db.flush()
        second = CandidateNorm(
            repository_id=repository.id,
            cluster_id=cluster.id,
            statement="Avoid the repository layer.",
            confidence=0.7,
            support_count=1,
        )
        db.add(second)
        await db.flush()
        with pytest.raises(ValueError, match="supporting evidence"):
            await mark_conflict(
                db,
                first.id,
                second.id,
                reason="Opposing guidance",
                changed_by="maintainer",
            )
        db.add(
            CandidateNormEvidence(
                candidate_norm_id=second.id,
                review_comment_id=1,
                pull_request_id=1,
            )
        )
        await db.commit()
        await mark_conflict(
            db,
            first.id,
            second.id,
            reason="Opposing guidance",
            changed_by="maintainer",
        )
        conflicts = list((await db.scalars(select(NormConflict))).all())
        assert len(conflicts) == 1
        assert conflicts[0].evidence_count == 2
        assert (await db.get(CandidateNorm, first.id)).status == "conflicting"
        assert (await db.get(CandidateNorm, second.id)).status == "conflicting"
        assert await db.scalar(select(func.count()).select_from(NormVersion)) == 4

    await engine.dispose()
