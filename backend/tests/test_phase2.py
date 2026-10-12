import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentCluster,
    CommentClusterMember,
    CommentEmbedding,
    Installation,
    PullRequest,
    Repository,
    ReviewComment,
)
from app.services.phase2 import cosine_similarity, mine_repository


class FakeEmbedder:
    model_name = "test-model"

    def encode(self, texts, batch_size):
        assert batch_size == 2
        return [[1.0, 0.0] if "repository" in text else [0.98, 0.02] for text in texts]


class FakeClusterer:
    def fit_predict(self, vectors):
        assert len(vectors) == 2
        return [0, 0]


class SingleCommentEmbedder:
    model_name = "single-comment-model"

    def encode(self, texts, batch_size):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_phase2_persists_embeddings_clusters_and_traceable_evidence():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with factory() as db:
        installation = Installation(github_id=1, account_login="acme")
        db.add(installation)
        await db.flush()
        repository = Repository(
            installation_id=installation.id,
            github_id=2,
            full_name="acme/demo",
        )
        db.add(repository)
        await db.flush()
        pr = PullRequest(
            repository_id=repository.id,
            github_id=3,
            number=1,
            title="Storage",
            state="open",
        )
        db.add(pr)
        await db.flush()
        db.add_all(
            [
                ReviewComment(
                    pull_request_id=pr.id,
                    github_id=10,
                    body="Use the repository layer.",
                    normalized_body="Use the repository layer.",
                    is_filtered=False,
                ),
                ReviewComment(
                    pull_request_id=pr.id,
                    github_id=11,
                    body="Keep database access in the repository layer.",
                    normalized_body="Keep database access in the repository layer.",
                    is_filtered=False,
                ),
            ]
        )
        await db.commit()

        result = await mine_repository(
            db,
            repository.id,
            embedder=FakeEmbedder(),
            clusterer=FakeClusterer(),
            batch_size=2,
        )

        assert result.embedded == 2
        assert result.clustered == 2
        assert result.clusters == 1
        assert result.norms == 1
        assert result.eligible == 2
        assert result.excluded == 0
        assert result.embeddings_created == 2
        assert result.embeddings_reused == 0
        assert await db.scalar(select(func.count()).select_from(CommentCluster)) == 1
        assert (
            await db.scalar(select(func.count()).select_from(CommentClusterMember)) == 2
        )
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 1
        evidence = list((await db.scalars(select(CandidateNormEvidence))).all())
        assert {item.review_comment_id for item in evidence} == {1, 2}
        assert {item.pull_request_id for item in evidence} == {pr.id}
        norm = await db.scalar(select(CandidateNorm))
        assert norm.support_count == 2
        assert 0.0 <= norm.confidence <= 1.0
        assert cosine_similarity([1, 0], [1, 0]) == 1.0

        rerun = await mine_repository(
            db,
            repository.id,
            embedder=FakeEmbedder(),
            clusterer=FakeClusterer(),
            batch_size=2,
        )
        assert rerun.norms == 1
        assert rerun.embeddings_reused == 2
        assert rerun.embeddings_created == 0
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 1
        comment = await db.scalar(
            select(ReviewComment).where(ReviewComment.github_id == 10)
        )
        assert comment.body == "Use the repository layer."
        assert comment.pull_request_id == pr.id

    await engine.dispose()


@pytest.mark.asyncio
async def test_phase2_handles_fewer_comments_than_cluster_size():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with factory() as db:
        installation = Installation(github_id=21, account_login="small")
        db.add(installation)
        await db.flush()
        repository = Repository(
            installation_id=installation.id,
            github_id=22,
            full_name="small/demo",
        )
        db.add(repository)
        await db.flush()
        pr = PullRequest(
            repository_id=repository.id,
            github_id=23,
            number=1,
            title="Small",
            state="open",
        )
        db.add(pr)
        await db.flush()
        db.add(
            ReviewComment(
                pull_request_id=pr.id,
                github_id=24,
                body="Keep this simple.",
                normalized_body="Keep this simple.",
                is_filtered=False,
            )
        )
        await db.commit()

        result = await mine_repository(
            db, repository.id, embedder=SingleCommentEmbedder(), batch_size=1
        )

        assert result.embedded == 1
        assert result.clustered == 0
        assert result.clusters == 0
        assert result.norms == 0
        assert result.eligible == 1
        assert result.embeddings_created == 1
        assert await db.scalar(select(func.count()).select_from(CommentEmbedding)) == 1

    await engine.dispose()
