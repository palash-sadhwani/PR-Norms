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
    IssueComment,
    NormVersion,
    PullRequest,
    Repository,
    ReviewComment,
    User,
)
from app.services.norms import approve_norm
from app.services.phase2 import cosine_similarity, mine_repository


class TopicEmbedder:
    model_name = "topic-model"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size):
        self.calls += 1
        vectors = []
        for text in texts:
            folded = text.casefold()
            if any(word in folded for word in ("repository", "database", "query", "layer")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(word in folded for word in ("logger", "print", "logging", "structured")):
                vectors.append([0.0, 1.0, 0.0])
            elif any(word in folded for word in ("test", "coverage", "assertion")):
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class NoiseClusterer:
    def fit_predict(self, vectors):
        return [-1] * len(vectors)


class ChainedEmbedder:
    model_name = "chained-topic-model"

    def encode(self, texts, batch_size):
        return [
            [1.0, 0.0]
            if "belongs" in text
            else [0.766, 0.643]
            if "api" in text
            else [0.174, 0.985]
            for text in texts
        ]


class ScriptedClusterer:
    def __init__(self, labels):
        self.labels = labels

    def fit_predict(self, vectors):
        assert len(vectors) == len(self.labels)
        return list(self.labels)


async def setup_database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, factory


async def add_repository(db, name="acme/demo"):
    installation = Installation(github_id=100, account_login="acme")
    db.add(installation)
    await db.flush()
    repository = Repository(
        installation_id=installation.id, github_id=101, full_name=name
    )
    reviewer = User(github_id=102, login="reviewer-one")
    db.add_all([repository, reviewer])
    await db.flush()
    return repository, reviewer


async def add_comments(db, repository, reviewer, comments):
    prs = {}
    created = []
    for index, (pr_number, body) in enumerate(comments):
        pr = prs.get(pr_number)
        if pr is None:
            pr = PullRequest(
                repository_id=repository.id,
                github_id=1000 + pr_number,
                number=pr_number,
                title=f"PR {pr_number}",
                state="closed",
            )
            db.add(pr)
            await db.flush()
            prs[pr_number] = pr
        comment = ReviewComment(
            pull_request_id=pr.id,
            github_id=2000 + index,
            author_id=reviewer.id,
            body=body,
            normalized_body=body,
            path="app/example.py",
            line=10 + index,
            diff_hunk="@@ -1 +1 @@",
            is_filtered=False,
        )
        db.add(comment)
        created.append(comment)
    await db.flush()
    return created, prs


REPOSITORY_COMMENTS = [
    "Move database access into the repository layer.",
    "Keep database queries inside the repository layer.",
    "Database access belongs in the repository layer.",
]
LOGGING_COMMENTS = [
    "Use the logger instead of print statements.",
    "Replace print statements with the logger.",
    "Please use structured logging rather than print.",
]


@pytest.mark.asyncio
async def test_noise_comments_recover_across_multiple_pull_requests():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        comments, prs = await add_comments(
            db,
            repository,
            reviewer,
            [(1, REPOSITORY_COMMENTS[0]), (2, REPOSITORY_COMMENTS[1]), (3, REPOSITORY_COMMENTS[2])],
        )
        await db.commit()
        result = await mine_repository(
            db, repository.id, embedder=TopicEmbedder(), clusterer=NoiseClusterer()
        )

        assert result.hdbscan_clusters == 0
        assert result.recovered_comments == 3
        assert result.clusters == result.candidate_norms_generated == 1
        assert result.noise == 0
        norm = await db.scalar(select(CandidateNorm))
        assert norm is not None
        assert norm.support_count == 3
        assert norm.statement in REPOSITORY_COMMENTS
        evidence = list((await db.scalars(select(CandidateNormEvidence))).all())
        assert {item.pull_request_id for item in evidence} == {
            pr.id for pr in prs.values()
        }
        assert {item.review_comment_id for item in evidence} == {
            comment.id for comment in comments
        }
        assert all(comment.author_id == reviewer.id for comment in comments)

    await engine.dispose()


@pytest.mark.asyncio
async def test_three_similar_comments_from_one_pr_fail_pr_diversity():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [(1, body) for body in REPOSITORY_COMMENTS],
        )
        await db.commit()

        result = await mine_repository(
            db, repository.id, embedder=TopicEmbedder(), clusterer=NoiseClusterer()
        )

        assert result.recovered_comments == 3
        assert result.rejected_clusters == 1
        assert result.rejected_insufficient_pr_diversity == 1
        assert result.clusters == result.norms == 0
        assert result.zero_norm_explanation == (
            "Clusters did not have evidence across enough distinct pull requests."
        )
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_unrelated_comments_are_not_combined_into_noise_cluster():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [
                (1, "Add tests covering this change."),
                (2, "Validate user input before saving."),
                (3, "Avoid loading the entire dataset into memory."),
            ],
        )
        await db.commit()

        result = await mine_repository(
            db, repository.id, embedder=TopicEmbedder(), clusterer=NoiseClusterer()
        )

        assert result.clusters == result.norms == 0
        assert result.recovered_comments == 0
        assert result.noise == 3
        assert result.zero_norm_explanation == (
            "Comments were too dissimilar to form meaningful clusters."
        )

    await engine.dispose()


@pytest.mark.asyncio
async def test_noise_recovery_does_not_chain_adjacent_topics():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [
                (1, "The repository layer belongs in settings."),
                (2, "The repository layer API configuration is important."),
                (3, "The repository layer logging configuration is important."),
            ],
        )
        await db.commit()

        result = await mine_repository(
            db, repository.id, embedder=ChainedEmbedder(), clusterer=NoiseClusterer()
        )

        assert result.recovered_comments == 0
        assert result.clusters == 0
        assert result.noise == 3
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_generic_words_do_not_form_a_noise_cluster():
    engine, factory = await setup_database()
    generic_comments = [
        "Please consider doing this here.",
        "Please consider changing this here.",
        "Please consider keeping this here.",
    ]
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [(index + 1, body) for index, body in enumerate(generic_comments)],
        )
        await db.commit()

        result = await mine_repository(
            db, repository.id, embedder=TopicEmbedder(), clusterer=NoiseClusterer()
        )

        assert result.recovered_comments == 0
        assert result.clusters == 0
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_hdbscan_clusters_that_meet_evidence_requirements_are_preserved():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [
                (1, REPOSITORY_COMMENTS[0]),
                (2, REPOSITORY_COMMENTS[1]),
                (3, REPOSITORY_COMMENTS[2]),
            ],
        )
        await db.commit()

        result = await mine_repository(
            db,
            repository.id,
            embedder=TopicEmbedder(),
            clusterer=ScriptedClusterer([7, 7, 7]),
        )

        assert result.hdbscan_clusters == 1
        assert result.recovered_comments == 0
        assert result.clusters == 1
        assert result.rejected_clusters == 0
        assert await db.scalar(select(func.count()).select_from(CommentCluster)) == 1
        assert await db.scalar(select(func.count()).select_from(CommentClusterMember)) == 3

    await engine.dispose()


@pytest.mark.asyncio
async def test_embedding_cache_is_reused():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [(1, REPOSITORY_COMMENTS[0]), (2, REPOSITORY_COMMENTS[1]), (3, REPOSITORY_COMMENTS[2])],
        )
        await db.commit()
        embedder = TopicEmbedder()
        first = await mine_repository(
            db, repository.id, embedder=embedder, clusterer=NoiseClusterer()
        )
        second = await mine_repository(
            db, repository.id, embedder=embedder, clusterer=NoiseClusterer()
        )

        assert first.embeddings_created == 3
        assert second.embeddings_reused == 3
        assert second.embeddings_created == 0
        assert embedder.calls == 1
        assert await db.scalar(select(func.count()).select_from(CommentEmbedding)) == 3

    await engine.dispose()


@pytest.mark.asyncio
async def test_repeated_mining_does_not_duplicate_candidate_norms():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [(1, REPOSITORY_COMMENTS[0]), (2, REPOSITORY_COMMENTS[1]), (3, REPOSITORY_COMMENTS[2])],
        )
        await db.commit()
        embedder = TopicEmbedder()
        first = await mine_repository(
            db, repository.id, embedder=embedder, clusterer=NoiseClusterer()
        )
        second = await mine_repository(
            db, repository.id, embedder=embedder, clusterer=NoiseClusterer()
        )

        assert first.norms_created == 1
        assert second.norms_created == 0
        assert second.norms == second.candidate_norms_generated == 1
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_approved_norm_keeps_identity_when_cluster_labels_change():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        await add_comments(
            db,
            repository,
            reviewer,
            [
                (1, REPOSITORY_COMMENTS[0]),
                (2, REPOSITORY_COMMENTS[1]),
                (3, REPOSITORY_COMMENTS[2]),
                (4, LOGGING_COMMENTS[0]),
                (5, LOGGING_COMMENTS[1]),
                (6, LOGGING_COMMENTS[2]),
            ],
        )
        await db.commit()
        embedder = TopicEmbedder()

        async def mine(labels):
            return await mine_repository(
                db, repository.id, embedder=embedder, clusterer=ScriptedClusterer(labels)
            )

        await mine([10, 10, 10, 20, 20, 20])
        norm = await db.scalar(
            select(CandidateNorm).where(
                CandidateNorm.statement.contains("repository")
            )
        )
        assert norm is not None
        old_id = norm.id
        old_statement = norm.statement
        old_evidence_ids = set(
            (
                await db.scalars(
                    select(CandidateNormEvidence.review_comment_id).where(
                        CandidateNormEvidence.candidate_norm_id == norm.id,
                        CandidateNormEvidence.active.is_(True),
                    )
                )
            ).all()
        )
        await approve_norm(db, norm.id, changed_by="reviewer", reason="Team agreed")

        result = await mine([20, 20, 20, 10, 10, 10])
        await db.refresh(norm)

        assert result.norms_created == 0
        assert norm.id == old_id
        assert norm.approved is True
        assert norm.statement == old_statement
        current_evidence_ids = set(
            (
                await db.scalars(
                    select(CandidateNormEvidence.review_comment_id).where(
                        CandidateNormEvidence.candidate_norm_id == norm.id,
                        CandidateNormEvidence.active.is_(True),
                    )
                )
            ).all()
        )
        assert current_evidence_ids == old_evidence_ids
        assert await db.scalar(select(func.count()).select_from(CandidateNorm)) == 2

    await engine.dispose()


@pytest.mark.asyncio
async def test_disappeared_evidence_flags_candidate_and_keeps_history():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        comments, _ = await add_comments(
            db,
            repository,
            reviewer,
            [
                (1, REPOSITORY_COMMENTS[0]),
                (2, REPOSITORY_COMMENTS[1]),
                (3, REPOSITORY_COMMENTS[2]),
                (4, LOGGING_COMMENTS[0]),
                (5, LOGGING_COMMENTS[1]),
                (6, LOGGING_COMMENTS[2]),
            ],
        )
        await db.commit()
        embedder = TopicEmbedder()
        await mine_repository(
            db,
            repository.id,
            embedder=embedder,
            clusterer=ScriptedClusterer([0, 0, 0, 1, 1, 1]),
        )
        norm = await db.scalar(
            select(CandidateNorm).where(
                CandidateNorm.statement.contains("repository")
            )
        )
        assert norm is not None
        for comment in comments[:3]:
            comment.is_filtered = True
            comment.filter_reason = "fixture_removed"
        await db.commit()

        result = await mine_repository(
            db,
            repository.id,
            embedder=embedder,
            clusterer=ScriptedClusterer([0, 0, 0]),
        )
        await db.refresh(norm)

        assert result.norms_retired == 1
        assert norm.status == "weakening"
        assert norm.support_count == 0
        active_evidence = await db.scalar(
            select(func.count()).select_from(CandidateNormEvidence).where(
                CandidateNormEvidence.candidate_norm_id == norm.id,
                CandidateNormEvidence.active.is_(True),
            )
        )
        assert active_evidence == 0
        historical_evidence = list(
            (
                await db.scalars(
                    select(CandidateNormEvidence).where(
                        CandidateNormEvidence.candidate_norm_id == norm.id
                    )
                )
            ).all()
        )
        assert len(historical_evidence) == 3
        assert all(not evidence.active for evidence in historical_evidence)
        versions = list(
            (
                await db.scalars(
                    select(NormVersion).where(
                        NormVersion.candidate_norm_id == norm.id
                    )
                )
            ).all()
        )
        assert any("no longer finds" in (version.reason or "") for version in versions)

    await engine.dispose()


@pytest.mark.asyncio
async def test_zero_eligible_comments_explain_exclusions():
    engine, factory = await setup_database()
    async with factory() as db:
        repository, reviewer = await add_repository(db)
        comments, prs = await add_comments(
            db,
            repository,
            reviewer,
            [
                (1, "A previously filtered comment."),
                (2, "LGTM"),
            ],
        )
        comments[0].is_filtered = True
        comments[0].filter_reason = "low_information"
        conversation_pr = prs[1]
        db.add(
            IssueComment(
                pull_request_id=conversation_pr.id,
                github_id=9000,
                author_id=reviewer.id,
                body="Thanks, this looks good.",
                normalized_body="Thanks, this looks good.",
            )
        )
        await db.commit()

        result = await mine_repository(
            db, repository.id, embedder=TopicEmbedder(), clusterer=NoiseClusterer()
        )

        assert result.total_imported == 3
        assert result.eligible == 0
        assert result.excluded == 3
        assert result.exclusion_reasons == {
            "low_information": 2,
            "general_pr_conversation": 1,
        }
        assert result.zero_norm_explanation == (
            "No eligible inline review comments were found."
        )
        assert result.norms == 0

    await engine.dispose()


def test_cosine_similarity_and_unloadable_embedding_model(monkeypatch):
    import sys
    import types

    from app.services.phase2 import SentenceTransformerEmbedder

    assert cosine_similarity([1, 0], [1, 0]) == 1.0

    def unavailable(name):
        raise OSError("403 Forbidden")

    module = types.SimpleNamespace(SentenceTransformer=unavailable)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        SentenceTransformerEmbedder("missing/model")
