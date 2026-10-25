"""Idempotent GitHub collection and checkpoint persistence."""

from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Commit,
    IngestionCheckpoint,
    IngestionJob,
    IssueComment,
    PullRequest,
    PullRequestFile,
    Repository,
    Review,
    ReviewComment,
    ReviewThread,
    User,
    WebhookDelivery,
)
from app.services.github import GitHubClient, GitHubRequestError
from app.services.preprocess import preprocess_comment

THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          id isResolved
          isOutdated
          path line
          comments(first: 100) { nodes { databaseId } }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC).replace(tzinfo=None)


async def upsert_user(db: AsyncSession, payload: dict | None) -> User | None:
    if not payload or not payload.get("id"):
        return None
    user = await db.scalar(select(User).where(User.github_id == payload["id"]))
    if user is None:
        user = User(github_id=payload["id"], login=payload.get("login", "unknown"))
        db.add(user)
    user.login = payload.get("login", user.login)
    user.name = payload.get("name") or payload.get("login")
    return user


async def upsert_pull_request(
    db: AsyncSession, repository: Repository, payload: dict
) -> PullRequest:
    pr = await db.scalar(
        select(PullRequest).where(
            PullRequest.repository_id == repository.id,
            PullRequest.number == payload["number"],
        )
    )
    if pr is None:
        pr = PullRequest(repository_id=repository.id, number=payload["number"])
        db.add(pr)
    pr.github_id = payload["id"]
    pr.title = payload.get("title") or ""
    pr.body = payload.get("body")
    pr.state = payload.get("state", "unknown")
    pr.merged = bool(payload.get("merged") or payload.get("merged_at"))
    pr.base_branch = (payload.get("base") or {}).get("ref")
    pr.head_branch = (payload.get("head") or {}).get("ref")
    pr.base_sha = (payload.get("base") or {}).get("sha")
    pr.head_sha = (payload.get("head") or {}).get("sha")
    pr.opened_at = parse_datetime(payload.get("created_at"))
    pr.closed_at = parse_datetime(payload.get("closed_at"))
    pr.merged_at = parse_datetime(payload.get("merged_at"))
    pr.raw = payload
    author = await upsert_user(db, payload.get("user"))
    pr.author_id = author.id if author else None
    await db.flush()
    return pr


async def upsert_review(db: AsyncSession, pr: PullRequest, payload: dict) -> Review:
    review = await db.scalar(
        select(Review).where(
            Review.pull_request_id == pr.id, Review.github_id == payload["id"]
        )
    )
    if review is None:
        review = Review(pull_request_id=pr.id, github_id=payload["id"])
        db.add(review)
    review.state = payload.get("state", "COMMENTED")
    review.body = payload.get("body")
    review.submitted_at = parse_datetime(payload.get("submitted_at"))
    review.raw = payload
    author = await upsert_user(db, payload.get("user"))
    review.author_id = author.id if author else None
    await db.flush()
    return review


async def upsert_review_comment(
    db: AsyncSession, pr: PullRequest, payload: dict, review: Review | None = None
) -> ReviewComment:
    comment = await db.scalar(
        select(ReviewComment).where(ReviewComment.github_id == payload["id"])
    )
    if comment is None:
        comment = ReviewComment(pull_request_id=pr.id, github_id=payload["id"], body="")
        db.add(comment)
    author = await upsert_user(db, payload.get("user"))
    processed = preprocess_comment(
        payload.get("body"), (payload.get("user") or {}).get("login")
    )
    comment.pull_request_id = pr.id
    comment.review_id = review.id if review else None
    comment.author_id = author.id if author else None
    comment.body = processed.raw_text
    comment.normalized_body = processed.normalized_text
    comment.is_filtered = processed.is_low_information
    comment.filter_reason = processed.reason
    comment.path = payload.get("path")
    comment.line = payload.get("line")
    comment.start_line = payload.get("start_line")
    comment.original_line = payload.get("original_line")
    comment.position = payload.get("position")
    comment.diff_hunk = payload.get("diff_hunk")
    comment.commit_sha = payload.get("commit_id")
    comment.in_reply_to_id = payload.get("in_reply_to_id")
    comment.is_outdated = (
        payload.get("position") is None and payload.get("line") is None
    )
    comment.raw = payload
    await db.flush()
    return comment


async def upsert_issue_comment(
    db: AsyncSession, pr: PullRequest, payload: dict
) -> IssueComment:
    comment = await db.scalar(
        select(IssueComment).where(IssueComment.github_id == payload["id"])
    )
    if comment is None:
        comment = IssueComment(pull_request_id=pr.id, github_id=payload["id"], body="")
        db.add(comment)
    author = await upsert_user(db, payload.get("user"))
    processed = preprocess_comment(
        payload.get("body"), (payload.get("user") or {}).get("login")
    )
    comment.author_id = author.id if author else None
    comment.body = processed.raw_text
    comment.normalized_body = processed.normalized_text
    comment.is_filtered = processed.is_low_information
    comment.filter_reason = processed.reason
    comment.raw = payload
    await db.flush()
    return comment


async def collect_threads(
    db: AsyncSession, github: GitHubClient, repository: Repository, pr: PullRequest
) -> None:
    owner, name = repository.full_name.split("/", 1)
    cursor: str | None = None
    while True:
        data = await github.graphql(
            THREADS_QUERY,
            {"owner": owner, "name": name, "number": pr.number, "cursor": cursor},
        )
        connection = data["repository"]["pullRequest"]["reviewThreads"]
        for payload in connection["nodes"]:
            thread = await db.scalar(
                select(ReviewThread).where(ReviewThread.github_node_id == payload["id"])
            )
            if thread is None:
                thread = ReviewThread(
                    pull_request_id=pr.id, github_node_id=payload["id"]
                )
                db.add(thread)
            thread.pull_request_id = pr.id
            thread.is_resolved = bool(payload.get("isResolved"))
            thread.is_outdated = bool(payload.get("isOutdated"))
            thread.path = payload.get("path")
            thread.line = payload.get("line")
            thread.raw = payload
            await db.flush()
            for comment_node in (payload.get("comments") or {}).get("nodes", []):
                database_id = comment_node.get("databaseId")
                if database_id is None:
                    continue
                comment = await db.scalar(
                    select(ReviewComment).where(ReviewComment.github_id == database_id)
                )
                if comment is not None:
                    comment.thread_id = thread.id
                    comment.is_resolved = thread.is_resolved
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            return
        cursor = page_info["endCursor"]


async def ingest_repository(
    db: AsyncSession,
    job_id: int,
    token: str,
    target_pr_number: int | None = None,
    max_pull_requests: int | None = None,
    recent_closed_only: bool = False,
    include_open: bool = False,
) -> IngestionJob:
    job = await db.get(IngestionJob, job_id)
    if job is None:
        raise ValueError(f"ingestion job {job_id} not found")
    repository_id = job.repository_id
    repository = await db.get(Repository, repository_id)
    if repository is None:
        raise ValueError(f"repository {repository_id} not found")
    job.status = "running"
    job.error = None
    await db.commit()
    checkpoint = await db.scalar(
        select(IngestionCheckpoint).where(
            IngestionCheckpoint.job_id == job.id,
            IngestionCheckpoint.resource == "pull_requests",
        )
    )
    last_number = (
        0
        if target_pr_number is not None or recent_closed_only
        else int(checkpoint.cursor) if checkpoint and checkpoint.cursor else 0
    )
    resume_offset = (
        int(checkpoint.cursor)
        if recent_closed_only and checkpoint and checkpoint.cursor
        else 0
    )
    processed_count = 0
    seen_count = 0
    seen_github_ids: set[int] = set()
    rate_limit: GitHubRequestError | None = None
    try:
        async with GitHubClient(token) as github:

            async def pull_request_items():
                if target_pr_number is not None:
                    yield (
                        await github.get(
                            f"/repos/{repository.full_name}/pulls/{target_pr_number}"
                        )
                    ).json()
                    return
                async for item in github.paginate(
                    f"/repos/{repository.full_name}/pulls",
                    state="closed" if recent_closed_only and not include_open else "all",
                    sort="updated" if recent_closed_only else "created",
                    direction="desc" if recent_closed_only else "asc",
                ):
                    yield item

            async for payload in pull_request_items():
                if (
                    max_pull_requests is not None
                    and processed_count >= max_pull_requests
                ):
                    break
                github_id = payload.get("id")
                if github_id in seen_github_ids:
                    continue
                seen_github_ids.add(github_id)
                if recent_closed_only:
                    if seen_count < resume_offset:
                        seen_count += 1
                        continue
                elif payload["number"] <= last_number:
                    continue
                seen_count += 1
                processed_count += 1
                job.total_items += 1
                job.current_stage = f"pull_request:{payload['number']}"
                try:
                    pr = await upsert_pull_request(db, repository, payload)
                    reviews = [
                        item
                        async for item in github.paginate(
                            f"/repos/{repository.full_name}/pulls/{pr.number}/reviews"
                        )
                    ]
                    review_map = {}
                    for review_payload in reviews:
                        review = await upsert_review(db, pr, review_payload)
                        review_map[review_payload["id"]] = review
                    for item in [
                        item
                        async for item in github.paginate(
                            f"/repos/{repository.full_name}/pulls/{pr.number}/comments"
                        )
                    ]:
                        await upsert_review_comment(
                            db,
                            pr,
                            item,
                            review_map.get(item.get("pull_request_review_id")),
                        )
                    for item in [
                        item
                        async for item in github.paginate(
                            f"/repos/{repository.full_name}/issues/{pr.number}/comments"
                        )
                    ]:
                        await upsert_issue_comment(db, pr, item)
                    for item in [
                        item
                        async for item in github.paginate(
                            f"/repos/{repository.full_name}/pulls/{pr.number}/files"
                        )
                    ]:
                        file = await db.scalar(
                            select(PullRequestFile).where(
                                PullRequestFile.pull_request_id == pr.id,
                                PullRequestFile.filename == item["filename"],
                            )
                        )
                        if file is None:
                            file = PullRequestFile(
                                pull_request_id=pr.id, filename=item["filename"]
                            )
                            db.add(file)
                        file.status = item.get("status")
                        file.additions = item.get("additions", 0)
                        file.deletions = item.get("deletions", 0)
                        file.patch = item.get("patch")
                        file.raw = item
                    for item in [
                        item
                        async for item in github.paginate(
                            f"/repos/{repository.full_name}/pulls/{pr.number}/commits"
                        )
                    ]:
                        commit = await db.scalar(
                            select(Commit).where(Commit.sha == item["sha"])
                        )
                        if commit is None:
                            commit = Commit(
                                pull_request_id=pr.id,
                                sha=item["sha"],
                                message=(item.get("commit") or {}).get("message", ""),
                            )
                            db.add(commit)
                        commit.pull_request_id = pr.id
                        commit.message = (item.get("commit") or {}).get("message", "")
                        commit.raw = item
                    # The GraphQL API rejects anonymous requests.
                    if token:
                        try:
                            await collect_threads(db, github, repository, pr)
                        except (httpx.HTTPError, KeyError, RuntimeError):
                            pass
                    last_number = payload["number"]
                    if checkpoint is None:
                        checkpoint = IngestionCheckpoint(
                            job_id=job.id,
                            resource="pull_requests",
                            cursor=str(
                                seen_count if recent_closed_only else last_number
                            ),
                        )
                        db.add(checkpoint)
                    else:
                        checkpoint.cursor = str(
                            seen_count if recent_closed_only else last_number
                        )
                    job.completed_items += 1
                    await db.commit()
                except (
                    httpx.HTTPError,
                    KeyError,
                    RuntimeError,
                    SQLAlchemyError,
                    ValueError,
                ) as exc:
                    # A rollback expires every loaded object; reload the ones the
                    # loop reads, or the next attribute access does implicit IO
                    # outside the async context (MissingGreenlet).
                    await db.rollback()
                    job = await db.get(IngestionJob, job_id)
                    if job is None:
                        raise ValueError(f"ingestion job {job_id} disappeared") from exc
                    repository = await db.get(Repository, repository_id)
                    if repository is None:
                        raise ValueError(f"repository {repository_id} disappeared") from exc
                    # The rollback undid this PR's total_items increment.
                    job.total_items += 1
                    job.error_count += 1
                    job.error = str(exc)
                    await db.commit()
                    checkpoint = await db.scalar(
                        select(IngestionCheckpoint).where(
                            IngestionCheckpoint.job_id == job_id,
                            IngestionCheckpoint.resource == "pull_requests",
                        )
                    )
                    if isinstance(exc, GitHubRequestError) and exc.rate_limited:
                        # Every further request would fail too; stop and let a
                        # retry resume from the checkpoint after the reset.
                        rate_limit = exc
                        break
        if rate_limit is not None:
            job.status = "rate_limited"
        else:
            job.status = "partial_failed" if job.error_count else "completed"
        job.current_stage = None
        await db.commit()
        return job
    except Exception as exc:
        await db.rollback()
        job = await db.get(IngestionJob, job_id)
        if job:
            rate_limited = isinstance(exc, GitHubRequestError) and exc.rate_limited
            job.status = "rate_limited" if rate_limited else "failed"
            job.error = str(exc)
            await db.commit()
        raise


async def checkpoint(
    db: AsyncSession, job_id: int, resource: str, cursor: str | None
) -> None:
    item = await db.scalar(
        select(IngestionCheckpoint).where(
            IngestionCheckpoint.job_id == job_id,
            IngestionCheckpoint.resource == resource,
        )
    )
    if item:
        item.cursor = cursor
    else:
        db.add(IngestionCheckpoint(job_id=job_id, resource=resource, cursor=cursor))
    await db.commit()


async def accept_delivery(
    db: AsyncSession, delivery_id: str, event: str, payload: dict
) -> bool:
    if await db.scalar(
        select(WebhookDelivery).where(WebhookDelivery.delivery_id == delivery_id)
    ):
        return False
    db.add(WebhookDelivery(delivery_id=delivery_id, event=event, payload=payload))
    await db.commit()
    return True
