import asyncio

from celery import Celery

from app.core.config import get_settings
from app.db.session import SessionLocal, engine
from app.models import Repository
from app.services.analysis import analyze_pull_request
from app.services.auth import repository_token
from app.services.ingestion import ingest_repository as run_ingestion
from app.services.phase2 import create_embedder
from app.services.phase2 import mine_repository as run_phase2

celery = Celery(
    "prnorms",
    broker=get_settings().redis_url,
    backend=get_settings().redis_url,
    broker_connection_timeout=2,
    broker_connection_retry=False,
)


def _run(main):
    """Run a task coroutine on a fresh event loop.

    Pooled asyncpg connections are bound to the loop that opened them, so the
    pool is emptied before the loop closes; otherwise the next task in this
    worker process would reuse a connection from a dead loop.
    """

    async def wrapper():
        try:
            return await main()
        finally:
            await engine.dispose()

    return asyncio.run(wrapper())


@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def ingest_repository_task(
    self, repository_id: int, job_id: int, target_pr_number: int | None = None
):
    async def run():
        async with SessionLocal() as db:
            repository = await db.get(Repository, repository_id)
            if repository is None:
                raise ValueError(f"repository {repository_id} not found")
            token = await repository_token(db, repository)
            return await run_ingestion(db, job_id, token, target_pr_number)

    _run(run)


ingest_repository = ingest_repository_task


@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def mine_repository_task(self, repository_id: int, model_name: str | None = None):
    async def run():
        async with SessionLocal() as db:
            return await run_phase2(db, repository_id, model_name=model_name)

    return _run(run)


@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def analyze_pull_request_task(
    self, pull_request_id: int, model_name: str | None = None
):
    async def run():
        async with SessionLocal() as db:
            embedder = create_embedder(model_name)
            return await analyze_pull_request(db, pull_request_id, embedder=embedder)

    return _run(run)
