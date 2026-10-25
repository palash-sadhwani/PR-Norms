import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app import tasks

BACKEND = Path(__file__).resolve().parents[1]


def test_each_task_disposes_engine_on_its_own_loop(monkeypatch):
    disposed = []

    class Engine:
        async def dispose(self):
            disposed.append(True)

    async def fake_mine(db, repository_id, model_name=None):
        return repository_id

    monkeypatch.setattr(tasks, "engine", Engine())
    monkeypatch.setattr(tasks, "run_phase2", fake_mine)
    assert tasks.mine_repository_task(1) == 1
    assert tasks.mine_repository_task(2) == 2
    assert len(disposed) == 2


@pytest.mark.skipif(
    not os.environ.get("PRNORMS_TEST_POSTGRES_URL"),
    reason="set PRNORMS_TEST_POSTGRES_URL to an asyncpg URL to run",
)
def test_consecutive_tasks_share_worker_process_on_postgres():
    # asyncpg binds connections to their event loop; SQLite does not reproduce this.
    script = textwrap.dedent(
        """
        from sqlalchemy import text
        from app import tasks

        async def query(db, repository_id, model_name=None):
            return (await db.execute(text("select cast(:n as integer)"), {"n": repository_id})).scalar()

        tasks.run_phase2 = query
        print([tasks.mine_repository_task(n) for n in (1, 2, 3)])
        """
    )
    env = {**os.environ, "DATABASE_URL": os.environ["PRNORMS_TEST_POSTGRES_URL"]}
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[1, 2, 3]"
