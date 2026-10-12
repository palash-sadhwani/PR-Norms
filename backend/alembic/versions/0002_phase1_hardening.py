"""Add thread links and targeted ingestion fields to existing installations."""

import sqlalchemy as sa

from alembic import op

revision = "0002_phase1_hardening"
down_revision = "0001_initial"


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    if "thread_id" not in _columns("review_comments"):
        op.add_column(
            "review_comments",
            sa.Column("thread_id", sa.Integer(), nullable=True),
        )
        op.create_index(
            "ix_review_comments_thread_id",
            "review_comments",
            ["thread_id"],
        )
        op.create_foreign_key(
            "fk_review_comments_thread_id",
            "review_comments",
            "review_threads",
            ["thread_id"],
            ["id"],
        )

    if "target_pr_number" not in _columns("ingestion_jobs"):
        op.add_column(
            "ingestion_jobs",
            sa.Column("target_pr_number", sa.Integer(), nullable=True),
        )

    thread_columns = _columns("review_threads")
    if "github_node_id" not in thread_columns:
        op.add_column(
            "review_threads",
            sa.Column("github_node_id", sa.String(length=255), nullable=True),
        )
        if "github_id" in thread_columns:
            op.execute(
                "UPDATE review_threads "
                "SET github_node_id = CAST(github_id AS VARCHAR) "
                "WHERE github_node_id IS NULL"
            )
        op.alter_column("review_threads", "github_node_id", nullable=False)
        op.create_unique_constraint(
            "uq_review_threads_github_node_id",
            "review_threads",
            ["github_node_id"],
        )


def downgrade() -> None:
    # Keep this migration non-destructive for preserved research data.
    pass
