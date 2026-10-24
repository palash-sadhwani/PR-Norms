"""Track active clusters, active evidence, and current analysis findings."""

import sqlalchemy as sa

from alembic import op

revision = "0006_stable_mining_identity"
down_revision = "0005_phase4_pr_analysis"

FLAGS = (
    ("comment_clusters", "active"),
    ("candidate_norm_evidence", "active"),
    ("norm_findings", "current"),
)


def _has_column(table: str, column: str) -> bool:
    return column in {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)
    }


def upgrade() -> None:
    for table, column in FLAGS:
        if not _has_column(table, column):
            op.add_column(
                table,
                sa.Column(column, sa.Boolean(), nullable=False, server_default=sa.true()),
            )


def downgrade() -> None:
    for table, column in FLAGS:
        if _has_column(table, column):
            with op.batch_alter_table(table) as batch:
                batch.drop_column(column)
