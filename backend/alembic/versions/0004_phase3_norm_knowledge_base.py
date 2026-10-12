"""Add Phase 3 norm classification, lifecycle, and audit history."""

import sqlalchemy as sa

from alembic import op

revision = "0004_phase3_norm_knowledge_base"
down_revision = "0003_phase2_semantic_mining"


def _columns(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def _has_table(table: str) -> bool:
    return table in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    columns = _columns("candidate_norms")
    additions = [
        ("classification", sa.String(32), True),
        ("status", sa.String(32), False, "emerging"),
        ("approved", sa.Boolean(), False, False),
        ("decision", sa.String(32), False, "provisional"),
        ("reviewed_by", sa.String(255), True),
        ("reviewed_at", sa.DateTime(), True),
    ]
    for item in additions:
        name, column_type, nullable, *default = item
        if name not in columns:
            kwargs = {"nullable": nullable}
            if default:
                kwargs["server_default"] = sa.text(
                    "NULL"
                    if default[0] is None
                    else ("FALSE" if default[0] is False else f"'{default[0]}'")
                )
            op.add_column(
                "candidate_norms",
                sa.Column(name, column_type, **kwargs),
            )

    if not _has_table("norm_versions"):
        op.create_table(
            "norm_versions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "candidate_norm_id",
                sa.Integer(),
                sa.ForeignKey("candidate_norms.id"),
                nullable=False,
            ),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("statement", sa.Text(), nullable=False),
            sa.Column("classification", sa.String(32), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("approved", sa.Boolean(), nullable=False),
            sa.Column("decision", sa.String(32), nullable=False),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("changed_by", sa.String(255), nullable=True),
            sa.Column("evidence_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("candidate_norm_id", "version"),
        )
    if not _has_table("norm_conflicts"):
        op.create_table(
            "norm_conflicts",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "candidate_norm_id",
                sa.Integer(),
                sa.ForeignKey("candidate_norms.id"),
                nullable=False,
            ),
            sa.Column(
                "conflicting_norm_id",
                sa.Integer(),
                sa.ForeignKey("candidate_norms.id"),
                nullable=False,
            ),
            sa.Column("reason", sa.Text(), nullable=False),
            sa.Column("evidence_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("candidate_norm_id", "conflicting_norm_id"),
        )


def downgrade() -> None:
    op.drop_table("norm_conflicts")
    op.drop_table("norm_versions")
    for name in (
        "reviewed_at",
        "reviewed_by",
        "decision",
        "approved",
        "status",
        "classification",
    ):
        op.drop_column("candidate_norms", name)
