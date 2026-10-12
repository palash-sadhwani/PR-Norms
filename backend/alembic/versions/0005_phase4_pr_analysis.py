"""Add Phase 4 pull-request analysis and provisional norm findings."""

import sqlalchemy as sa

from alembic import op

revision = "0005_phase4_pr_analysis"
down_revision = "0004_phase3_norm_knowledge_base"


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table: str, column: str) -> bool:
    return column in {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)
    }


def upgrade() -> None:
    if not _has_table("pull_request_analyses"):
        op.create_table(
            "pull_request_analyses",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "pull_request_id",
                sa.Integer(),
                sa.ForeignKey("pull_requests.id"),
                nullable=False,
            ),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("formula_version", sa.String(32), nullable=False),
            sa.Column("findings_count", sa.Integer(), nullable=False),
            sa.Column("highest_risk", sa.String(16), nullable=False),
            sa.Column("context_available", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("pull_request_id"),
        )
    if not _has_table("norm_findings"):
        op.create_table(
            "norm_findings",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "analysis_id",
                sa.Integer(),
                sa.ForeignKey("pull_request_analyses.id"),
                nullable=False,
            ),
            sa.Column(
                "candidate_norm_id",
                sa.Integer(),
                sa.ForeignKey("candidate_norms.id"),
                nullable=False,
            ),
            sa.Column(
                "pull_request_file_id",
                sa.Integer(),
                sa.ForeignKey("pull_request_files.id"),
                nullable=True,
            ),
            sa.Column("file_path", sa.String(1024), nullable=True),
            sa.Column("line_start", sa.Integer(), nullable=True),
            sa.Column("line_end", sa.Integer(), nullable=True),
            sa.Column("relevance_score", sa.Float(), nullable=False),
            sa.Column("evidence_strength", sa.Float(), nullable=False),
            sa.Column("risk_score", sa.Float(), nullable=False),
            sa.Column("risk_level", sa.String(16), nullable=False),
            sa.Column("provisional", sa.Boolean(), nullable=False),
            sa.Column("violation_suspected", sa.Boolean(), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column("evidence_comment_ids", sa.JSON(), nullable=False),
            sa.Column("evidence_pull_request_ids", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint(
                "analysis_id", "candidate_norm_id", "file_path", "line_start"
            ),
        )
    elif not _has_column("norm_findings", "evidence_pull_request_ids"):
        op.add_column(
            "norm_findings",
            sa.Column(
                "evidence_pull_request_ids",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            ),
        )
    if not _has_table("norm_finding_feedback"):
        op.create_table(
            "norm_finding_feedback",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "finding_id",
                sa.Integer(),
                sa.ForeignKey("norm_findings.id"),
                nullable=False,
            ),
            sa.Column("decision", sa.String(32), nullable=False),
            sa.Column("note", sa.Text(), nullable=True),
            sa.Column("changed_by", sa.String(255), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("norm_finding_feedback")
    op.drop_table("norm_findings")
    op.drop_table("pull_request_analyses")
