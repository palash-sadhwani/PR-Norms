"""Add Phase 2 embeddings, semantic clusters, and candidate norms."""

import sqlalchemy as sa

from alembic import op

revision = "0003_phase2_semantic_mining"
down_revision = "0002_phase1_hardening"


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("comment_embeddings"):
        op.create_table(
            "comment_embeddings",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "review_comment_id",
                sa.Integer(),
                sa.ForeignKey("review_comments.id"),
                nullable=False,
            ),
            sa.Column("model_name", sa.String(255), nullable=False),
            sa.Column("source_text_hash", sa.String(64), nullable=False),
            sa.Column("dimensions", sa.Integer(), nullable=False),
            sa.Column("vector", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("review_comment_id", "model_name"),
        )
    if not _has_table("comment_clusters"):
        op.create_table(
            "comment_clusters",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "repository_id",
                sa.Integer(),
                sa.ForeignKey("repositories.id"),
                nullable=False,
            ),
            sa.Column("model_name", sa.String(255), nullable=False),
            sa.Column("algorithm", sa.String(64), nullable=False),
            sa.Column("algorithm_config", sa.JSON(), nullable=False),
            sa.Column("label", sa.Integer(), nullable=False),
            sa.Column("size", sa.Integer(), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("repository_id", "model_name", "algorithm", "label"),
        )
    if not _has_table("comment_cluster_members"):
        op.create_table(
            "comment_cluster_members",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "cluster_id",
                sa.Integer(),
                sa.ForeignKey("comment_clusters.id"),
                nullable=False,
            ),
            sa.Column(
                "review_comment_id",
                sa.Integer(),
                sa.ForeignKey("review_comments.id"),
                nullable=False,
            ),
            sa.Column("similarity_to_centroid", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("cluster_id", "review_comment_id"),
        )
    if not _has_table("candidate_norms"):
        op.create_table(
            "candidate_norms",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "repository_id",
                sa.Integer(),
                sa.ForeignKey("repositories.id"),
                nullable=False,
            ),
            sa.Column(
                "cluster_id",
                sa.Integer(),
                sa.ForeignKey("comment_clusters.id"),
                nullable=False,
            ),
            sa.Column("statement", sa.Text(), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("support_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("cluster_id"),
        )
    if not _has_table("candidate_norm_evidence"):
        op.create_table(
            "candidate_norm_evidence",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "candidate_norm_id",
                sa.Integer(),
                sa.ForeignKey("candidate_norms.id"),
                nullable=False,
            ),
            sa.Column(
                "review_comment_id",
                sa.Integer(),
                sa.ForeignKey("review_comments.id"),
                nullable=False,
            ),
            sa.Column(
                "pull_request_id",
                sa.Integer(),
                sa.ForeignKey("pull_requests.id"),
                nullable=False,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("candidate_norm_id", "review_comment_id"),
        )


def downgrade() -> None:
    op.drop_table("candidate_norm_evidence")
    op.drop_table("candidate_norms")
    op.drop_table("comment_cluster_members")
    op.drop_table("comment_clusters")
    op.drop_table("comment_embeddings")
