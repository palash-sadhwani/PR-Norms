from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Timestamped(Base):
    __abstract__ = True
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Installation(Timestamped):
    __tablename__ = "installations"
    github_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    account_login: Mapped[str] = mapped_column(String(255))
    account_type: Mapped[str] = mapped_column(String(32), default="Organization")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)


class Repository(Timestamped):
    __tablename__ = "repositories"
    installation_id: Mapped[int] = mapped_column(
        ForeignKey("installations.id"), index=True
    )
    github_id: Mapped[int] = mapped_column(BigInteger, index=True)
    full_name: Mapped[str] = mapped_column(String(255), unique=True)
    default_branch: Mapped[str] = mapped_column(String(255), default="main")
    raw: Mapped[dict] = mapped_column(JSON, default=dict)

    @property
    def source(self) -> str:
        if self.raw.get("synthetic_demo") == "pr-norms-synthetic-demo-v1":
            return "pr-norms-synthetic-demo-v1"
        return str(self.raw.get("source", "github_app"))


class User(Timestamped):
    __tablename__ = "users"
    github_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    login: Mapped[str] = mapped_column(String(255))
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)


class PullRequest(Timestamped):
    __tablename__ = "pull_requests"
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), index=True
    )
    github_id: Mapped[int] = mapped_column(BigInteger)
    number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(32))
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    merged: Mapped[bool] = mapped_column(Boolean, default=False)
    base_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    head_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    base_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    head_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    merged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (
        UniqueConstraint("repository_id", "number"),
        UniqueConstraint("repository_id", "github_id"),
    )


class Review(Timestamped):
    __tablename__ = "reviews"
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    github_id: Mapped[int] = mapped_column(BigInteger)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    state: Mapped[str] = mapped_column(String(32))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (UniqueConstraint("pull_request_id", "github_id"),)


class ReviewComment(Timestamped):
    __tablename__ = "review_comments"
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    review_id: Mapped[int | None] = mapped_column(
        ForeignKey("reviews.id"), nullable=True, index=True
    )
    thread_id: Mapped[int | None] = mapped_column(
        ForeignKey("review_threads.id"), nullable=True, index=True
    )
    github_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    body: Mapped[str] = mapped_column(Text)
    path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diff_hunk: Mapped[str | None] = mapped_column(Text, nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    in_reply_to_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    normalized_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_filtered: Mapped[bool] = mapped_column(Boolean, default=False)
    filter_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_outdated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_resolved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)


class IssueComment(ReviewComment):
    __tablename__ = "issue_comments"
    id: Mapped[int] = mapped_column(ForeignKey("review_comments.id"), primary_key=True)


class PullRequestFile(Timestamped):
    __tablename__ = "pull_request_files"
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    filename: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
    patch: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (UniqueConstraint("pull_request_id", "filename"),)


class Commit(Timestamped):
    __tablename__ = "commits"
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    sha: Mapped[str] = mapped_column(String(64), unique=True)
    message: Mapped[str] = mapped_column(Text)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)


class ReviewThread(Timestamped):
    __tablename__ = "review_threads"
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    github_node_id: Mapped[str] = mapped_column(String(255), unique=True)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    is_outdated: Mapped[bool] = mapped_column(Boolean, default=False)
    path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)


class CommentEmbedding(Timestamped):
    __tablename__ = "comment_embeddings"
    review_comment_id: Mapped[int] = mapped_column(
        ForeignKey("review_comments.id"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(255))
    source_text_hash: Mapped[str] = mapped_column(String(64))
    dimensions: Mapped[int] = mapped_column(Integer)
    vector: Mapped[list] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("review_comment_id", "model_name"),)


class CommentCluster(Timestamped):
    __tablename__ = "comment_clusters"
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(255))
    algorithm: Mapped[str] = mapped_column(String(64), default="hdbscan")
    algorithm_config: Mapped[dict] = mapped_column(JSON, default=dict)
    label: Mapped[int] = mapped_column(Integer)
    size: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column()
    __table_args__ = (
        UniqueConstraint("repository_id", "model_name", "algorithm", "label"),
    )


class CommentClusterMember(Timestamped):
    __tablename__ = "comment_cluster_members"
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("comment_clusters.id"), index=True
    )
    review_comment_id: Mapped[int] = mapped_column(
        ForeignKey("review_comments.id"), index=True
    )
    similarity_to_centroid: Mapped[float] = mapped_column()
    __table_args__ = (UniqueConstraint("cluster_id", "review_comment_id"),)


class CandidateNorm(Timestamped):
    __tablename__ = "candidate_norms"
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), index=True
    )
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("comment_clusters.id"), index=True
    )
    statement: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column()
    classification: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="emerging")
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    decision: Mapped[str] = mapped_column(String(32), default="provisional")
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    support_count: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("cluster_id"),)


class CandidateNormEvidence(Timestamped):
    __tablename__ = "candidate_norm_evidence"
    candidate_norm_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_norms.id"), index=True
    )
    review_comment_id: Mapped[int] = mapped_column(
        ForeignKey("review_comments.id"), index=True
    )
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    __table_args__ = (UniqueConstraint("candidate_norm_id", "review_comment_id"),)


class NormVersion(Timestamped):
    __tablename__ = "norm_versions"
    candidate_norm_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_norms.id"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    statement: Mapped[str] = mapped_column(Text)
    classification: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float] = mapped_column()
    status: Mapped[str] = mapped_column(String(32))
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    decision: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("candidate_norm_id", "version"),)


class NormConflict(Timestamped):
    __tablename__ = "norm_conflicts"
    candidate_norm_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_norms.id"), index=True
    )
    conflicting_norm_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_norms.id"), index=True
    )
    reason: Mapped[str] = mapped_column(Text)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("candidate_norm_id", "conflicting_norm_id"),)


class PullRequestAnalysis(Timestamped):
    __tablename__ = "pull_request_analyses"
    pull_request_id: Mapped[int] = mapped_column(
        ForeignKey("pull_requests.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="completed")
    formula_version: Mapped[str] = mapped_column(String(32), default="v1")
    findings_count: Mapped[int] = mapped_column(Integer, default=0)
    highest_risk: Mapped[str] = mapped_column(String(16), default="none")
    context_available: Mapped[bool] = mapped_column(Boolean, default=False)
    __table_args__ = (UniqueConstraint("pull_request_id"),)


class NormFinding(Timestamped):
    __tablename__ = "norm_findings"
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("pull_request_analyses.id"), index=True
    )
    candidate_norm_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_norms.id"), index=True
    )
    pull_request_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("pull_request_files.id"), nullable=True, index=True
    )
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance_score: Mapped[float] = mapped_column()
    evidence_strength: Mapped[float] = mapped_column()
    risk_score: Mapped[float] = mapped_column()
    risk_level: Mapped[str] = mapped_column(String(16))
    provisional: Mapped[bool] = mapped_column(Boolean, default=True)
    violation_suspected: Mapped[bool] = mapped_column(Boolean, default=False)
    explanation: Mapped[str] = mapped_column(Text)
    evidence_comment_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence_pull_request_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="open")
    __table_args__ = (
        UniqueConstraint("analysis_id", "candidate_norm_id", "file_path", "line_start"),
    )


class NormFindingFeedback(Timestamped):
    __tablename__ = "norm_finding_feedback"
    finding_id: Mapped[int] = mapped_column(ForeignKey("norm_findings.id"), index=True)
    decision: Mapped[str] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)


class IngestionJob(Timestamped):
    __tablename__ = "ingestion_jobs"
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    target_pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)


class IngestionCheckpoint(Timestamped):
    __tablename__ = "ingestion_checkpoints"
    job_id: Mapped[int] = mapped_column(ForeignKey("ingestion_jobs.id"), index=True)
    resource: Mapped[str] = mapped_column(String(64))
    cursor: Mapped[str | None] = mapped_column(String(255), nullable=True)
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("job_id", "resource"),)


class WebhookDelivery(Timestamped):
    __tablename__ = "webhook_deliveries"
    delivery_id: Mapped[str] = mapped_column(String(255), unique=True)
    event: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict] = mapped_column(JSON)
