from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RepositoryOut(APIModel):
    id: int
    full_name: str
    default_branch: str
    source: str = "github_app"


class PullRequestOut(APIModel):
    id: int
    number: int
    title: str
    body: str | None
    state: str
    merged: bool


class IngestionOut(APIModel):
    id: int
    status: str
    error: str | None
    created_at: datetime
    total_items: int
    completed_items: int
    current_stage: str | None
    error_count: int


class NormUpdateIn(BaseModel):
    statement: str | None = None
    classification: str | None = None
    confidence: float | None = None
    status: str | None = None
    reason: str | None = None
    changed_by: str | None = None


class NormDecisionIn(BaseModel):
    reason: str | None = None
    changed_by: str | None = None


class NormConflictIn(BaseModel):
    conflicting_norm_id: int
    reason: str
    changed_by: str | None = None


class FindingFeedbackIn(BaseModel):
    decision: str
    note: str | None = None
    changed_by: str | None = None


class PublicImportIn(BaseModel):
    repository: str
    max_pull_requests: int = Field(default=50, ge=1, le=1000)
