from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./prnorms.db"
    redis_url: str = "redis://localhost:6379/0"
    local_demo_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("LOCAL_DEMO_MODE", "PRNORMS_LOCAL_DEMO_MODE"),
    )
    github_app_id: int | None = None
    github_private_key: str | None = None
    github_webhook_secret: str = ""
    github_client_id: str | None = None
    github_client_secret: str | None = None
    github_app_slug: str | None = None
    github_callback_url: str = "http://localhost:8000/api/v1/github/callback"
    github_state_secret: str | None = None
    github_api_url: str = "https://api.github.com"
    github_token: str | None = None
    backend_cors_origins: str = "http://localhost:5173"
    phase2_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "sentence-transformers" (default) or "demo-hashing", an explicitly
    # labelled lexical fallback for environments that cannot download models.
    phase2_embedding_backend: str = "sentence-transformers"
    phase2_embedding_batch_size: int = 32
    phase2_min_cluster_size: int = 3
    phase2_min_samples: int | None = None
    phase2_random_seed: int = 42
    public_import_max_pull_requests: int = 1000
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [x.strip() for x in self.backend_cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
