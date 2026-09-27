from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://docupilot:docupilot@localhost:56432/docupilot"
    mock_llm: bool = True
    similarity_threshold: float = Field(default=0.16, ge=0, le=1)
    top_k: int = Field(default=3, ge=1, le=10)
    embedding_dim: int = Field(default=384, ge=32, le=3072)
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr = SecretStr("")
    max_upload_bytes: int = 1_000_000

    @property
    def embedding_signature(self) -> str:
        model = "mock-hash-v1" if self.mock_llm else self.embedding_model
        return f"{model}:{self.embedding_dim}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
