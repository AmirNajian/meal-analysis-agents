"""Application config from environment (pydantic-settings) with cached getter."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-loaded settings. Uses .env when present."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for this API (e.g. used by Gradio UI)",
    )


@lru_cache(maxsize=1)
def get_config() -> Settings:
    """Return the application settings. Cached so env is read once."""
    return Settings()
