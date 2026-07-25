from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str | None = None

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str | None = None

    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL:str | None = None
    OPENAI_SEARCH_MODEL: str | None = None

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str | None = None

 

    ENV: str | None = "development"
    API_PREFIX: str | None = "/api/v1"

settings = Settings()


