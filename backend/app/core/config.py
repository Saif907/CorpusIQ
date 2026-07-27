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

    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "Enron_Email_Dataset"
    THREADED_EMAILS_FILE: str | None = "threaded_emails.json"

    INGESTION_BATCH_SIZE: int = 100         
    INGESTION_MAX_WORKERS: int = 4   

    ENV: str | None = "development"
    API_PREFIX: str | None = "/api/v1"

settings = Settings()


