# =============================================================================
# SkyLex — Centralized Configuration & Settings
# All project-wide settings are managed here using Pydantic BaseSettings.
# Environment variables are automatically loaded from the .env file.
# =============================================================================

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory — used for building absolute paths
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    SkyLex application settings.
    All values are loaded from environment variables or .env file.
    Pydantic will raise a validation error at startup if required fields are missing.
    """

    # -------------------------------------------------------------------------
    # OpenAI Configuration
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Primary LLM model")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # -------------------------------------------------------------------------
    # LangSmith Configuration
    # -------------------------------------------------------------------------
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_tracing_v2: bool = Field(default=True, description="Enable LangSmith tracing")
    langchain_project: str = Field(default="skylex", description="LangSmith project name")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", description="LangSmith endpoint"
    )

    # -------------------------------------------------------------------------
    # MLflow Configuration
    # -------------------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="skylex", description="MLflow experiment name"
    )

    # -------------------------------------------------------------------------
    # Application Configuration
    # -------------------------------------------------------------------------
    app_env: str = Field(default="development", description="Application environment")
    app_port: int = Field(default=8000, description="FastAPI server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # -------------------------------------------------------------------------
    # Data Paths — Absolute paths built from PROJECT_ROOT
    # -------------------------------------------------------------------------
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }


# =============================================================================
# Single instance — import this everywhere in the project
# Usage: from config.settings import settings
# =============================================================================
settings = Settings()