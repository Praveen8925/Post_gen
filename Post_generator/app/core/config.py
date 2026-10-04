"""
config.py — Single source of truth for all settings.

HOW IT WORKS:
  1. Reads values from the .env file in the project root
  2. Exposes them as a Python object (settings.OLLAMA_MODEL, etc.)
  3. Every other file imports `settings` from here — never hardcodes values

WHY pydantic_settings:
  - Type validation (e.g. OLLAMA_TEMPERATURE must be a float)
  - Default values if .env key is missing
  - Auto-loads .env without you calling dotenv.load_dotenv()
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4:31b-cloud"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_TEMPERATURE: float = 0.7

    # ── App ───────────────────────────────────────────────────────────────
    APP_TITLE: str = "Content Generator AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── File Paths ────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"
    CHROMA_DIR: str = "chroma_db"

    # ── Phase 2 APIs ──────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    HEYGEN_API_KEY: str = ""

    # ── Media Generation (NVIDIA NIM → swap to DALL-E in Phase 2) ────────
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = "https://gopal-moxqsanx-eastus2.cognitiveservices.azure.com"
    AZURE_OPENAI_IMAGE_DEPLOYMENT: str = "gpt-image-1-mini"
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    MEDIA_OUTPUT_DIR: str = "media_output"

    # Tells Pydantic: read from .env file in project root
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# lru_cache = "run this function ONCE, cache the result forever"
# So Settings() is only created one time, no matter how many files import it
@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ── The one object every file imports ────────────────────────────────────────
# Usage anywhere: from app.core.config import settings
settings = get_settings()
