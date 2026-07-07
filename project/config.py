"""
DrugFX Configuration
====================
Centralized, validated configuration loaded from environment variables.
All application constants live here — nothing is hardcoded elsewhere.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables."""

    # ── Application ────────────────────────────────────────────
    APP_NAME: str = "DrugFX"
    APP_VERSION: str = "3.0.0"
    APP_ENV: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM Provider ──────────────────────────────────────────
    LLM_PROVIDER: str = Field(default="groq", description="groq | gemini")
    GROQ_API_KEY: str = Field(default="", description="Groq API key from console.groq.com")
    GROQ_MODEL_PRIMARY: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_FALLBACK: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TEMPERATURE: float = 0.3

    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    GEMINI_MODEL_PRIMARY: str = "gemini-2.5-flash"
    GEMINI_MODEL_FALLBACK: str = "gemini-2.0-flash"

    # ── OCR ────────────────────────────────────────────────────
    OCR_PROVIDER: str = Field(default="auto", description="auto | tesseract | gemini_vision")
    OCR_CONFIDENCE_THRESHOLD: float = 0.5

    # ── Upload ────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/webp,image/gif,image/bmp,image/tiff"
    ALLOWED_EXTENSIONS: str = ".jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.tif,.pdf"

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    # ── RAG ───────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Retry / Resilience ────────────────────────────────────
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BASE_DELAY: float = 2.0
    LLM_REQUEST_TIMEOUT: int = 30

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True

    # ── Computed Properties ───────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def allowed_image_types_set(self) -> set:
        return set(t.strip() for t in self.ALLOWED_IMAGE_TYPES.split(","))

    @property
    def allowed_extensions_set(self) -> set:
        return set(e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(","))

    @property
    def cors_origins_list(self) -> list:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


# Singleton instance — import this everywhere
settings = Settings()
