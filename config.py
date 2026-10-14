"""Central configuration. Reads environment variables; never stores or prints key values."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Anuman AI"
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".pptx")


def _int_env(name: str, default: int) -> int:
    try:
        value = int((os.getenv(name) or "").strip() or default)
        return value if value > 0 else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    max_file_mb: int = 20
    max_files: int = 10
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 6
    max_context_chars: int = 12000   
    max_agent_steps: int = 6         
    request_timeout_s: int = 60
    max_retries: int = 3


def load_settings() -> Settings:
    size = _int_env("CHUNK_SIZE", 1500)
    overlap = max(0, min(_int_env("CHUNK_OVERLAP", 200), size // 2))
    return Settings(
        gemini_api_key=(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip(),
        gemini_model=(os.getenv("GEMINI_MODEL") or "").strip(),
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
        groq_model=(os.getenv("GROQ_MODEL") or "").strip(),
        max_file_mb=_int_env("MAX_FILE_MB", 20),
        max_files=_int_env("MAX_FILES", 10),
        chunk_size=size,
        chunk_overlap=overlap,
        top_k=_int_env("TOP_K", 6),
        max_context_chars=_int_env("MAX_CONTEXT_CHARS", 12000),
        max_agent_steps=_int_env("MAX_AGENT_STEPS", 6),
        request_timeout_s=_int_env("REQUEST_TIMEOUT_S", 60),
        max_retries=_int_env("MAX_RETRIES", 3),
    )


@dataclass
class RuntimeContext:
    """Per-session objects: settings, response cache and API-call counters (never shared between users)."""
    settings: Settings
    cache: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: {"gemini": 0, "groq": 0})