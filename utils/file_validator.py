"""Upload validation: extension, size, emptiness and basic signature checks."""
from __future__ import annotations

import os

# pyrefly: ignore [missing-import]
from config import SUPPORTED_EXTENSIONS


class ValidationError(ValueError):
    pass


def safe_filename(name: str | None) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    return base or "unnamed"


def validate_file(filename: str, data: bytes | None, max_mb: int) -> None:
    if not filename or not isinstance(filename, str):
        raise ValidationError("Invalid or missing filename.")
    if data is None or len(data) == 0:
        raise ValidationError("The file is empty.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError(f"Unsupported file type '{ext or 'none'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    max_bytes = max(1, max_mb) * 1024 * 1024
    if len(data) > max_bytes:
        raise ValidationError(f"The file is larger than the {max_mb} MB limit.")
    if ext == ".pdf" and b"%PDF" not in data[:1024]:
        raise ValidationError("This does not look like a valid PDF file.")
    if ext in (".docx", ".xlsx", ".pptx") and not data.startswith(b"PK"):
        raise ValidationError(f"This does not look like a valid {ext} file (wrong extension or corrupted).")


def unique_name(name: str | None, existing) -> str:
    name = safe_filename(name)
    if existing is None:
        return name
    if name not in existing:
        return name
    stem, ext = os.path.splitext(name)
    n = 2
    while f"{stem} ({n}){ext}" in existing:
        n += 1
    return f"{stem} ({n}){ext}"