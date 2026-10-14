"""Session history helpers. History lives only in Streamlit session state (lost on refresh/close)."""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from models import HistoryEntry


def add_entry(history: list[HistoryEntry], kind: str, title: str | None, **fields) -> HistoryEntry:
    clean_title = (title or "Untitled")[:120]
    entry_id = max((e.id for e in history), default=0) + 1 if history else 1
    entry = HistoryEntry(id=entry_id, kind=kind, title=clean_title, **fields)
    history.append(entry)
    return entry


def get_entry(history: list[HistoryEntry] | None, entry_id: int) -> HistoryEntry | None:
    if not history:
        return None
    return next((e for e in history if e.id == entry_id), None)


def entries(history: list[HistoryEntry] | None, kind: str | None = None) -> list[HistoryEntry]:
    if not history:
        return []
    return [e for e in reversed(history) if kind is None or e.kind == kind]


def clear_history(history: list[HistoryEntry] | None) -> int:
    """Clear history only. Uploaded documents and chat state are separate and untouched."""
    if history is None:
        return 0
    count = len(history)
    history.clear()
    return count