"""Plain data containers shared across the app (no Streamlit or provider imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class Segment:
    text: str
    location: str  


@dataclass
class LoadedDocument:
    name: str
    extension: str
    size_bytes: int
    segments: list[Segment] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(s.text or "") for s in self.segments)

    def preview(self, limit: int = 3000) -> str:
        text = "\n\n".join(f"--- {s.location} ---\n{s.text or ''}" for s in self.segments)
        if len(text) > limit:
            return text[:limit] + "\n\n[Preview shortened. The full text is still used for analysis.]"
        return text


@dataclass
class StepLog:
    name: str
    status: str  
    detail: str = ""


@dataclass
class AnalysisResult:
    task: str
    title: str
    request: str
    documents: list[str]
    ok: bool = False
    content: str = ""                     
    table: pd.DataFrame | None = None
    steps: list[StepLog] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    coverage: float | None = None          
    error: str = ""
    error_kind: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReportData:
    title: str
    summary: str
    sections: list[tuple[str, str]]
    generated_at: datetime = field(default_factory=datetime.now)
    documents: list[dict] = field(default_factory=list)
    table: pd.DataFrame | None = None
    sources: list[str] = field(default_factory=list)
    methodology: str = ""


@dataclass
class HistoryEntry:
    id: int
    kind: str  # "chat" | "analysis" | "report"
    title: str
    created_at: datetime = field(default_factory=datetime.now)
    result: AnalysisResult | None = None
    question: str = ""
    answer: str = ""
    refs: list[str] = field(default_factory=list)
    report: ReportData | None = None