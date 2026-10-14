"""Export dispatch: which formats are valid for a report, and how to build each."""
from __future__ import annotations

import io
import re

import pandas as pd

# pyrefly: ignore [missing-import]
from models import ReportData
# pyrefly: ignore [missing-import]
from utils.report_generator import build_docx, build_markdown, build_pdf, build_text

FORMATS = {
    "pdf": ("PDF", "application/pdf"),
    "docx": ("Word (DOCX)", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "txt": ("Text (TXT)", "text/plain"),
    "md": ("Markdown (MD)", "text/markdown"),
    "csv": ("CSV (table)", "text/csv"),
    "xlsx": ("Excel (table)", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
}
_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def safe_filename(title: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", title or "").strip("_")[:60] or "report"


def available_formats(report: ReportData) -> list[str]:
    formats = ["pdf", "docx", "txt", "md"]
    if report is not None and report.table is not None and not report.table.empty:
        formats += ["csv", "xlsx"]
    return formats


def _clean_table(df: pd.DataFrame | None) -> pd.DataFrame:
    """Strip illegal characters and neutralize spreadsheet-formula injection from text cells."""
    if df is None:
        return pd.DataFrame()
    def fix(v):
        if isinstance(v, str):
            v = _ILLEGAL_XLSX.sub("", v)
            if v.startswith(("=", "+", "@")):
                v = "'" + v
        return v
    if df.empty:
        cleaned = df.copy()
        cleaned.columns = [_ILLEGAL_XLSX.sub("", str(c)) for c in cleaned.columns]
        return cleaned
    cleaned = df.astype(object).apply(lambda col: col.map(fix))
    cleaned.columns = [_ILLEGAL_XLSX.sub("", str(c)) for c in cleaned.columns]
    return cleaned


def table_to_csv(df: pd.DataFrame) -> bytes:
    return _clean_table(df).to_csv(index=False).encode("utf-8-sig")


def table_to_xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _clean_table(df).to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()


def export_report(fmt: str, report: ReportData, logo_svg: str | None) -> tuple[bytes, str, str]:
    """Return (bytes, mime type, file name)."""
    if report is None:
        raise ValueError("ReportData is required.")
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'.")
    if fmt in ("csv", "xlsx") and (report.table is None or report.table.empty):
        raise ValueError("This report has no table to export.")
    if fmt == "pdf":
        data = build_pdf(report, logo_svg)
    elif fmt == "docx":
        data = build_docx(report)
    elif fmt == "txt":
        data = build_text(report).encode("utf-8")
    elif fmt == "md":
        data = build_markdown(report).encode("utf-8")
    elif fmt == "csv":
        data = table_to_csv(report.table)
    else:
        data = table_to_xlsx(report.table)
    return data, FORMATS[fmt][1], f"{safe_filename(report.title)}.{fmt}"