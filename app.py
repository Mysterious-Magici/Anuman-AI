"""DocuMind AI - Streamlit front end.  Run with:  streamlit run app.py"""
from __future__ import annotations

import hashlib

import streamlit as st

# pyrefly: ignore [missing-import]
from config import APP_NAME, SUPPORTED_EXTENSIONS, RuntimeContext, load_settings
# pyrefly: ignore [missing-import]
from models import AnalysisResult
# pyrefly: ignore [missing-import]
from services import agent_workflow as aw
# pyrefly: ignore [missing-import]
from services import gemini_service, groq_service
# pyrefly: ignore [missing-import]
from services.errors import ProviderError
# pyrefly: ignore [missing-import]
from utils import history_manager as hm
# pyrefly: ignore [missing-import]
from utils import table_tools
# pyrefly: ignore [missing-import]
from utils.document_loader import DocumentError, load_document, ocr_available
# pyrefly: ignore [missing-import]
from utils.export_utils import FORMATS, available_formats, export_report, table_to_csv, table_to_xlsx
# pyrefly: ignore [missing-import]
from utils.file_validator import ValidationError, safe_filename, unique_name, validate_file
# pyrefly: ignore [missing-import]
from utils.report_generator import build_markdown, build_report_data, default_summary

st.set_page_config(page_title=APP_NAME, page_icon="\U0001F4C4", layout="wide")


# ---------- logo (defined here, reused in the sidebar, page headers and PDF reports) ----------

def logo_svg(size: int = 64) -> str:
    """Original logo: a document with a folded corner and a small neural-network graph."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64">'
        '<rect x="10" y="4" width="38" height="52" rx="5" fill="#EEF2FF" stroke="#4F46E5" stroke-width="3"/>'
        '<path d="M36 4 L48 16 L36 16 Z" fill="#4F46E5"/>'
        '<g stroke="#4F46E5" stroke-width="2" stroke-linecap="round">'
        '<line x1="19" y1="30" x2="30" y2="23"/><line x1="19" y1="30" x2="30" y2="40"/>'
        '<line x1="30" y1="23" x2="41" y2="31"/><line x1="30" y1="40" x2="41" y2="31"/>'
        '<line x1="30" y1="40" x2="21" y2="48"/></g>'
        '<g fill="#4F46E5"><circle cx="19" cy="30" r="3.2"/><circle cx="30" cy="23" r="3.2"/>'
        '<circle cx="30" cy="40" r="3.2"/><circle cx="41" cy="31" r="3.2"/><circle cx="21" cy="48" r="3.2"/></g>'
        '</svg>'
    )


st.markdown(
    """<style>
    .block-container{padding-top:2rem;max-width:1200px}
    /* Base metric styling and text colors for contrast */
    [data-testid=stMetric]{border:1px solid #E5E7EB;border-radius:10px;padding:12px 16px; color: #1F2937 !important;}
    [data-testid=stMetric] label {color: #4B5563 !important; font-weight: 600;}
    
    /* Background images injected via CSS nth-of-type for the 4 metric columns */
    [data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] { 
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), url('https://images.unsplash.com/photo-1568667256549-094345857637?w=400&q=80') center/cover; 
    }
    [data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { 
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&q=80') center/cover; 
    }
    [data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { 
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), url('https://images.unsplash.com/photo-1577563908411-50cb989766a3?w=400&q=80') center/cover; 
    }
    [data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { 
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=400&q=80') center/cover; 
    }
    </style>""",
    unsafe_allow_html=True,
)

PAGES = ["Dashboard", "Document Workspace", "AI Assistant", "Document Analysis",
         "Report Studio", "History", "Settings & Help"]
TASK_CHOICES = {"auto": "Auto-detect from my request", **aw.TASK_LABELS}
TASK_HINTS = {
    "auto": "Describe what you want, e.g. 'Compare the two contracts' or 'Summarize chapter 2'.",
    "summarize": "Optional focus, e.g. 'financial risks'. Leave empty for a general summary.",
    "key_points": "Optional focus, e.g. 'deadlines and obligations'.",
    "extract": "Fields to extract, e.g. 'invoice number, date, vendor, total amount'.",
    "qa": "Your question about the selected documents.",
    "compare": "Optional focus, e.g. 'payment terms'. Requires two or more documents.",
    "data_profile": "No input needed. Works on CSV/Excel files and tables inside documents.",
}


# ---------- state ----------

def init_state() -> None:
    ss = st.session_state
    if "ctx" not in ss:
        ss["ctx"] = RuntimeContext(settings=load_settings())
    for key, default in (("docs", {}), ("doc_hashes", {}), ("chat", []), ("history", []), ("nav", "Dashboard"),
                         ("confirm_clear", False), ("upload_log", []), ("report_files", None)):
        ss.setdefault(key, default)
    # Selection is cleaned/merged BEFORE the sidebar multiselect is created.
    pending = ss.pop("pending_select", [])
    ss["sel"] = [n for n in dict.fromkeys(ss.get("sel", []) + pending) if n in ss["docs"]]


def goto(page: str) -> None:
    st.session_state["nav"] = page


def open_in_report(entry_id: int) -> None:
    st.session_state["report_entry"] = entry_id
    st.session_state["nav"] = "Report Studio"


def selected_docs():
    return [st.session_state.docs[n] for n in st.session_state.get("sel", [])]


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">{logo_svg(44)}'
        f'<div><div style="font-size:1.6rem;font-weight:700;color:#1F2937">{title}</div>'
        f'<div style="color:#6B7280">{subtitle}</div></div></div>', unsafe_allow_html=True)
    st.divider()


def render_result(result: AnalysisResult) -> None:
    if not result.ok:
        st.error(result.error or "The analysis did not complete.")
    for w in result.warnings:
        st.warning(w)
    if result.coverage is not None:
        st.progress(min(result.coverage, 1.0), text=f"Text sent to the model: {result.coverage:.0%} of the selected content")
    if result.content:
        st.markdown(result.content)
    if result.table is not None and not result.table.empty:
        st.dataframe(result.table, use_container_width=True)
    with st.expander("Workflow steps"):
        icons = {"ok": "\u2705", "failed": "\u274C", "skipped": "\u23ED\uFE0F"}
        st.markdown("\n".join(f"- {icons.get(s.status, '')} **{s.name}** {s.detail}" for s in result.steps))
    if result.sources:
        with st.expander(f"Source references ({len(result.sources)})"):
            st.markdown("\n".join(f"- {r}" for r in result.sources))


def doc_info(doc) -> dict:
    return {"Document": doc.name, "Type": doc.extension.upper().lstrip("."),
            "Size": f"{doc.size_bytes / 1024:.1f} KB",
            "Details": ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())}


# ---------- sidebar ----------

def sidebar() -> None:
    ss = st.session_state
    s = ss.ctx.settings
    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px">{logo_svg(38)}'
            f'<b style="font-size:1.2rem;color:#4F46E5">{APP_NAME}</b></div>', unsafe_allow_html=True)
        st.caption("Agentic document intelligence")
        st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.multiselect("Active documents", options=list(ss.docs), key="sel",
                       help="Analysis and chat use these documents.")
        st.caption(f"Gemini: {'ready' if s.gemini_api_key and s.gemini_model else 'not configured'}  \n"
                   f"Groq: {'ready' if s.groq_api_key and s.groq_model else 'not configured'}")
        st.caption(f"API calls this session - Gemini: {ss.ctx.usage['gemini']}, Groq: {ss.ctx.usage['groq']}")


# ---------- pages ----------

def page_dashboard() -> None:
    ss = st.session_state
    header("Dashboard", "Upload documents, ask questions, analyze and export professional reports")
    analyses = hm.entries(ss.history, "analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents loaded", len(ss.docs))
    c2.metric("Analyses run", len(analyses))
    c3.metric("Chat messages", len(ss.chat))
    c4.metric("AI calls (Gemini / Groq)", f"{ss.ctx.usage['gemini']} / {ss.ctx.usage['groq']}")
    st.caption("Free-tier quotas are set by the providers and can change. Counters show this session only.")
    st.subheader("Quick actions")
    a, b, c, d = st.columns(4)
    a.button("Upload documents", on_click=goto, args=("Document Workspace",), use_container_width=True)
    b.button("Ask the assistant", on_click=goto, args=("AI Assistant",), use_container_width=True)
    c.button("Analyze documents", on_click=goto, args=("Document Analysis",), use_container_width=True)
    d.button("Build a report", on_click=goto, args=("Report Studio",), use_container_width=True)
    st.subheader("Recent activity")
    recent = hm.entries(ss.history)[:5]
    if not recent:
        st.info("Nothing yet. Start by uploading a document in the Document Workspace.")
    for e in recent:
        st.markdown(f"- `{e.created_at:%H:%M}` **{e.kind}** - {e.title}")


def process_uploads(files) -> None:
    ss, s = st.session_state, st.session_state.ctx.settings
    log, new = [], []
    bar = st.progress(0.0, text="Starting...")
    for i, f in enumerate(files, start=1):
        bar.progress((i - 1) / len(files), text=f"Reading {f.name} ({i}/{len(files)})")
        data = f.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if digest in ss.doc_hashes:
            log.append((f.name, "skipped", f"Identical to '{ss.doc_hashes[digest]}' (already loaded)."))
            continue
        if len(ss.docs) >= s.max_files:
            log.append((f.name, "error", f"File limit reached ({s.max_files})."))
            continue
        try:
            validate_file(f.name, data, s.max_file_mb)
            doc = load_document(safe_filename(f.name), data)
        except (ValidationError, DocumentError) as exc:
            log.append((f.name, "error", str(exc)))
            continue
        doc.name = unique_name(safe_filename(f.name), ss.docs)
        ss.docs[doc.name] = doc
        ss.doc_hashes[digest] = doc.name
        new.append(doc.name)
        log.append((doc.name, "ok", f"{len(doc.segments)} segment(s), {doc.total_chars:,} characters"))
    bar.progress(1.0, text="Done")
    ss.upload_log, ss.pending_select = log, new


def remove_document(name: str) -> None:
    ss = st.session_state
    ss.docs.pop(name, None)
    for h in [h for h, n in ss.doc_hashes.items() if n == name]:
        ss.doc_hashes.pop(h)


def page_workspace() -> None:
    ss, s = st.session_state, st.session_state.ctx.settings
    header("Document Workspace", "Upload, preview and manage your documents")
    files = st.file_uploader("Drag and drop files or browse", accept_multiple_files=True,
                             type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                             help=f"Up to {s.max_file_mb} MB per file and {s.max_files} files per session.")
    if files and st.button("Process files", type="primary"):
        process_uploads(files)
        st.rerun()
    if not ocr_available():
        st.caption("Scanned PDFs need the optional OCR tools (see Settings & Help).")
    for name, status, msg in ss.upload_log:
        {"ok": st.success, "skipped": st.info, "error": st.error}[status](f"{name}: {msg}")
    if not ss.docs:
        st.info("No documents loaded yet.")
        return

    name = st.selectbox("Inspect document", list(ss.docs))
    doc = ss.docs[name]
    m1, m2, m3 = st.columns(3)
    m1.metric("Segments", len(doc.segments))
    m2.metric("Characters", f"{doc.total_chars:,}")
    m3.metric("Size", f"{doc.size_bytes / 1024:.1f} KB")
    for w in doc.warnings:
        st.warning(w)
    t1, t2, t3, t4 = st.tabs(["Extracted text", "Tables", "Metadata", "Manage"])
    with t1:
        st.text_area("Preview", doc.preview(), height=320, disabled=True, label_visibility="collapsed")
    with t2:
        if not doc.tables:
            st.info("This document has no tables.")
        else:
            tname = st.selectbox("Table", list(doc.tables))
            df = doc.tables[tname]
            st.caption(f"{len(df):,} rows x {df.shape[1]} columns (showing up to 100 rows)")
            st.dataframe(df.head(100), use_container_width=True)
            with st.expander("Create a cleaned copy (does not change the data used for analysis)"):
                op = st.selectbox("Operation", list(table_tools.OPERATIONS), format_func=table_tools.OPERATIONS.get)
                col = cmp_ = val = None
                if op == "filter":
                    col = st.selectbox("Column", list(df.columns))
                    cmp_ = st.selectbox("Condition", table_tools.COMPARISONS)
                    val = st.text_input("Value")
                if st.button("Preview cleaned copy"):
                    try:
                        ss["cleaned"] = (name, tname, table_tools.apply_operation(df, op, col, cmp_, val))
                    except ValueError as exc:
                        st.error(str(exc))
                cleaned = ss.get("cleaned")
                if cleaned and cleaned[0] == name and cleaned[1] == tname:
                    out = cleaned[2]
                    st.caption(f"{len(df):,} -> {len(out):,} rows")
                    st.dataframe(out.head(50), use_container_width=True)
                    if len(out):
                        d1, d2 = st.columns(2)
                        d1.download_button("Download CSV", table_to_csv(out), "cleaned.csv", "text/csv")
                        d2.download_button("Download Excel", table_to_xlsx(out), "cleaned.xlsx",
                                           FORMATS["xlsx"][1])
    with t3:
        st.json({"name": doc.name, "type": doc.extension, "size_bytes": doc.size_bytes, **doc.metadata}, expanded=True)
    with t4:
        st.write("Removing a document does not affect History.")
        if st.button("Remove this document"):
            remove_document(name)
            st.rerun()
        if st.checkbox("I want to remove ALL documents") and st.button("Remove all documents"):
            for n in list(ss.docs):
                remove_document(n)
            st.rerun()


def page_assistant() -> None:
    ss = st.session_state
    header("AI Assistant", "Chat powered by Groq, with relevant excerpts from your active documents")
    s = ss.ctx.settings
    if not (s.groq_api_key and s.groq_model):
        st.warning("Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL in your .env file (see Settings & Help).")
    docs = selected_docs()
    use_docs = st.toggle("Use active documents as context", value=bool(docs), disabled=not docs)
    if st.button("Reset conversation"):
        ss.chat = []
        st.rerun()
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("refs"):
                st.caption("Document sources: " + "; ".join(m["refs"][:6]))
    question = st.chat_input("Ask about your documents or anything else...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    history = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
                    answer, refs = aw.answer_chat(ss.ctx, question, history, docs, use_docs)
                st.markdown(answer)
                if refs:
                    st.caption("Document sources: " + "; ".join(refs[:6]))
                ss.chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "refs": refs}]
                hm.add_entry(ss.history, "chat", f"Chat: {question[:70]}", question=question, answer=answer, refs=refs)
            except ProviderError as exc:
                st.error(str(exc))


def page_analysis() -> None:
    ss = st.session_state
    header("Document Analysis", "Gemini-powered summaries, extraction, comparison and data profiling")
    s = ss.ctx.settings
    if not (s.gemini_api_key and s.gemini_model):
        st.warning("Gemini is not configured. Set GEMINI_API_KEY and GEMINI_MODEL in your .env file.")
    docs = selected_docs()
    if docs:
        st.caption("Active documents: " + ", ".join(d.name for d in docs))
    else:
        st.info("Select documents in the sidebar (upload them in the Document Workspace first).")
    task = st.selectbox("Analysis type", list(TASK_CHOICES), format_func=TASK_CHOICES.get)
    request = st.text_area("Your request", placeholder=TASK_HINTS[task], height=90)
    st.caption(f"Each run makes at most {s.max_agent_steps} Gemini call(s). Local steps (parsing, retrieval, statistics) cost nothing.")
    if st.button("Run analysis", type="primary", disabled=not docs):
        with st.spinner("Running workflow..."):
            result = aw.run_task(ss.ctx, task, request, docs)
        ss["last_result"] = result
        if result.ok:
            entry = hm.add_entry(ss.history, "analysis", result.title, result=result)
            ss["last_entry_id"] = entry.id
        else:
            ss["last_entry_id"] = None
    result = ss.get("last_result")
    if result is not None:
        st.divider()
        st.subheader(result.title)
        render_result(result)
        if result.ok and ss.get("last_entry_id"):
            st.button("Send to Report Studio", on_click=open_in_report, args=(ss["last_entry_id"],))


def page_reports() -> None:
    ss = st.session_state
    header("Report Studio", "Preview a professional report, then download it")
    analyses = hm.entries(ss.history, "analysis")
    if not analyses:
        st.info("Run an analysis first. Successful results appear here.")
        return
    ids = [e.id for e in analyses]
    wanted = ss.get("report_entry")
    index = ids.index(wanted) if wanted in ids else 0
    entry_id = st.selectbox("Source analysis", ids, index=index,
                            format_func=lambda i: f"{hm.get_entry(ss.history, i).created_at:%H:%M}  {hm.get_entry(ss.history, i).title}")
    ss["report_entry"] = entry_id
    entry = hm.get_entry(ss.history, entry_id)
    result = entry.result
    title = st.text_input("Report title", value=entry.title, key=f"rt_{entry_id}")
    summary = st.text_area("Executive summary (edit freely)", value=default_summary(result.content),
                           key=f"rs_{entry_id}", height=120)
    has_table = result.table is not None and not result.table.empty
    o1, o2, o3 = st.columns(3)
    inc_table = o1.checkbox("Include data table", value=has_table, disabled=not has_table, key=f"it_{entry_id}")
    inc_src = o2.checkbox("Include source references", value=True, key=f"is_{entry_id}")
    inc_meth = o3.checkbox("Include methodology and limitations", value=True, key=f"im_{entry_id}")
    if st.button("Build report", type="primary"):
        infos = [doc_info(ss.docs[n]) if n in ss.docs else {"Document": n} for n in result.documents]
        report = build_report_data(result, infos, title, summary, inc_table, inc_src, inc_meth)
        files, failures = {}, []
        with st.spinner("Generating files..."):
            for fmt in available_formats(report):
                try:
                    files[fmt] = export_report(fmt, report, logo_svg(64))
                except Exception as exc:
                    failures.append((fmt, str(exc)[:200] or type(exc).__name__))
        ss["report_files"] = {"entry": entry_id, "report": report, "files": files, "failures": failures}
        hm.add_entry(ss.history, "report", f"Report: {report.title}", report=report)
    built = ss.get("report_files")
    if built and built["entry"] == entry_id:
        with st.container(border=True):
            st.markdown(build_markdown(built["report"], max_rows=30))
        for fmt, msg in built["failures"]:
            st.warning(f"{FORMATS[fmt][0]} could not be generated: {msg}")
        cols = st.columns(max(len(built["files"]), 1))
        for col, (fmt, (data, mime, fname)) in zip(cols, built["files"].items()):
            col.download_button(FORMATS[fmt][0], data, fname, mime, key=f"dl_{fmt}", use_container_width=True)


def page_history() -> None:
    ss = st.session_state
    header("History", "Questions, analyses and reports from this session")
    st.info("History is kept only in this browser session (Streamlit session state). It is NOT saved to disk and "
            "disappears when you refresh or close the tab. Clearing it does not remove your uploaded documents.")
    kind = st.selectbox("Show", ["all", "chat", "analysis", "report"])
    items = hm.entries(ss.history, None if kind == "all" else kind)
    if not items:
        st.write("No entries.")
    for e in items:
        with st.expander(f"{e.created_at:%H:%M:%S}  |  {e.kind}  |  {e.title}"):
            if e.kind == "chat":
                st.markdown(f"**You:** {e.question}")
                st.markdown(e.answer)
            elif e.kind == "analysis" and e.result:
                render_result(e.result)
                st.button("Open in Report Studio", on_click=open_in_report, args=(e.id,), key=f"or_{e.id}")
            elif e.kind == "report" and e.report:
                st.markdown(build_markdown(e.report, max_rows=30))
    st.divider()
    if not ss.confirm_clear:
        if st.button("Clear history", disabled=not ss.history):
            ss.confirm_clear = True
            st.rerun()
    else:
        st.warning(f"Delete all {len(ss.history)} history entries? Documents and the current chat are not affected.")
        y, n = st.columns(2)
        if y.button("Yes, clear history", type="primary"):
            hm.clear_history(ss.history)
            ss.confirm_clear = False
            ss.pop("last_result", None)
            ss.report_files = None
            st.rerun()
        if n.button("Cancel"):
            ss.confirm_clear = False
            st.rerun()


def page_settings() -> None:
    ss = st.session_state
    s = ss.ctx.settings
    header("Settings & Help", "Configuration status, privacy and troubleshooting")
    st.subheader("Model configuration")
    st.markdown(
        f"- **Gemini** key: {'set' if s.gemini_api_key else 'MISSING'} | model: `{s.gemini_model or 'MISSING (GEMINI_MODEL)'}`\n"
        f"- **Groq** key: {'set' if s.groq_api_key else 'MISSING'} | model: `{s.groq_model or 'MISSING (GROQ_MODEL)'}`\n"
        f"- Limits: {s.max_file_mb} MB per file, {s.max_files} files, {s.max_agent_steps} Gemini calls per analysis, "
        f"{s.max_context_chars:,} characters per model call")
    st.caption("Key values are never displayed. Change settings in the .env file and restart the app.")
    t1, t2 = st.columns(2)
    if t1.button("Test Gemini connection"):
        try:
            with st.spinner("Contacting Gemini..."):
                gemini_service.generate(ss.ctx, "Reply with the single word: OK", use_cache=False)
            st.success("Gemini responded successfully.")
        except ProviderError as exc:
            st.error(str(exc))
    if t2.button("Test Groq connection"):
        try:
            with st.spinner("Contacting Groq..."):
                groq_service.chat(ss.ctx, [{"role": "user", "content": "Reply with the single word: OK"}])
            st.success("Groq responded successfully.")
        except ProviderError as exc:
            st.error(str(exc))
    st.caption("Connection tests use a tiny amount of your free-tier quota.")
    st.subheader("Supported formats")
    st.write(", ".join(SUPPORTED_EXTENSIONS) + f". OCR for scanned PDFs: {'available' if ocr_available() else 'not installed (optional)'}.")
    st.subheader("Privacy")
    st.markdown(
        "- Files are parsed locally. Only short, relevant excerpts are sent to Gemini or Groq, and they are processed "
        "under those providers' own terms and policies.\n"
        "- Documents, chat and history live only in this session's memory. Nothing is written to disk by this app.\n"
        "- API keys are read from environment variables and are never shown, logged or put in reports.")
    st.subheader("Troubleshooting")
    st.markdown(
        "- **Missing key/model:** fill in `.env`, then restart `streamlit run app.py`.\n"
        "- **Model not found:** set `GEMINI_MODEL` / `GROQ_MODEL` to an ID your account supports (check the provider's model list).\n"
        "- **Rate limit or quota:** wait a minute, run fewer analyses, or lower `MAX_AGENT_STEPS`.\n"
        "- **No text in a PDF:** it is probably scanned; install the optional OCR tools from the README.\n"
        "- **Partial coverage warning:** the document is larger than one run can read; see the warning for options.\n"
        "- **Non-Western characters in PDF reports:** PDFs use built-in fonts, so such characters appear as '?'. Use DOCX or Markdown instead.")


# ---------- main ----------

init_state()
sidebar()
{
    "Dashboard": page_dashboard,
    "Document Workspace": page_workspace,
    "AI Assistant": page_assistant,
    "Document Analysis": page_analysis,
    "Report Studio": page_reports,
    "History": page_history,
    "Settings & Help": page_settings,
}[st.session_state["nav"]]()