import pytest

# pyrefly: ignore [missing-import]
from config import RuntimeContext, Settings
# pyrefly: ignore [missing-import]
from services import agent_workflow as aw
# pyrefly: ignore [missing-import]
from services.errors import ProviderError
# pyrefly: ignore [missing-import]
from utils.document_loader import load_document


def _ctx(**kw):
    base = dict(gemini_api_key="k", gemini_model="m", groq_api_key="g", groq_model="m",
                chunk_size=200, chunk_overlap=50)
    base.update(kw)
    return RuntimeContext(settings=Settings(**base))


def _txt(name="a.txt", text="The invoice total is 500 dollars."):
    return load_document(name, text.encode())


def test_classify_task():
    assert aw.classify_task("Compare these two files") == "compare"
    assert aw.classify_task("summarize this") == "summarize"
    assert aw.classify_task("extract invoice number") == "extract"
    assert aw.classify_task("what is the total?") == "qa"


def test_qa_success(monkeypatch):
    monkeypatch.setattr(aw.gemini_service, "generate", lambda *a, **k: "It is 500 dollars [a.txt | full text].")
    result = aw.run_task(_ctx(), "qa", "What is the invoice total?", [_txt()])
    assert result.ok and "500" in result.content and not result.warnings


def test_provider_failure_is_reported_not_hidden(monkeypatch):
    def boom(*a, **k):
        raise ProviderError("quota hit", "rate_limit")
    monkeypatch.setattr(aw.gemini_service, "generate", boom)
    result = aw.run_task(_ctx(), "qa", "invoice total", [_txt()])
    assert not result.ok and result.error_kind == "rate_limit" and result.content == ""


def test_extract_valid_json_becomes_table(monkeypatch):
    monkeypatch.setattr(aw.gemini_service, "generate", lambda *a, **k:
                        '```json\n{"records":[{"total":500,"source":"a.txt | full text"}],"notes":"ok"}\n```')
    result = aw.run_task(_ctx(), "extract", "total", [_txt()])
    assert result.ok and list(result.table.columns) == ["total", "source"]


def test_extract_malformed_json_fails_honestly(monkeypatch):
    monkeypatch.setattr(aw.gemini_service, "generate", lambda *a, **k: "not json at all")
    result = aw.run_task(_ctx(), "extract", "total", [_txt()])
    assert not result.ok and "JSON" in result.error


def test_compare_needs_two_documents():
    result = aw.run_task(_ctx(), "compare", "", [_txt()])
    assert not result.ok and "two" in result.error


def test_long_document_respects_call_budget_and_reports_coverage(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return "Summary [a.txt | full text]"
    monkeypatch.setattr(aw.gemini_service, "generate", fake)
    ctx = _ctx(max_agent_steps=3, max_context_chars=500)
    result = aw.run_task(ctx, "summarize", "", [_txt(text="word " * 1000)])
    assert result.ok and calls["n"] <= 3
    assert result.coverage < 1 and any("Partial coverage" in w for w in result.warnings)


def test_tool_permissions_are_enforced():
    box = aw.ToolBox("summarize", 3)
    with pytest.raises(PermissionError):
        box.run("parse", lambda: None)


def test_parse_json_records_accepts_plain_list():
    records, notes = aw.parse_json_records('[{"a": 1}]')
    assert records == [{"a": 1}] and notes == ""


def test_data_profile_survives_ai_failure(monkeypatch):
    def boom(*a, **k):
        raise ProviderError("down", "unavailable")
    monkeypatch.setattr(aw.gemini_service, "generate", boom)
    doc = load_document("d.csv", b"a,b\n1,\n1,\n")
    result = aw.run_task(_ctx(), "data_profile", "", [doc])
    assert result.ok and "Duplicate rows" in result.content and result.warnings


def test_unknown_task_fails_gracefully():
    result = aw.run_task(_ctx(), "non_existent_task", "", [_txt()])
    assert not result.ok
    assert "Unknown task" in result.error


def test_parse_json_records_with_code_block_and_text():
    text = "Here is the data:\n```json\n[{\"col\": \"val\"}]\n```\nHope that helps!"
    records, notes = aw.parse_json_records(text)
    assert records == [{"col": "val"}]


def test_data_profile_handles_duplicate_column_names(monkeypatch):
    monkeypatch.setattr(aw.gemini_service, "generate", lambda *a, **k: "Diagnostic info")
    doc = load_document("dup.csv", b"a,a\n1,2\n3,4\n")
    result = aw.run_task(_ctx(), "data_profile", "", [doc])
    assert result.ok and result.table is not None


def test_agent_py_missing_key():
    from agent import analyze_data_profile
    msg = analyze_data_profile({}, [], api_key="")
    assert "error" in msg.lower() and "api_key" in msg.lower()