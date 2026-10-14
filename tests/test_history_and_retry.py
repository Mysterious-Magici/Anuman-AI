import pytest

# pyrefly: ignore [missing-import]
from models import HistoryEntry
# pyrefly: ignore [missing-import]
from services.errors import ProviderError, call_with_retries
# pyrefly: ignore [missing-import]
from utils import history_manager as hm
# pyrefly: ignore [missing-import]
from utils import table_tools


def test_retry_succeeds_after_transient_failures():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ProviderError("busy", "rate_limit", retryable=True)
        return "ok"
    assert call_with_retries(flaky, 3, sleep=lambda s: None) == "ok"


def test_retry_is_bounded():
    state = {"n": 0}

    def always():
        state["n"] += 1
        raise ProviderError("busy", "rate_limit", retryable=True)
    with pytest.raises(ProviderError):
        call_with_retries(always, 3, sleep=lambda s: None)
    assert state["n"] == 3


def test_non_retryable_errors_fail_immediately():
    state = {"n": 0}

    def bad():
        state["n"] += 1
        raise ProviderError("bad key", "auth")
    with pytest.raises(ProviderError):
        call_with_retries(bad, 3, sleep=lambda s: None)
    assert state["n"] == 1


def test_history_add_get_and_clear_are_independent_of_other_state():
    history: list[HistoryEntry] = []
    docs = {"a.txt": object()}
    e1 = hm.add_entry(history, "chat", "q1", question="q", answer="a")
    hm.add_entry(history, "analysis", "t2")
    assert hm.get_entry(history, e1.id).title == "q1"
    assert [e.id for e in hm.entries(history)] == [2, 1]
    assert hm.clear_history(history) == 2 and history == []
    assert "a.txt" in docs


def test_table_tools_do_not_mutate_original():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    out = table_tools.apply_operation(df, "filter", "a", ">", "1")
    assert len(df) == 3 and list(out["b"]) == ["y"]
    with pytest.raises(ValueError):
        table_tools.apply_operation(df, "filter", "a", ">", "abc")
    assert len(table_tools.apply_operation(df, "drop_missing")) == 2


def test_table_tools_edge_cases():
    import pandas as pd
    with pytest.raises(ValueError):
        table_tools.apply_operation(None, "drop_missing")

    # Unhashable type (list) in duplicate checking
    df_unhashable = pd.DataFrame({"a": [[1, 2], [1, 2], [3]]})
    dedup = table_tools.apply_operation(df_unhashable, "drop_duplicates")
    assert len(dedup) == 2

    # Duplicate column names
    df_dup_cols = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"])
    filtered = table_tools.apply_operation(df_dup_cols, "filter", "x", ">", "1")
    assert len(filtered) == 1


def test_history_manager_none_safety():
    assert hm.get_entry(None, 1) is None
    assert hm.entries(None) == []
    assert hm.clear_history(None) == 0
    h = []
    e = hm.add_entry(h, "chat", None)
    assert e.title == "Untitled"