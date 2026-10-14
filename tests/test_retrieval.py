# pyrefly: ignore [missing-import]
from models import LoadedDocument, Segment
# pyrefly: ignore [missing-import]
from utils.retrieval import chunk_documents, pack_batches, retrieve, sample_chunks, split_text


def _doc(name, *texts):
    return LoadedDocument(name, ".txt", 1, segments=[Segment(t, f"page {i}") for i, t in enumerate(texts, 1)])


def test_split_text_makes_progress_and_covers_text():
    text = "word " * 1000
    parts = split_text(text, 200, 50)
    assert len(parts) > 10 and all(len(p) <= 200 for p in parts)


def test_chunk_locations_include_part_numbers():
    chunks = chunk_documents([_doc("a.txt", "x " * 500)], 200, 20)
    assert chunks[0].location.startswith("page 1, part 1/")


def test_retrieve_ranks_relevant_chunk_first():
    chunks = chunk_documents([_doc("a.txt", "apples and oranges", "the invoice total is 500 dollars", "birds fly")], 1500, 0)
    hits = retrieve(chunks, "what is the invoice total", top_k=2)
    assert hits and "invoice" in hits[0].text


def test_retrieve_returns_nothing_without_match():
    chunks = chunk_documents([_doc("a.txt", "apples")], 1500, 0)
    assert retrieve(chunks, "zebra") == []


def test_sample_chunks_reports_partial_coverage():
    chunks = chunk_documents([_doc("a.txt", "word " * 2000)], 200, 0)
    picked, coverage = sample_chunks(chunks, 1000)
    assert 0 < coverage < 1 and sum(len(c.text) for c in picked) <= 1000


def test_pack_batches_respects_budget():
    chunks = chunk_documents([_doc("a.txt", "word " * 2000)], 200, 0)
    assert all(sum(len(c.text) for c in b) <= 500 for b in pack_batches(chunks, 500))


def test_retrieval_edge_cases():
    # pyrefly: ignore [missing-import]
    from utils.retrieval import format_context
    assert split_text("", 100, 20) == []
    assert split_text("   ", 100, 20) == []
    assert split_text("hello world", -10, -5) == ["hello world"]
    assert sample_chunks([], 500) == ([], 0.0)
    assert format_context([]) == ""