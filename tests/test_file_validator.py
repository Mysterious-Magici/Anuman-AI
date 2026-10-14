import pytest

# pyrefly: ignore [missing-import]
from utils.file_validator import ValidationError, unique_name, validate_file


def test_rejects_unknown_extension():
    with pytest.raises(ValidationError):
        validate_file("virus.exe", b"x", 5)


def test_rejects_empty_file():
    with pytest.raises(ValidationError):
        validate_file("a.txt", b"", 5)


def test_rejects_oversized_file():
    with pytest.raises(ValidationError):
        validate_file("a.txt", b"x" * (2 * 1024 * 1024), 1)


def test_rejects_fake_office_file():
    with pytest.raises(ValidationError):
        validate_file("a.docx", b"not a zip", 5)


def test_accepts_text_file():
    validate_file("notes.txt", b"hello", 5)


def test_unique_name_handles_duplicates():
    assert unique_name("a.pdf", {"a.pdf"}) == "a (2).pdf"
    assert unique_name("a.pdf", {"a.pdf", "a (2).pdf"}) == "a (3).pdf"
    assert unique_name("b.pdf", {"a.pdf"}) == "b.pdf"
    assert unique_name(None, None) == "unnamed"


def test_validate_file_edge_cases():
    with pytest.raises(ValidationError):
        validate_file(None, b"abc", 5)
    with pytest.raises(ValidationError):
        validate_file("test.txt", None, 5)