"""Local chunking and keyword retrieval (TF-IDF style). No API calls, no extra dependencies."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from models import LoadedDocument

STOPWORDS = {"the", "and", "for", "are", "was", "with", "this", "that", "from", "have", "has", "what", "which",
             "who", "how", "why", "when", "where", "does", "did", "can", "you", "your", "about", "into", "than",
             "then", "them", "they", "their", "its", "of", "in", "on", "to", "is", "it", "a", "an", "be", "by"}
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Chunk:
    doc_name: str
    location: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc_name} | {self.location}"


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    size = max(50, size)
    overlap = max(0, min(overlap, size // 2))
    if len(text) <= size:
        return [text] if text.strip() else []
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def chunk_documents(docs: list[LoadedDocument], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for doc in docs:
        for seg in doc.segments:
            pieces = split_text(seg.text, size, overlap)
            for i, piece in enumerate(pieces, start=1):
                loc = seg.location if len(pieces) == 1 else f"{seg.location}, part {i}/{len(pieces)}"
                chunks.append(Chunk(doc.name, loc, piece))
    return chunks


def retrieve(chunks: list[Chunk], query: str, top_k: int = 6, max_chars: int = 12000) -> list[Chunk]:
    """Return the most relevant chunks for the query (empty list if nothing matches)."""
    terms = set(tokenize(query))
    if not terms or not chunks:
        return []
    counts = [Counter(tokenize(c.text)) for c in chunks]
    df = Counter()
    for tf in counts:
        df.update(tf.keys())
    n = len(chunks)
    scored = []
    for chunk, tf in zip(chunks, counts):
        score = sum(math.log(1 + n / df[t]) * (1 + math.log(tf[t])) for t in terms if t in tf)
        if score > 0:
            scored.append((score / math.sqrt(sum(tf.values()) or 1), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked, total = [], 0
    for _, chunk in scored[:top_k]:
        if picked and total + len(chunk.text) > max_chars:
            break
        picked.append(chunk)
        total += len(chunk.text)
    return picked


def sample_chunks(chunks: list[Chunk], max_chars: int) -> tuple[list[Chunk], float]:
    """Evenly spaced chunks within the budget, plus the fraction of text covered."""
    if not chunks:
        return [], 0.0
    total = sum(len(c.text) for c in chunks)
    if total == 0:
        return [], 0.0
    if total <= max_chars:
        return list(chunks), 1.0
    k = min(len(chunks), max(1, int(max_chars // (total / len(chunks)))))
    step = len(chunks) / k
    out, used = [], 0
    for i in range(k):
        c = chunks[int(i * step)]
        if out and used + len(c.text) > max_chars:
            break
        out.append(c)
        used += len(c.text)
    return out, used / total


def pack_batches(chunks: list[Chunk], max_chars: int) -> list[list[Chunk]]:
    batches, cur, size = [], [], 0
    for c in chunks:
        if cur and size + len(c.text) > max_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c.text)
    if cur:
        batches.append(cur)
    return batches


def format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    return "\n\n".join(f"[Source: {c.ref}]\n{c.text}" for c in chunks)