"""SQLite FTS retrieval for grounding FinLLM answers in local evidence."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_&.-]{1,}")
SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class SearchResult:
    rank: int
    rowid: int
    source: str
    chunk_index: int
    score: float
    text: str


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def query_terms(query: str, *, limit: int = 12) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    stopwords = {
        "about",
        "after",
        "because",
        "before",
        "between",
        "could",
        "does",
        "from",
        "have",
        "into",
        "their",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "while",
        "why",
        "with",
        "would",
        "did",
        "how",
        "can",
    }
    for match in TOKEN_RE.finditer(query.lower()):
        term = match.group(0).strip(".-_&")
        if len(term) < 3 or term in stopwords or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def fts_query(query: str) -> str:
    terms = query_terms(query)
    if not terms:
        return '""'
    return " OR ".join(f'"{term}"' for term in terms)


def chunk_text(text: str, *, chunk_chars: int = 1200, overlap: int = 160) -> Iterable[str]:
    text = normalize_text(text)
    if not text:
        return
    if len(text) <= chunk_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(0, end - overlap)


def connect(index_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_index(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(text, source UNINDEXED, chunk_index UNINDEXED);
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def build_index(
    *,
    corpus_path: str | Path,
    index_path: str | Path,
    chunk_chars: int = 1200,
    overlap: int = 160,
    max_chunks: int | None = None,
    reset: bool = True,
) -> dict:
    """Build a local full-text retrieval index from a corpus file."""

    corpus_path = Path(corpus_path)
    index_path = Path(index_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    conn = connect(index_path)
    initialize_index(conn)
    if reset:
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("DELETE FROM metadata")
        conn.commit()

    inserted = 0
    with corpus_path.open("r", encoding="utf8", errors="ignore") as handle:
        for line_number, line in enumerate(tqdm(handle, desc="indexing corpus", unit="lines"), 1):
            for chunk_index, chunk in enumerate(chunk_text(line, chunk_chars=chunk_chars, overlap=overlap)):
                conn.execute(
                    "INSERT INTO chunks_fts(text, source, chunk_index) VALUES (?, ?, ?)",
                    (chunk, f"{corpus_path.name}:{line_number}", chunk_index),
                )
                inserted += 1
                if inserted % 1000 == 0:
                    conn.commit()
                if max_chunks is not None and inserted >= max_chunks:
                    break
            if max_chunks is not None and inserted >= max_chunks:
                break
    conn.commit()
    metadata = {
        "corpus_path": str(corpus_path),
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": inserted,
        "built_at_unix": int(time.time()),
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    for key, value in metadata.items():
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
    conn.commit()
    conn.close()
    return metadata


def index_stats(index_path: str | Path) -> dict:
    index_path = Path(index_path)
    if not index_path.exists():
        return {"exists": False, "path": str(index_path), "chunks": 0}
    conn = connect(index_path)
    initialize_index(conn)
    chunks = int(conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
    metadata = {
        row["key"]: json.loads(row["value"])
        for row in conn.execute("SELECT key, value FROM metadata").fetchall()
    }
    conn.close()
    return {"exists": True, "path": str(index_path), "chunks": chunks, "metadata": metadata}


def search(index_path: str | Path, query: str, *, limit: int = 6) -> list[SearchResult]:
    index_path = Path(index_path)
    if not index_path.exists():
        return []
    match_query = fts_query(query)
    if match_query == '""':
        return []
    conn = connect(index_path)
    initialize_index(conn)
    rows = conn.execute(
        """
        SELECT rowid, source, chunk_index, text, bm25(chunks_fts) AS score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match_query, limit),
    ).fetchall()
    conn.close()
    return [
        SearchResult(
            rank=index + 1,
            rowid=int(row["rowid"]),
            source=str(row["source"]),
            chunk_index=int(row["chunk_index"]),
            score=float(row["score"]),
            text=str(row["text"]),
        )
        for index, row in enumerate(rows)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the FinLLM retrieval index.")
    parser.add_argument("--corpus", default="financial_training_corpus_clean.txt")
    parser.add_argument("--index", default="data/retrieval/finance_fts.sqlite")
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--query")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query:
        for result in search(args.index, args.query, limit=6):
            print(json.dumps(asdict(result), indent=2))
        return
    metadata = build_index(
        corpus_path=args.corpus,
        index_path=args.index,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
        max_chunks=args.max_chunks,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
