"""sqlite-vec dense-vector store + FTS5 keyword index — one file, both retrievers.

Tables joined by rowid:
  - `vec_chunks`  : a vec0 virtual table holding the float[dim] embeddings (dense)
  - `fts_chunks`  : an FTS5 virtual table over the chunk text (BM25 / keyword)
  - `chunks`      : a plain table holding the citable metadata + chunk text
A separate `meta` key/value table records the embedding model and dim so the
index is self-describing. Both retrievers live in one SQLite file, so the
hybrid (dense + BM25) needs no second service and no extra infrastructure.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any, Mapping

import sqlite_vec

# Metadata columns mirrored from chunks.jsonl (the citation trail, design §2.3).
_META_COLS = ("chunk_id", "url", "source_url", "page_title", "section_path", "anchor", "text")


def connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _serialize(vec: list[float]) -> bytes:
    """Pack a float vector into sqlite-vec's compact little-endian float32 blob."""
    return struct.pack(f"<{len(vec)}f", *vec)


def create(db: sqlite3.Connection, dim: int) -> None:
    db.execute(
        f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
        f"embedding float[{dim}] distance_metric=cosine)"
    )
    # BM25 keyword index over the same chunks. `porter` stemming lets a query
    # for "cache" match "caching"; the index keeps its own copy of the text (the
    # corpus is small, so the few MB beat the footguns of external-content sync).
    db.execute(
        "CREATE VIRTUAL TABLE fts_chunks USING fts5(text, tokenize='porter unicode61')"
    )
    db.execute(
        "CREATE TABLE chunks ("
        "rowid INTEGER PRIMARY KEY, chunk_id TEXT, url TEXT, source_url TEXT, "
        "page_title TEXT, section_path TEXT, anchor TEXT, text TEXT)"
    )
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")


def insert(db: sqlite3.Connection, rowid: int, embedding: list[float],
           meta: Mapping[str, Any]) -> None:
    db.execute(
        "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
        (rowid, _serialize(embedding)),
    )
    db.execute(
        "INSERT INTO chunks(rowid, chunk_id, url, source_url, page_title, "
        "section_path, anchor, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rowid, meta["id"], meta["url"], meta["source_url"], meta["page_title"],
         meta["section_path"], meta["anchor"], meta["text"]),
    )
    db.execute(
        "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)", (rowid, meta["text"])
    )


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def get_meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def knn(db: sqlite3.Connection, query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    """Return the k nearest chunks (smallest cosine distance first) with metadata."""
    cur = db.execute(
        "SELECT c.chunk_id, c.url, c.source_url, c.page_title, c.section_path, "
        "c.anchor, c.text, v.distance "
        "FROM vec_chunks v JOIN chunks c ON c.rowid = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? "
        "ORDER BY v.distance",
        (_serialize(query_embedding), k),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def bm25_search(db: sqlite3.Connection, match_query: str, k: int) -> list[dict[str, Any]]:
    """Return the top-k chunks by BM25 keyword relevance, with metadata.

    `match_query` is an FTS5 MATCH expression (see search.fts_match_query). FTS5's
    bm25() returns a negative score where *more negative = more relevant*, so
    ascending order puts the best match first. An empty query yields no rows.
    """
    if not match_query:
        return []
    cur = db.execute(
        "SELECT c.chunk_id, c.url, c.source_url, c.page_title, c.section_path, "
        "c.anchor, c.text, bm25(fts_chunks) AS bm25 "
        "FROM fts_chunks f JOIN chunks c ON c.rowid = f.rowid "
        "WHERE fts_chunks MATCH ? "
        "ORDER BY bm25 LIMIT ?",
        (match_query, k),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
