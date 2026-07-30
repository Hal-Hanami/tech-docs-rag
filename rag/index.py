"""Stage: build the vector index from `data/chunks.jsonl`."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import store
from .ports import Embedder


def _read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found — run the ingest step first (`python -m ingest all`).")
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return rows[:limit] if limit else rows


def build(chunks_path: Path, db_path: Path, embedder: Embedder, *, limit: int = 0) -> dict[str, Any]:
    """Embed every chunk and (re)build the sqlite-vec index from scratch."""
    rows = _read_jsonl(chunks_path, limit)
    texts = [r["text"] for r in rows]

    t0 = time.time()
    embeddings = embedder.embed(texts, input_type="document")
    embed_secs = time.time() - t0
    if not embeddings:
        raise SystemExit("no embeddings returned")
    dim = len(embeddings[0])

    db_path.unlink(missing_ok=True)  # fresh build — avoid stale rows / dim drift
    db = store.connect(db_path)
    store.create(db, dim)
    for rowid, (row, vec) in enumerate(zip(rows, embeddings)):
        if len(vec) != dim:
            raise SystemExit(f"ragged embedding dim at row {rowid}: {len(vec)} != {dim}")
        store.insert(db, rowid, vec, row)
    store.set_meta(db, "model", embedder.model)
    store.set_meta(db, "dim", str(dim))
    store.set_meta(db, "count", str(len(rows)))
    db.commit()
    db.close()

    return {
        "count": len(rows),
        "dim": dim,
        "model": embedder.model,
        "embed_secs": round(embed_secs, 1),
        "db_path": str(db_path),
    }
