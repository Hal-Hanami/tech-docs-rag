"""Offline tests for the store + index + search round-trip — design §2.3, §3.

No network: a deterministic bag-of-words `FakeEmbedder` stands in for Voyage so
we can assert that sqlite-vec actually orders results by semantic overlap and
that the citation metadata survives the round-trip. (Real Voyage embeddings are
exercised separately via `python -m rag query`.)
"""

from __future__ import annotations

import json
from pathlib import Path

from rag import index as index_mod
from rag import search as search_mod


class FakeEmbedder:
    """Maps text to a normalized term-count vector over a fixed vocabulary.

    Shared vocabulary => non-zero cosine similarity; disjoint => ~0. Good enough
    to verify nearest-neighbour ordering deterministically and offline.
    """

    model = "fake-bow"
    dim = 8
    VOCAB = ["claude", "prompt", "cache", "tool", "stream", "token", "model", "vision"]

    def embed(self, texts, *, input_type):
        vecs = []
        for t in texts:
            low = t.lower()
            v = [float(low.count(w)) for w in self.VOCAB]
            norm = (sum(x * x for x in v) ** 0.5) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


DOCS = [
    {"id": "pc#0", "url": "u/pc#a", "source_url": "u/pc", "page_title": "Prompt caching",
     "section_path": "Prompt caching > Overview", "anchor": "a",
     "text": "Prompt cache reduces cost by caching the prompt prefix across requests."},
    {"id": "tu#0", "url": "u/tu#b", "source_url": "u/tu", "page_title": "Tool use",
     "section_path": "Tool use > Defining tools", "anchor": "b",
     "text": "Define a tool with a schema and let Claude call the tool."},
    {"id": "vs#0", "url": "u/vs#c", "source_url": "u/vs", "page_title": "Vision",
     "section_path": "Vision > Images", "anchor": "c",
     "text": "Vision lets the model read an image and answer questions about it."},
]


def _build(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    db = tmp_path / "index.db"
    stats = index_mod.build(chunks, db, FakeEmbedder())
    assert stats["count"] == 3
    assert stats["dim"] == 8
    return db


def test_nearest_neighbour_ranks_semantic_overlap_first(tmp_path):
    db = _build(tmp_path)
    results = search_mod.search("how do I cache the prompt prefix?", db, FakeEmbedder(), k=3)
    assert results[0]["chunk_id"] == "pc#0"
    # cosine distance for the matching doc should beat the unrelated ones
    assert results[0]["distance"] < results[-1]["distance"]


def test_results_carry_citation_metadata(tmp_path):
    db = _build(tmp_path)
    top = search_mod.search("call a tool with a schema", db, FakeEmbedder(), k=1)[0]
    assert top["chunk_id"] == "tu#0"
    assert top["source_url"] == "u/tu"
    assert top["section_path"] == "Tool use > Defining tools"
    assert top["url"] == "u/tu#b"


def test_k_limits_result_count(tmp_path):
    db = _build(tmp_path)
    assert len(search_mod.search("model", db, FakeEmbedder(), k=2)) == 2
    assert len(search_mod.search("model", db, FakeEmbedder(), k=1)) == 1
