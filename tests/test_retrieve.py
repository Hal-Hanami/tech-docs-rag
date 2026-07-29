"""Offline tests for the M5 retrieval pipeline: BM25 (FTS5), RRF fusion, hybrid
search, and reranking.

No network, no keys: the FakeEmbedder (M2) stands in for Voyage embeddings and a
scripted reranker stands in for the Voyage reranker. FTS5/BM25 runs in the real
SQLite, so the keyword path is exercised for real. (The live Voyage reranker is
exercised via `python -m rag query`/`eval`.)
"""

from __future__ import annotations

import json
from pathlib import Path

from rag import index as index_mod
from rag import observe
from rag import search as search_mod
from rag import store

from test_store import DOCS, FakeEmbedder  # reuse the M2 fixtures


def _build(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_mod.build(chunks, db, FakeEmbedder())
    return db


# --- pure helpers -----------------------------------------------------------

def test_fts_match_query_quotes_each_token_and_ors_them():
    assert search_mod.fts_match_query("How do I cache?") == '"how" OR "do" OR "i" OR "cache"'


def test_fts_match_query_is_empty_when_no_word_tokens():
    # punctuation-only → no MATCH, so bm25_search short-circuits to no rows
    assert search_mod.fts_match_query("?!  ...") == ""


def test_rrf_fuse_ranks_documents_in_both_lists_highest():
    dense = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    sparse = [{"chunk_id": "y"}, {"chunk_id": "w"}]
    fused = search_mod.rrf_fuse([dense, sparse])
    ids = [d["chunk_id"] for d in fused]
    assert ids[0] == "y"                       # in both lists → highest combined score
    assert set(ids) == {"x", "y", "z", "w"}    # union of both lists
    assert "rrf_score" in fused[0]
    # scores are monotonically non-increasing
    scores = [d["rrf_score"] for d in fused]
    assert scores == sorted(scores, reverse=True)


# --- BM25 over the real FTS5 index ------------------------------------------

def test_bm25_search_finds_the_keyword_match(tmp_path):
    db = store.connect(_build(tmp_path))
    try:
        rows = store.bm25_search(db, search_mod.fts_match_query("prompt prefix caching"), 5)
    finally:
        db.close()
    assert rows[0]["chunk_id"] == "pc#0"       # the prompt-caching chunk
    assert rows[0]["source_url"] == "u/pc"     # citation metadata survives


def test_bm25_search_empty_query_returns_nothing(tmp_path):
    db = store.connect(_build(tmp_path))
    try:
        assert store.bm25_search(db, "", 5) == []
    finally:
        db.close()


# --- hybrid search ----------------------------------------------------------

def test_hybrid_surfaces_a_keyword_dense_is_blind_to(tmp_path):
    db = _build(tmp_path)
    # "schema" is in tu#0's text but absent from the FakeEmbedder vocab, so the
    # dense vector is blind to it; BM25 catches it and the fusion surfaces tu#0.
    top = search_mod.search("schema", db, FakeEmbedder(), k=1, hybrid=True)
    assert top[0]["chunk_id"] == "tu#0"
    assert "score" in top[0]


def test_search_records_trace_spans(tmp_path):
    # M6: a Trace passed to search() captures one span per retrieval stage, in order.
    db = _build(tmp_path)
    trace = observe.Trace()
    search_mod.search("how do I cache the prompt prefix?", db, FakeEmbedder(),
                      k=2, hybrid=True, reranker=None, trace=trace)
    assert [name for name, _ in trace.spans] == ["embed", "dense", "bm25", "fuse"]
    # dense-only drops the BM25/fuse stages
    trace2 = observe.Trace()
    search_mod.search("model", db, FakeEmbedder(), k=2, hybrid=False, trace=trace2)
    assert [name for name, _ in trace2.spans] == ["embed", "dense"]


def test_dense_only_skips_bm25(tmp_path):
    db = _build(tmp_path)
    # hybrid=False must not error and returns dense top-k with a similarity score.
    res = search_mod.search("how do I cache the prompt prefix?", db, FakeEmbedder(),
                            k=2, hybrid=False)
    assert len(res) == 2
    assert res[0]["chunk_id"] == "pc#0"
    assert "score" in res[0] and "rrf_score" not in res[0]


# --- reranking --------------------------------------------------------------

class SpyReranker:
    """Records the candidate docs it received and returns them in reverse order,
    so a test can assert that search applies the reranker's ordering verbatim."""

    model = "spy-reverse"

    def __init__(self) -> None:
        self.received: list[str] | None = None

    def rerank(self, query, documents, *, top_k=None):
        self.received = list(documents)
        n = len(documents)
        ranked = [(idx, float(pos)) for pos, idx in enumerate(reversed(range(n)))]
        return ranked[:top_k] if top_k is not None else ranked


def test_reranker_reorders_results_and_sets_score(tmp_path):
    db = _build(tmp_path)
    spy = SpyReranker()
    res = search_mod.search("how do I cache the prompt prefix?", db, FakeEmbedder(),
                            k=3, hybrid=True, reranker=spy)
    assert spy.received is not None                       # the reranker was called
    # search returns the candidates in the reranker's (reversed) order, top-k
    expected = list(reversed(spy.received))[:3]
    assert [r["text"] for r in res] == expected
    # the final score is the rerank score, and citation metadata is intact
    assert res[0]["score"] == res[0]["rerank_score"]
    assert "section_path" in res[0] and "url" in res[0]
