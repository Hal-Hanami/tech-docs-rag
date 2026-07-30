"""The surface another repository imports — design §9.1, pinned on the provider's side.

`incident-triage-agent` reuses this package as its runbook search: it does
`from rag import index, search` and `from rag.clients import voyage`, then calls
them. Nothing in either repository checked that those calls still resolve, so when
`rag.embed` and `rag.rerank` were folded into `rag.clients.voyage`, the consumer's
live path broke while both test suites stayed green — its tests drive a hand-written
stub of this package, and a stub cannot notice that the real thing moved.

These tests fail in *this* repository's CI the moment the shape changes, which is
the only place the change is visible before someone runs the other system. They
need no key and no network: the clients resolve an explicit `api_key` without
touching the environment, and nothing here calls a method that opens a socket.

Moving any of this is allowed. Moving it silently is not — update this file in the
same commit, so the break is a decision someone made rather than one they shipped.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from rag import index, search
from rag.clients import voyage

from test_store import DOCS, FakeEmbedder


def test_the_modules_the_consumer_imports_resolve():
    # `from rag import index, search` / `from rag.clients import voyage`
    assert callable(search.search) and callable(index.build)
    assert hasattr(voyage, "VoyageEmbedder") and hasattr(voyage, "VoyageReranker")


def test_search_accepts_the_call_the_consumer_makes():
    # triage/retrieve.py: search.search(query, db_path, embedder, k=, hybrid=, reranker=)
    inspect.signature(search.search).bind(
        "query", Path("index.db"), FakeEmbedder(), k=4, hybrid=True, reranker=None)


def test_index_build_accepts_the_call_the_consumer_makes():
    # triage/retrieve.py: index.build(chunks_path, db_path, embedder)
    inspect.signature(index.build).bind(Path("chunks.jsonl"), Path("index.db"), FakeEmbedder())


def test_the_voyage_clients_construct_without_touching_the_environment():
    # The consumer builds both eagerly in its constructor and reads `.model` /
    # `.usage` to split its own cost report per model.
    embedder = voyage.VoyageEmbedder(api_key="not-a-real-key")
    reranker = voyage.VoyageReranker(api_key="not-a-real-key")
    assert embedder.model == "voyage-4-lite" and reranker.model == "rerank-2.5-lite"
    assert embedder.usage["total_tokens"] == 0 and reranker.usage["total_tokens"] == 0


def test_a_result_row_carries_the_fields_the_consumer_reads(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index.build(chunks, db, FakeEmbedder())

    row = search.search("cache the prompt prefix", db, FakeEmbedder(), k=1)[0]
    # the consumer folds chunks to runbooks by `source_url`, ranks by `score`,
    # and cites with `section_path` + `url`
    assert {"source_url", "score", "section_path", "url", "text", "chunk_id"} <= set(row)


def test_index_build_reports_the_stats_the_consumer_passes_on(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    stats = index.build(chunks, tmp_path / "index.db", FakeEmbedder())
    assert {"count", "dim", "model"} <= set(stats)
