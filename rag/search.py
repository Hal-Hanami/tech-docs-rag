"""Stage: retrieval — dense, hybrid (dense + BM25), and reranked (M2 + M5).

The pipeline:
  [2] hybrid  : dense kNN (sqlite-vec) ∪ BM25 (FTS5), fused by RRF
  [3] rerank  : a cross-encoder reorders the fused pool against the query

`search()` is the single entry point; `hybrid` and `reranker` toggle the M5
stages so the eval harness can measure before→after on one index:
  - dense only          : hybrid=False, reranker=None        (the M2 baseline)
  - hybrid              : hybrid=True,  reranker=None
  - hybrid + rerank     : hybrid=True,  reranker=<Reranker>   (the M5 product)

`fts_match_query` and `rrf_fuse` are pure functions, unit-tested offline; the
reranker sits behind the `Reranker` Protocol (faked offline). Every result keeps
the full citation metadata from M1, plus a unified `score` for display/ranking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from . import observe, store
from .ports import Embedder, Reranker
from .observe import Trace

# How many candidates each retriever contributes before fusion/rerank. Generous
# enough that the right page is in the pool (recall headroom), small enough to
# keep the rerank call cheap.
CANDIDATES = 50
RRF_K = 60  # standard RRF damping constant; larger = flatter rank weighting


def fts_match_query(query: str) -> str:
    """Turn a natural-language query into a safe FTS5 MATCH expression.

    Each word becomes a quoted phrase OR-ed together: quoting neutralizes FTS5
    operators/punctuation (so "How do I?" can't blow up the parser), and OR keeps
    it recall-oriented (any keyword can match). Returns "" when there's no token.
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


def rrf_fuse(rankings: Sequence[Sequence[dict[str, Any]]], *,
             k: int = RRF_K, key: str = "chunk_id") -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of several ranked result lists.

    score(d) = Σ_lists 1 / (k + rank_in_that_list). Rank-based, so dense cosine
    distances and BM25 scores — wildly different scales — combine without any
    normalization. Documents in multiple lists accumulate score and float up.
    """
    scores: dict[Any, float] = {}
    rep: dict[Any, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            ident = doc[key]
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            rep.setdefault(ident, doc)
    fused = []
    for ident in sorted(scores, key=lambda i: scores[i], reverse=True):
        doc = dict(rep[ident])
        doc["rrf_score"] = scores[ident]
        fused.append(doc)
    return fused


def _final_score(doc: dict[str, Any]) -> float:
    """Pick the score that determined a doc's final rank, for display."""
    if "rerank_score" in doc:
        return doc["rerank_score"]
    if "rrf_score" in doc:
        return doc["rrf_score"]
    if "distance" in doc:  # dense-only: cosine distance -> similarity
        return max(0.0, 1.0 - doc["distance"])
    return 0.0


def search(query: str, db_path: Path, embedder: Embedder, k: int = 5, *,
           hybrid: bool = True, reranker: Reranker | None = None,
           candidates: int = CANDIDATES, trace: Trace | None = None) -> list[dict[str, Any]]:
    """Retrieve the top-k chunks for `query` with citation metadata.

    hybrid=True    fuses dense kNN with BM25 (RRF); False is dense-only (M2).
    reranker!=None reorders the fused pool with a cross-encoder (M5 precision).
    trace          if given, records per-stage latency for the observability layer.
    """
    if not db_path.exists():
        raise SystemExit(f"{db_path} not found — build it first with `python -m rag index`.")
    db = store.connect(db_path)
    try:
        with observe.span(trace, "embed"):
            query_vec = embedder.embed([query], input_type="query")[0]
        with observe.span(trace, "dense"):
            dense = store.knn(db, query_vec, candidates)
        if hybrid:
            with observe.span(trace, "bm25"):
                sparse = store.bm25_search(db, fts_match_query(query), candidates)
            with observe.span(trace, "fuse"):
                pool = rrf_fuse([dense, sparse])
        else:
            pool = dense

        if reranker is not None and pool:
            with observe.span(trace, "rerank"):
                ranked = reranker.rerank(query, [d["text"] for d in pool], top_k=k)
            pool = [dict(pool[idx], rerank_score=score) for idx, score in ranked]

        out = []
        for doc in pool[:k]:
            doc = dict(doc)
            doc["score"] = _final_score(doc)
            out.append(doc)
        return out
    finally:
        db.close()
