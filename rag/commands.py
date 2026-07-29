"""Orchestration: wiring concrete clients to policy, one function per command.

This is the only layer that knows both halves — that a `VoyageEmbedder` is the
thing satisfying `Embedder`, and that `rag.search` is what wants one. Policy
modules stay unaware of Voyage and Anthropic; `rag.cli` stays unaware of both
and only maps flags to these calls.

The functions take plain keyword arguments rather than an `argparse.Namespace`,
so they can be called from a test, a notebook, or another entry point without
constructing a fake parser result.
"""

from __future__ import annotations

from . import config, index as index_mod, observe, report
from . import eval as eval_mod
from . import generate as generate_mod
from . import search as search_mod
from .clients.claude import ClaudeAnswerer, ClaudeJudge
from .clients.voyage import DEFAULT_EMBED_MODEL, VoyageEmbedder, VoyageReranker
from .ports import Reranker


def _reranker(no_rerank: bool) -> Reranker | None:
    """The Voyage reranker unless it was switched off."""
    return None if no_rerank else VoyageReranker()


def _mode(dense_only: bool, no_rerank: bool) -> str:
    """One-line description of the active retrieval configuration, for the banner.

    Printed on every run because the ablation flags are independent: seeing
    `dense + rerank` in the output is what stops a reader from assuming
    `--dense-only` alone meant the un-reranked baseline.
    """
    base = "dense" if dense_only else "hybrid (dense+BM25)"
    return base + ("" if no_rerank else " + rerank")


def run_index(*, model: str = DEFAULT_EMBED_MODEL, limit: int = 0) -> None:
    """Embed `data/chunks.jsonl` into the SQLite index."""
    embedder = VoyageEmbedder(model=model)
    print(f"embedding chunks with {model} (input_type=document) ...")
    stats = index_mod.build(config.CHUNKS_FILE, config.DB_FILE, embedder, limit=limit)
    print(
        f"\nindexed {stats['count']} chunks  dim={stats['dim']}  "
        f"model={stats['model']}  embed={stats['embed_secs']}s\n"
        f"  -> {stats['db_path']}"
    )


def run_query(text: str, *, k: int = 5, model: str = DEFAULT_EMBED_MODEL,
              dense_only: bool = False, no_rerank: bool = False) -> None:
    """Print the top-k retrieved chunks with their citations."""
    embedder = VoyageEmbedder(model=model)
    results = search_mod.search(text, config.DB_FILE, embedder, k=k,
                                hybrid=not dense_only, reranker=_reranker(no_rerank))
    if not results:
        print("no results (is the index built?)")
        return
    print(f'query: "{text}"  (top {len(results)}, {_mode(dense_only, no_rerank)})\n')
    for rank, r in enumerate(results, 1):
        snippet = " ".join(r["text"].split())[:200]
        print(f"[{rank}] score={r['score']:.3f}  {r['section_path']}")
        print(f"     {r['url']}")
        print(f"     {snippet}…\n")


def run_ask(text: str, *, k: int = 5, max_tokens: int = 4096,
            dense_only: bool = False, no_rerank: bool = False) -> None:
    """Answer one question and print the answer, its citations, and its trace."""
    embedder = VoyageEmbedder()  # must match the model the index was built with
    reranker = _reranker(no_rerank)
    answerer = ClaudeAnswerer(max_tokens=max_tokens)

    trace = observe.Trace()
    embed_before = embedder.usage["total_tokens"]
    rerank_before = reranker.usage["total_tokens"] if reranker is not None else 0
    out = generate_mod.answer(text, config.DB_FILE, embedder, answerer, k=k,
                              hybrid=not dense_only, reranker=reranker, trace=trace)
    # Voyage clients report cumulative totals, so the per-request figure is the
    # delta across this call.
    trace.add_usage(embedder.model,
                    {"total_tokens": embedder.usage["total_tokens"] - embed_before})
    if reranker is not None:
        trace.add_usage(reranker.model,
                        {"total_tokens": reranker.usage["total_tokens"] - rerank_before})
    trace.add_usage(answerer.model, out["usage"])

    print(f"Q: {text}\n")
    print(out["answer"] + "\n")
    if out["sources"]:
        print(f"Sources ({'grounded' if out['grounded'] else 'abstained'}):")
        for i, r in enumerate(out["sources"], 1):
            print(f"  [{i}] {r['section_path']}")
            print(f"      {r['url']}")
    print("\n" + "\n".join(observe.format_trace(trace)))


def run_eval(*, k: int = 5, tag: str = "", limit: int = 0, retrieval_only: bool = False,
             no_judge: bool = False, max_tokens: int = 4096,
             dense_only: bool = False, no_rerank: bool = False) -> None:
    """Score the eval set and print the per-question table plus headline metrics."""
    embedder = VoyageEmbedder()  # must match the model the index was built with
    reranker = _reranker(no_rerank)
    items = eval_mod.load_items(config.EVAL_FILE, tag=tag)
    if limit:
        items = items[:limit]

    answerer = None if retrieval_only else ClaudeAnswerer(max_tokens=max_tokens)
    judge = None if (retrieval_only or no_judge) else ClaudeJudge()

    mode = ("retrieval-only" if retrieval_only
            else "generate (no judge)" if no_judge else "full")
    print(f"evaluating {len(items)} questions  k={k}  mode={mode}  "
          f"retrieval={_mode(dense_only, no_rerank)} ...\n")

    rows = eval_mod.evaluate(items, config.DB_FILE, embedder, answerer, judge, k=k,
                             hybrid=not dense_only, reranker=reranker)
    print(report.format_report(rows, eval_mod.summarize(rows), k=k))
