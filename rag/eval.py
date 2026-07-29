"""Evaluation policy — scoring a versioned Q&A set, with no I/O of its own.

Three questions, answered in one run:

  - **retrieval rank** — did an expected source page land in the top-k, and how
    high? recall@1 / recall@3 / recall@k and MRR. Embeddings only, so it runs
    without an LLM and is cheap enough to A/B chunking and reranking freely.
  - **faithfulness** — of the answers actually given, how many are supported by
    the retrieved sources? (LLM-as-judge, and see the caveat in the README:
    this metric is too noisy run-to-run to carry a claim.)
  - **abstention** — of the out-of-corpus questions, how many were correctly
    declined instead of answered from the model's own memory?

Rank is the metric that matters here. recall@5 saturated early at 100%, which
means the right page is essentially always *in* the candidate pool — so the
interesting question is not whether retrieval finds it but whether it puts it
where the generator will actually read it.

Everything below is pure given its inputs. Both LLMs arrive as `Answerer` and
`Judge` Protocols, so the retrieval -> score -> aggregate loop runs offline
against fakes. Rendering lives in `rag.report`; this module returns numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import generate as generate_mod
from . import observe
from . import search as search_mod
from .ports import Answerer, Embedder, Judge, Reranker


@dataclass
class EvalItem:
    """One graded question.

    `expected_source_urls` are the pages that should answer it — recall is scored
    at page level, not chunk level, because a question answered from a different
    chunk of the right page is a success, not a miss.

    An empty list together with `in_corpus=False` marks an abstention test: there
    is no right answer, and the correct behaviour is to decline.

    `tag` slices the set (`core` = the general questions, `hard` = exact-keyword
    and sibling-page cases) so the two can be A/B'd separately. Averaging them
    hides the effect being measured, since reranking helps mainly on `hard`.
    """

    id: str
    question: str
    expected_source_urls: list[str]
    in_corpus: bool
    tag: str = "core"


def load_items(path: Path, *, tag: str = "") -> list[EvalItem]:
    """Load the JSONL eval set, optionally keeping only one `tag` slice."""
    if not path.exists():
        raise SystemExit(f"{path} not found — the eval set ships in the repo at eval/qa.jsonl.")
    items: list[EvalItem] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        r = json.loads(ln)
        items.append(EvalItem(
            id=r["id"],
            question=r["question"],
            expected_source_urls=r.get("expected_source_urls", []),
            in_corpus=r.get("in_corpus", True),
            tag=r.get("tag", "core"),
        ))
    if tag:
        items = [it for it in items if it.tag == tag]
    return items


@dataclass
class EvalRow:
    """One question's outcome.

    `grounded` and `faithful` are None when not evaluated — a retrieval-only run
    never generates, and an abstention has no claims to judge. None means "not
    measured", which is different from False, and the summary keeps them apart
    so a retrieval-only run reports `n/a` rather than a misleading 0%.

    `rank` is the 1-based position of the first expected page, or 0 for a miss.
    """

    item: EvalItem
    retrieved_urls: list[str]
    recall_hit: bool
    grounded: bool | None
    faithful: bool | None
    answer: str
    usage_by_model: dict[str, dict[str, int]]
    rank: int = 0
    latency_s: float = 0.0


def first_hit_rank(item: EvalItem, retrieved_urls: Sequence[str]) -> int:
    """1-based rank of the first retrieved page that is an expected source; 0 if none.

    This is the signal reranking actually moves. A hit at rank 1 and a hit at
    rank 4 are identical under recall@5 and completely different under recall@1
    and MRR — and only the former is close enough to the top to be read.
    """
    expected = set(item.expected_source_urls)
    for i, url in enumerate(retrieved_urls, 1):
        if url in expected:
            return i
    return 0


def recall_hit(item: EvalItem, retrieved_urls: Sequence[str]) -> bool:
    """True if any expected page appears in the retrieved top-k."""
    return first_hit_rank(item, retrieved_urls) > 0


def _voyage_total(obj: object) -> int:
    """Cumulative billed tokens on a Voyage client, or 0 for fakes that don't meter.

    The clients accumulate across calls, so snapshotting before and after one
    query yields that query's spend without the client needing to know what a
    query is.
    """
    usage = getattr(obj, "usage", None)
    return usage.get("total_tokens", 0) if usage else 0


def _record_voyage(into: dict[str, dict[str, int]], obj: object, before: int) -> None:
    """File one client's per-query token delta under its model name."""
    delta = _voyage_total(obj) - before
    if delta:
        observe.merge_usage(into, getattr(obj, "model", "voyage"), {"total_tokens": delta})


def evaluate(items: Iterable[EvalItem], db_path: Path, embedder: Embedder,
             answerer: Answerer | None, judge: Judge | None,
             *, k: int = 5, hybrid: bool = True,
             reranker: Reranker | None = None) -> list[EvalRow]:
    """Score each item. One retrieval per item, reused for recall and generation.

    answerer=None -> retrieval-only: rank metrics, no LLM spend.
    judge=None    -> generate and score abstention, but skip faithfulness.

    Tokens are filed per model as they are spent: Voyage from before/after
    snapshots of the clients' counters, Anthropic from the usage each call
    returns. Attributing at the point of spend is what makes the per-model cost
    split real rather than an estimate reconstructed afterwards.
    """
    rows: list[EvalRow] = []
    for item in items:
        t0 = time.perf_counter()
        usage_by_model: dict[str, dict[str, int]] = {}

        embed_before, rerank_before = _voyage_total(embedder), _voyage_total(reranker)
        results = search_mod.search(item.question, db_path, embedder, k=k,
                                    hybrid=hybrid, reranker=reranker)
        _record_voyage(usage_by_model, embedder, embed_before)
        _record_voyage(usage_by_model, reranker, rerank_before)

        retrieved_urls = [r["source_url"] for r in results]
        rank = first_hit_rank(item, retrieved_urls)

        grounded: bool | None = None
        faithful: bool | None = None
        answer_text = ""

        if answerer is not None:
            out = generate_mod.answer_from_results(item.question, results, answerer)
            grounded = out["grounded"]
            answer_text = out["answer"]
            if out["usage"]:
                observe.merge_usage(usage_by_model, answerer.model, out["usage"])
            # Judge only a real answer to an in-corpus question: an abstention has
            # no claims to check, and an out-of-corpus question has no right answer
            # to be faithful to.
            if judge is not None and item.in_corpus and grounded:
                block = generate_mod.build_sources_block(results)
                verdict = judge.judge(item.question, answer_text, block)
                faithful = verdict.faithful
                if verdict.usage:
                    observe.merge_usage(usage_by_model, judge.model, verdict.usage)

        rows.append(EvalRow(
            item=item, retrieved_urls=retrieved_urls, recall_hit=rank > 0,
            grounded=grounded, faithful=faithful, answer=answer_text,
            usage_by_model=usage_by_model, rank=rank,
            latency_s=time.perf_counter() - t0,
        ))
    return rows


def summarize(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """Aggregate per-question rows into the headline metrics.

    Pure, and separated from `evaluate` so the arithmetic can be tested against
    hand-built rows without running a pipeline.
    """
    in_corpus = [r for r in rows if r.item.in_corpus]
    oob = [r for r in rows if not r.item.in_corpus]

    answered = [r for r in in_corpus if r.grounded is True]
    judged = [r for r in answered if r.faithful is not None]
    false_abstentions = [r for r in in_corpus if r.grounded is False]
    # Score abstention only where generation actually ran. A retrieval-only pass
    # cannot measure it, so the rate stays n/a instead of reporting a fake 0%.
    oob_generated = [r for r in oob if r.grounded is not None]
    oob_abstained = [r for r in oob_generated if r.grounded is False]

    def ratio(num: int, den: int) -> float | None:
        return (num / den) if den else None

    n_ic = len(in_corpus)

    def recall_at(cutoff: int) -> float | None:
        return ratio(sum(1 for r in in_corpus if 0 < r.rank <= cutoff), n_ic)

    mrr = (sum(1.0 / r.rank for r in in_corpus if r.rank) / n_ic) if n_ic else None

    usage_by_model: dict[str, dict[str, int]] = {}
    for r in rows:
        for model, usage in r.usage_by_model.items():
            observe.merge_usage(usage_by_model, model, usage)
    latencies = [r.latency_s for r in rows if r.latency_s]

    return {
        "n_total": len(rows),
        "n_in_corpus": n_ic,
        "n_oob": len(oob),
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        # Retrieval returns at most k results, so any rank > 0 is within top-k.
        "recall_at_k": recall_at(10**9),
        "mrr": mrr,
        "n_answered": len(answered),
        "n_judged": len(judged),
        "faithfulness": ratio(sum(1 for r in judged if r.faithful), len(judged)),
        "n_false_abstentions": len(false_abstentions),
        "abstention_rate": ratio(len(oob_abstained), len(oob_generated)),
        "usage_by_model": usage_by_model,
        "cost": observe.cost_usd(usage_by_model),
        "latencies": latencies,
    }
