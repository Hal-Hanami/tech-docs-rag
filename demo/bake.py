"""Bake the cached public demo (M7).

Runs the real `rag ask` pipeline over a curated set of questions and writes
`demo/examples.json` — for each question: the grounded answer, its citations
(`section_path` + deep-link URL, **never the chunk body**), the grounded/abstained
flag, and the M6 trace (per-stage latency + per-model token cost). The public
Streamlit app (`app.py`) serves these precomputed entries, so the public URL
makes **zero LLM calls** and ships **no `index.db`**: free, always-on, no key on
the host, and no corpus redistribution.

The numbers in the demo are therefore *real measured* numbers from one honest run
at the M6-optimized `k=3`, not mock-ups.

Re-bake when the corpus or models change (needs VOYAGE + ANTHROPIC keys in
`.env` and a built `data/index.db`):

    .venv/bin/python -m demo.bake
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from rag import generate as generate_mod
from rag import observe
from rag.clients.claude import ANSWER_MODEL, ClaudeAnswerer
from rag.clients.voyage import VoyageEmbedder, VoyageReranker
from rag.config import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data" / "index.db"
OUT_FILE = ROOT / "demo" / "examples.json"

K = 3  # generation context size, cut from 5 after recall@3 measured 100%

# Curated so the demo lands on robust, talking-point-friendly pages, plus one
# deliberately out-of-corpus question that must trigger an abstention.
QUESTIONS = [
    "How does prompt caching reduce cost, and how do I enable it?",
    "How does tool use work with the Claude Messages API?",
    "What is extended thinking and when should I use it?",
    "How do I stream a response from the Messages API?",
    "How do I get structured JSON output from Claude?",
    "How does the Message Batches API work, and what does it cost?",
    "How can I count the tokens in a request before I send it?",
    "What is a good recipe for sourdough bread?",  # out-of-corpus -> must abstain
]


def bake_one(question: str, embedder: VoyageEmbedder, reranker: VoyageReranker,
             answerer: ClaudeAnswerer) -> dict:
    """Run one real request and capture answer + citations + trace (mirrors cmd_ask)."""
    trace = observe.Trace()
    embed_before = embedder.usage["total_tokens"]
    rerank_before = reranker.usage["total_tokens"]
    out = generate_mod.answer(question, DB_FILE, embedder, answerer, k=K,
                              hybrid=True, reranker=reranker, trace=trace)
    trace.add_usage(embedder.model,
                    {"total_tokens": embedder.usage["total_tokens"] - embed_before})
    trace.add_usage(reranker.model,
                    {"total_tokens": reranker.usage["total_tokens"] - rerank_before})
    trace.add_usage(answerer.model, out["usage"])

    costs = observe.cost_usd(trace.usage_by_model)
    # Citations carry NO corpus body — only the section path + the deep link.
    sources = [
        {"n": i, "section_path": r["section_path"],
         "page_title": r["page_title"], "url": r["url"]}
        for i, r in enumerate(out["sources"], 1)
    ]
    return {
        "question": question,
        "answer": out["answer"],
        "grounded": out["grounded"],
        "sources": sources,
        "trace": {
            "stages": [[name, round(secs * 1000, 1)] for name, secs in trace.spans],
            "total_ms": round(trace.total_seconds * 1000, 1),
            "cost_by_model": {
                model: {**usage, "usd": round(costs.get(model, 0.0), 6)}
                for model, usage in sorted(trace.usage_by_model.items())
            },
            "total_usd": round(costs["total"], 6),
        },
    }


def main() -> None:
    load_dotenv(ROOT)
    if not DB_FILE.exists():
        raise SystemExit(f"{DB_FILE} not found — build it first with `python -m rag index`.")
    embedder = VoyageEmbedder()
    reranker = VoyageReranker()
    answerer = ClaudeAnswerer()

    print(f"baking {len(QUESTIONS)} examples at k={K} (real pipeline) ...\n")
    examples = []
    for q in QUESTIONS:
        ex = bake_one(q, embedder, reranker, answerer)
        flag = "grounded" if ex["grounded"] else "ABSTAINED"
        print(f"  [{flag:9}] ${ex['trace']['total_usd']:.4f}  {q}")
        examples.append(ex)

    total = round(sum(e["trace"]["total_usd"] for e in examples), 4)
    payload = {
        "generated_at": date.today().isoformat(),
        "k": K,
        "generation_model": ANSWER_MODEL,
        "retrieval": "hybrid (dense + BM25, RRF) + Voyage rerank-2.5-lite",
        "note": ("Precomputed answers for the public demo. Contains model-written, "
                 "cited prose + source URLs + measured cost/latency only — no corpus "
                 "body. Re-bake with `python -m demo.bake`."),
        "bake_cost_usd": total,
        "examples": examples,
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {OUT_FILE.relative_to(ROOT)}  ({len(examples)} examples, "
          f"bake cost ${total:.4f})")


if __name__ == "__main__":
    main()
