"""Grounded answer generation — the hallucination-suppression policy.

The rule this module enforces: the model sees the retrieved sources and nothing
else, cites a source number for every claim, and when the sources do not support
an answer it says so in one exact sentence rather than guessing.

That last part is why `IDK` is a constant compared with `==` rather than a
fuzzy check. Abstention has to be a *detectable* outcome — the eval harness
counts abstentions and false abstentions, and a substring or case-insensitive
match would let a hedged answer ("I don't know if that's right, but...") be
scored as a clean decline.

No network here. The `Answerer` seam does the talking, so prompt assembly and
grounding logic are exercised in tests with a fake and no key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import observe
from . import search as search_mod
from .observe import Trace
from .ports import Answerer, Embedder, Reranker

# The exact decline sentence. Both the prompt and the grounded/abstained check
# key off this one string, so they cannot drift apart.
IDK = "I don't know based on the provided documentation."

SYSTEM = (
    "You are a precise assistant for the Claude developer documentation. "
    "Answer the question using ONLY the numbered sources given in the user "
    "message.\n"
    "Rules:\n"
    f'- If the sources do not contain enough information to answer, reply with '
    f'exactly "{IDK}" and nothing else. Do not guess.\n'
    "- Ground every claim in the sources. Put the supporting source number(s) in "
    "square brackets right after each claim, e.g. [1] or [2][3].\n"
    "- Use only facts present in the sources. Never invent API names, parameters, "
    "values, or URLs that are not in the sources.\n"
    "- Be concise and direct."
)


def build_sources_block(results: Sequence[dict[str, Any]]) -> str:
    """Render retrieved chunks as `[n] (section_path)` blocks.

    The URL is deliberately withheld from the model and re-attached to each
    number at print time. A model that never sees a link cannot fabricate one,
    and citation integrity is the property this project is selling.
    """
    blocks = []
    for i, r in enumerate(results, 1):
        text = " ".join(r["text"].split())  # collapse whitespace; keep the prompt compact
        blocks.append(f"[{i}] ({r['section_path']})\n{text}")
    return "\n\n".join(blocks)


def build_user_message(query: str, results: Sequence[dict[str, Any]]) -> str:
    return f"Question: {query}\n\nSources:\n{build_sources_block(results)}"


def answer_from_results(query: str, results: Sequence[dict[str, Any]],
                        answerer: Answerer, *, trace: Trace | None = None) -> dict[str, Any]:
    """Answer `query` from already-retrieved `results`.

    Split from `answer()` so a caller that has already retrieved — the eval
    harness scores recall on the same hits it grounds on — does not pay for a
    second retrieval, and so recall and generation are measured against
    identical evidence rather than two separate searches.

    Empty results short-circuit to a decline: there is nothing to ground on, and
    asking the model anyway invites it to answer from memory.
    """
    if not results:
        return {"answer": IDK, "sources": [], "grounded": False, "usage": {}}
    with observe.span(trace, "generate"):
        completion = answerer.complete(SYSTEM, build_user_message(query, results))
    text = completion.text.strip()
    return {
        "answer": text,
        "sources": list(results),
        "grounded": text != IDK,
        "usage": completion.usage,
    }


def answer(query: str, db_path: Path, embedder: Embedder, answerer: Answerer,
           *, k: int = 5, hybrid: bool = True, reranker: Reranker | None = None,
           trace: Trace | None = None) -> dict[str, Any]:
    """Retrieve top-k, then ground an answer in exactly those chunks."""
    results = search_mod.search(query, db_path, embedder, k=k,
                                hybrid=hybrid, reranker=reranker, trace=trace)
    return answer_from_results(query, results, answerer, trace=trace)
