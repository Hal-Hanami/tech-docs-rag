"""The seams where this system meets the outside world.

Four Protocols, and nothing else. Everything else under `rag/` is pure policy —
retrieval strategy, prompt assembly, scoring, reporting — and depends only on
these interfaces. Everything that actually opens a socket (Voyage for
embeddings and reranking, Anthropic for generation and judging) implements one
of them and lives in `rag/clients/`.

The dependency points inward: policy declares the interface it needs, and the
clients conform to it. That inversion is what lets the whole
retrieval -> generate -> judge -> aggregate path run under test with no API key
and no network — the fakes in `tests/` are simply other implementations of the
same four Protocols, and the policy code cannot tell the difference.

If you are looking for "where would I plug in a different vector store / a
different LLM / a local embedding model", the answer is: implement the matching
Protocol here and pass it in. No policy module needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


class Embedder(Protocol):
    """Turns text into vectors.

    `input_type` is not decoration: retrieval quality depends on passing
    `"document"` when indexing and `"query"` when searching, because providers
    prepend different instructions for each. An implementation that ignores it
    will still work and will quietly retrieve worse.
    """

    model: str
    dim: int

    def embed(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]: ...


class Reranker(Protocol):
    """Re-scores (query, documents) and returns (original_index, score), best first.

    Returning indices rather than the documents themselves keeps this interface
    ignorant of what a "document" is — the caller holds the metadata and uses
    the indices to reorder its own list.
    """

    model: str

    def rerank(self, query: str, documents: Sequence[str], *,
               top_k: int | None = None) -> list[tuple[int, float]]: ...


@dataclass
class Completion:
    """What an `Answerer` returns: the text, plus whatever token usage it reported.

    `usage` is empty for implementations that don't meter (the test fakes), which
    is why every consumer treats it as optional rather than assuming a cost.
    """

    text: str
    usage: dict[str, int] = field(default_factory=dict)


class Answerer(Protocol):
    """Generates an answer from a system prompt and a user message.

    Deliberately narrower than any real LLM client: no tools, no streaming, no
    model selection. The prompt is assembled by policy (`rag.generate`) and this
    seam only carries it across the network.
    """

    model: str

    def complete(self, system: str, user: str) -> Completion: ...


@dataclass
class Verdict:
    """A faithfulness ruling on one answer.

    `unsupported_claims` is the useful half — a bare boolean tells you the score
    moved but not what to fix, so the judge is asked to name the claims it could
    not find support for.
    """

    faithful: bool
    unsupported_claims: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class Judge(Protocol):
    """Rules on whether an answer is supported by the sources it was given.

    Separate from `Answerer` because the two are deliberately different models:
    generation wants the strongest model available, judging wants the cheapest
    one that can read. Keeping them as distinct seams is what makes that split
    visible in the cost breakdown instead of hidden inside one client.
    """

    model: str

    def judge(self, question: str, answer: str, sources_block: str) -> Verdict: ...
