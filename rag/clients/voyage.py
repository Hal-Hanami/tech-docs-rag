"""Voyage AI adapters — embeddings (`Embedder`) and reranking (`Reranker`).

Anthropic ships no first-party embedding model and its docs recommend Voyage, so
both retrieval-side models come from the same vendor and share one API key. They
also share this module's `_post` helper: the two endpoints differ only in payload
and response shape, and duplicating retry/usage bookkeeping across two files was
how the two copies drifted apart in the first place.

Called over `urllib` rather than an SDK on purpose — these are two JSON POSTs, and
avoiding the dependency keeps the runtime install to `sqlite-vec` alone. The cost
is that retry and error handling are ours to write, which is what `_post` is.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

EMBED_URL = "https://api.voyageai.com/v1/embeddings"
RERANK_URL = "https://api.voyageai.com/v1/rerank"

DEFAULT_EMBED_MODEL = "voyage-4-lite"   # ample at ~1.3k chunks; voyage-4 is the upgrade lever
DEFAULT_EMBED_DIM = 1024                # voyage-4-lite default; re-derived from the API at index time
DEFAULT_RERANK_MODEL = "rerank-2.5-lite"  # mirrors the lite embedder; rerank-2.5 is the upgrade lever

# Texts per embedding request. ~100 x ~484 tokens is comfortably inside Voyage's
# per-request cap, and larger batches mean fewer round trips when indexing.
BATCH = 100

# 429 and 5xx are worth retrying; a 400 means our payload is wrong and retrying
# it just burns time producing the same error.
_RETRYABLE = {429, 500, 502, 503, 529}


def _require_key(api_key: str | None, *, disable_hint: str = "") -> str:
    """Resolve the Voyage key or exit with an actionable message.

    Exits rather than raising because every caller is the CLI, and a stack trace
    for "you have not set a key yet" is noise rather than information.
    """
    key = api_key or os.environ.get("VOYAGE_API_KEY")
    if key:
        return key
    raise SystemExit(
        "VOYAGE_API_KEY not set.\n"
        "  Embeddings and reranking share one Voyage key.\n"
        "  Put  VOYAGE_API_KEY=...  in .env  (gitignored), or export it."
        + disable_hint
    )


def _post(url: str, payload: dict[str, Any], key: str, *, retries: int = 4) -> dict[str, Any]:
    """POST JSON, retrying transient failures with exponential backoff.

    Returns the decoded response. Raises `SystemExit` with the server's own
    message on a non-retryable error — the API's explanation of what it disliked
    is more useful to the reader than anything we could paraphrase.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code in _RETRYABLE and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            detail = e.read().decode("utf-8", "replace")[:300]
            raise SystemExit(f"Voyage API error {e.code}: {detail}")
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"Voyage API connection error: {e}")
    raise SystemExit("Voyage API: exhausted retries")  # unreachable; keeps type-checkers happy


class VoyageEmbedder:
    """Batched embedding client. Implements `rag.ports.Embedder`.

    Accumulates `usage["total_tokens"]` across calls so a caller can snapshot it
    before and after a query and attribute that query's embedding spend, which is
    how the eval harness separates embedding cost from LLM cost.
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dim: int = DEFAULT_EMBED_DIM,
                 api_key: str | None = None) -> None:
        self.model = model
        self.dim = dim
        self.usage = {"total_tokens": 0}
        self._key = _require_key(api_key)

    def embed(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            out.extend(self._embed_batch(list(texts[start:start + BATCH]), input_type))
        return out

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        data = _post(EMBED_URL,
                     {"input": texts, "model": self.model, "input_type": input_type},
                     self._key)
        self.usage["total_tokens"] += data.get("usage", {}).get("total_tokens", 0)
        # Sort by the API's own index rather than trusting response order: the
        # embedding at position i must line up with texts[i] or the whole index
        # is silently wrong, and that failure is invisible until recall drops.
        rows = sorted(data["data"], key=lambda d: d["index"])
        return [r["embedding"] for r in rows]


class VoyageReranker:
    """Cross-encoder reranking client. Implements `rag.ports.Reranker`."""

    def __init__(self, model: str = DEFAULT_RERANK_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.usage = {"total_tokens": 0}
        self._key = _require_key(
            api_key, disable_hint="\n  Or disable reranking with `--no-rerank`.")

    def rerank(self, query: str, documents: Sequence[str], *,
               top_k: int | None = None) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload: dict[str, Any] = {
            "query": query, "documents": list(documents), "model": self.model,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        data = _post(RERANK_URL, payload, self._key)
        self.usage["total_tokens"] += data.get("usage", {}).get("total_tokens", 0)
        ranked = [(int(d["index"]), float(d["relevance_score"])) for d in data["data"]]
        # The API returns these already sorted; sorting again costs nothing and
        # removes a silent dependency on undocumented response ordering.
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked
