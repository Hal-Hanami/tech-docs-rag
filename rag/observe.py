"""Cross-cutting: observability — tracing, token/cost accounting, latency. Design §6.

The cross-cutting concern: make a request explainable after
the fact (which stages ran, how long each took, how many tokens per model) and
put a dollar figure and a p50/p95 latency on it.

Two consumers, one vocabulary:
  - a single request (`python -m rag ask`) builds a `Trace` — per-stage spans +
    per-model token usage — so "why this answer, and what did it cost" is printable.
  - the eval harness (`python -m rag eval`) aggregates per-model usage and per-query
    latency across the whole set into a cost breakdown + percentiles.

Cost is split *by model* — Opus (generation) vs Haiku (judge) vs Voyage
(embeddings / rerank) — because that separation is the whole point of the
two-model design, and the lever for optimizing it. Token counts are
authoritative (returned by each API); USD is an estimate from the rate table
below. Everything here is pure/offline-testable — no key, no network.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Iterator, Sequence

# USD per 1M tokens. NOT memorized — confirmed from the source each release:
#   Claude:  `claude-api` skill pricing table (cached 2026-05-26).
#   Voyage:  https://docs.voyageai.com/docs/pricing (checked 2026-06-10).
# Re-check on model/price changes. Voyage bills one "total" token stream per call;
# Claude bills input and output separately (cache read ~0.1x / write 1.25x is not
# modeled — this pipeline sends a per-query-varying sources block, so the static
# prefix is too small to cache profitably).
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "voyage-4-lite": {"total": 0.02},
    "rerank-2.5-lite": {"total": 0.02},
    "rerank-2.5": {"total": 0.05},
}

# Maps a PRICING rate key to the usage dict key the APIs report.
_USAGE_KEY = {"input": "input_tokens", "output": "output_tokens", "total": "total_tokens"}


def merge_usage(into: dict[str, dict[str, int]], model: str, usage: dict[str, int]) -> None:
    """Accumulate one model's `{*_tokens: n}` usage into a by-model ledger (in place)."""
    bucket = into.setdefault(model, {})
    for key, val in usage.items():
        bucket[key] = bucket.get(key, 0) + val


def cost_usd(usage_by_model: dict[str, dict[str, int]]) -> dict[str, float]:
    """Per-model USD cost from a by-model token ledger, plus a `"total"` key.

    Models absent from PRICING contribute 0 and are skipped (e.g. the fake
    embedder in tests) — token counts stay authoritative even when a price isn't.
    """
    breakdown: dict[str, float] = {}
    total = 0.0
    for model, usage in usage_by_model.items():
        rates = PRICING.get(model)
        if rates is None:
            continue
        c = sum(usage.get(_USAGE_KEY[kind], 0) / 1_000_000 * rate
                for kind, rate in rates.items())
        if c:
            breakdown[model] = c
            total += c
    breakdown["total"] = total
    return breakdown


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated p-th percentile (p in [0,100]); None for empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


@dataclass
class Trace:
    """One request's observability record: ordered stage spans + per-model usage.

    `span(name)` times a stage; `add_usage(model, usage)` files token counts under
    the model that produced them. Stages are sequential, so `total_seconds` (their
    sum) is the traced wall-clock.
    """

    spans: list[tuple[str, float]] = field(default_factory=list)
    usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append((name, time.perf_counter() - t0))

    def add_usage(self, model: str, usage: dict[str, int]) -> None:
        if usage:
            merge_usage(self.usage_by_model, model, usage)

    @property
    def total_seconds(self) -> float:
        return sum(s for _, s in self.spans)


def span(trace: Trace | None, name: str):
    """`with span(trace, name):` — times the block if a Trace is given, else no-op.

    Lets `search`/`generate` carry one optional `trace` param without sprouting
    `if trace is not None` around every stage.
    """
    return trace.span(name) if trace is not None else nullcontext()


def format_cost_block(usage_by_model: dict[str, dict[str, int]], *, indent: str = "  ",
                      latencies: Sequence[float] | None = None,
                      n_queries: int | None = None) -> list[str]:
    """Render the per-model token/cost table (+ optional latency percentiles).

    Shared by the `ask` trace and the `eval` report so both speak the same units.
    """
    lines: list[str] = []
    costs = cost_usd(usage_by_model)
    for model in sorted(usage_by_model):
        u = usage_by_model[model]
        toks = ", ".join(f"{k.replace('_tokens', '')}={v}" for k, v in u.items())
        usd = costs.get(model)
        money = f"  ${usd:.4f}" if usd is not None else ""
        lines.append(f"{indent}{model:<18} {toks}{money}")
    total = costs["total"]
    if n_queries:
        lines.append(f"{indent}{'TOTAL':<18} ${total:.4f}  (${total / n_queries:.4f}/query over {n_queries})")
    else:
        lines.append(f"{indent}{'TOTAL':<18} ${total:.4f}")
    if latencies:
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        lines.append(f"{indent}{'latency':<18} p50={p50:.2f}s  p95={p95:.2f}s  "
                     f"(end-to-end per query, n={len(latencies)})")
    return lines


def format_trace(trace: Trace) -> list[str]:
    """Render one request's spans + cost — the `ask` observability footer."""
    lines = ["--- trace ---"]
    for name, secs in trace.spans:
        lines.append(f"  {name:<10} {secs * 1000:7.1f} ms")
    lines.append(f"  {'total':<10} {trace.total_seconds * 1000:7.1f} ms")
    if trace.usage_by_model:
        lines.append("  cost:")
        lines.extend(format_cost_block(trace.usage_by_model, indent="    "))
    return lines
