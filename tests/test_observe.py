"""Offline tests for the M6 observability layer.

Cost from a per-model token ledger, latency percentiles, usage merging, and the
request Trace — all pure / no key / no network. The live cost numbers come out of
`python -m rag eval` and `python -m rag ask`; here we pin the arithmetic.
"""

from __future__ import annotations

from rag import observe


def test_cost_usd_splits_by_model_and_totals():
    usage = {
        "claude-opus-4-8": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        "claude-haiku-4-5": {"input_tokens": 1_000_000, "output_tokens": 0},
        "voyage-4-lite": {"total_tokens": 1_000_000},
    }
    cost = observe.cost_usd(usage)
    assert cost["claude-opus-4-8"] == 5.0 + 25.0    # $5/1M in + $25/1M out
    assert cost["claude-haiku-4-5"] == 1.0          # $1/1M in, no output
    assert cost["voyage-4-lite"] == 0.02            # $0.02/1M total
    assert cost["total"] == 30.0 + 1.0 + 0.02


def test_cost_usd_skips_unpriced_models():
    # an unknown/fake model contributes 0 and is absent from the breakdown, so a
    # fake embedder in tests never breaks costing — token counts stay authoritative.
    cost = observe.cost_usd({"fake-bow": {"total_tokens": 9_999}})
    assert "fake-bow" not in cost
    assert cost["total"] == 0.0


def test_percentile_interpolates_and_handles_edges():
    assert observe.percentile([], 50) is None
    assert observe.percentile([4.2], 95) == 4.2          # single value
    data = [1, 2, 3, 4]
    assert observe.percentile(data, 0) == 1
    assert observe.percentile(data, 100) == 4
    assert observe.percentile(data, 50) == 2.5           # midpoint of 4 points


def test_merge_usage_accumulates_per_model():
    into: dict[str, dict[str, int]] = {}
    observe.merge_usage(into, "m", {"input_tokens": 3})
    observe.merge_usage(into, "m", {"input_tokens": 4, "output_tokens": 1})
    assert into == {"m": {"input_tokens": 7, "output_tokens": 1}}


def test_trace_records_spans_and_usage():
    trace = observe.Trace()
    with trace.span("embed"):
        pass
    with trace.span("generate"):
        pass
    trace.add_usage("claude-opus-4-8", {"input_tokens": 10, "output_tokens": 2})
    trace.add_usage("claude-opus-4-8", {"input_tokens": 5})
    trace.add_usage("claude-opus-4-8", {})               # empty usage is a no-op
    assert [name for name, _ in trace.spans] == ["embed", "generate"]
    assert trace.total_seconds >= 0
    assert trace.usage_by_model["claude-opus-4-8"] == {"input_tokens": 15, "output_tokens": 2}


def test_span_helper_is_noop_without_trace():
    # search()/generate() pass trace=None on retrieval-only / offline paths
    with observe.span(None, "x"):
        pass  # must not raise
