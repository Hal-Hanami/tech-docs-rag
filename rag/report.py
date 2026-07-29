"""Rendering the eval results as text.

Split from `rag.eval` so that scoring and presentation can change independently:
adding a metric should not mean editing table-formatting code, and rewording a
column header should not risk touching arithmetic.

Everything here is pure — rows and a summary in, a string out — so the exact
output is assertable in tests rather than eyeballed.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import observe
from .eval import EvalRow


def _pct(x: float | None) -> str:
    """Percentage, or `n/a` for a metric that was never measured.

    None and 0.0 mean different things here — "we did not run generation" versus
    "it abstained every time" — so None must not be rendered as a number.
    """
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _mrr(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:6.3f}"


def _rank_cell(row: EvalRow) -> str:
    """Rank column: the 1-based hit position, MISS, or `-` for out-of-corpus.

    Out-of-corpus questions have no expected page, so a rank of 0 there would
    read as a retrieval failure when it is the intended state.
    """
    if not row.item.in_corpus:
        return "   -"
    return f"{row.rank:>4}" if row.rank > 0 else "MISS"


def _answered_cell(row: EvalRow) -> str:
    if row.grounded is None:
        return "    -"
    return "  yes" if row.grounded else "abstain"


def _faithful_cell(row: EvalRow) -> str:
    if row.faithful is None:
        return "    -"
    return "  yes" if row.faithful else "   NO"


def format_report(rows: Sequence[EvalRow], summary: dict[str, Any], *, k: int) -> str:
    """Render the per-question table followed by the headline metrics."""
    lines: list[str] = [
        f"{'id':<22} {'rank':>4} {'answered':>8} {'faithful':>8}  question",
        "-" * 78,
    ]
    for r in rows:
        q = r.item.question if len(r.item.question) <= 30 else r.item.question[:29] + "…"
        lines.append(f"{r.item.id:<22} {_rank_cell(r):>4} {_answered_cell(r):>8} "
                     f"{_faithful_cell(r):>8}  {q}")

    s = summary
    n_ic = s["n_in_corpus"]
    lines += [
        "",
        f"=== metrics (k={k}) ===",
        f"  retrieval recall@1    {_pct(s['recall_at_1'])}   ({n_ic} in-corpus questions)",
        f"  retrieval recall@3    {_pct(s['recall_at_3'])}",
    ]
    if k > 3:  # at k<=3 this would just repeat recall@3
        lines.append(f"  retrieval recall@{k:<3} {_pct(s['recall_at_k'])}")
    lines += [
        f"  retrieval MRR         {_mrr(s['mrr'])}   "
        f"(mean reciprocal rank of the first expected page)",
        f"  answer faithfulness   {_pct(s['faithfulness'])}   "
        f"({s['n_judged']} of {s['n_answered']} answered questions judged)",
        f"  abstention rate       {_pct(s['abstention_rate'])}   "
        f"({s['n_oob']} out-of-corpus questions)",
        f"  false abstentions     {s['n_false_abstentions']:>5}   "
        f"(in-corpus questions the model declined to answer)",
    ]
    if s["usage_by_model"]:
        lines += ["", "=== cost / latency ==="]
        lines += observe.format_cost_block(
            s["usage_by_model"], indent="  ",
            latencies=s["latencies"], n_queries=s["n_total"])
    return "\n".join(lines)
