"""Tests for eval report rendering.

The property worth protecting here is that "not measured" and "measured as zero"
never render the same way. A retrieval-only run does not generate anything, so
its abstention rate is unknown — printing `0.0%` there would claim the system
failed to decline a single out-of-corpus question, which is the opposite of what
happened.
"""

from __future__ import annotations

from rag import report
from rag.eval import EvalItem, EvalRow, summarize


def _row(item: EvalItem, *, rank: int = 1, grounded: bool | None = None,
         faithful: bool | None = None, latency: float = 0.01) -> EvalRow:
    return EvalRow(
        item=item, retrieved_urls=["u/x"], recall_hit=rank > 0, grounded=grounded,
        faithful=faithful, answer="", usage_by_model={}, rank=rank, latency_s=latency,
    )


IN_CORPUS = EvalItem("q1", "how do I cache a prompt?", ["u/x"], True)
OUT_OF_CORPUS = EvalItem("oob", "how do I bake sourdough?", [], False)


def test_unmeasured_metrics_render_as_na_not_zero():
    rows = [_row(IN_CORPUS)]  # retrieval-only: grounded/faithful stay None
    text = report.format_report(rows, summarize(rows), k=5)

    def line(label: str) -> str:
        return next(ln for ln in text.splitlines() if label in ln)

    # Unmeasured metrics carry no percentage at all — not even 0.0%.
    assert "n/a" in line("answer faithfulness") and "%" not in line("answer faithfulness")
    assert "n/a" in line("abstention rate") and "%" not in line("abstention rate")
    # Retrieval *was* measured, so it must still render as a number.
    assert "%" in line("retrieval recall@1")


def test_out_of_corpus_row_shows_a_dash_rather_than_a_rank():
    """An out-of-corpus question has no expected page, so rank 0 there is the
    intended state, not a retrieval miss."""
    rows = [_row(OUT_OF_CORPUS, rank=0, grounded=False)]
    text = report.format_report(rows, summarize(rows), k=5)
    line = next(ln for ln in text.splitlines() if ln.startswith("oob"))
    assert "MISS" not in line
    assert "abstain" in line


def test_missed_in_corpus_question_is_marked_miss():
    rows = [_row(IN_CORPUS, rank=0)]
    text = report.format_report(rows, summarize(rows), k=5)
    assert "MISS" in next(ln for ln in text.splitlines() if ln.startswith("q1"))


def test_recall_at_k_line_is_suppressed_when_it_would_repeat_recall_at_3():
    rows = [_row(IN_CORPUS)]
    at_five = report.format_report(rows, summarize(rows), k=5)
    at_three = report.format_report(rows, summarize(rows), k=3)
    assert "recall@5" in at_five
    assert "recall@3" in at_three
    assert "recall@3  " in at_three and "recall@1" in at_three
    # k=3 would make recall@k identical to recall@3; printing both invites the
    # reader to think two different things were measured.
    assert at_three.count("recall@") == 2


def test_long_questions_are_truncated_so_the_table_stays_aligned():
    long_item = EvalItem("q1", "x" * 200, ["u/x"], True)
    text = report.format_report([_row(long_item)], summarize([_row(long_item)]), k=5)
    line = next(ln for ln in text.splitlines() if ln.startswith("q1"))
    assert "…" in line
    assert len(line) < 100


def test_cost_block_appears_only_when_something_was_spent():
    """A retrieval-only run against fakes records no usage; printing an empty
    cost table would suggest the run was free rather than unmeasured."""
    rows = [_row(IN_CORPUS)]
    assert "cost / latency" not in report.format_report(rows, summarize(rows), k=5)

    priced = _row(IN_CORPUS)
    priced.usage_by_model = {"claude-haiku-4-5": {"input_tokens": 100, "output_tokens": 10}}
    text = report.format_report([priced], summarize([priced]), k=5)
    assert "cost / latency" in text
    assert "claude-haiku-4-5" in text
