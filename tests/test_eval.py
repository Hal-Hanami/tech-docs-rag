"""Offline tests for the M4 eval harness.

No network, no keys: scripted fakes stand in for both LLM seams (the Opus answerer
and the Haiku judge), so the retrieval -> generate -> judge -> aggregate loop and
the metric math are all asserted without a real model call. (The real
claude-opus-4-8 + claude-haiku-4-5 path runs via `python -m rag eval`.)
"""

from __future__ import annotations

import json
from pathlib import Path

from rag import eval as eval_mod
from rag import generate as gen
from rag import index as index_mod
from rag.clients.claude import ANSWER_MODEL, JUDGE_MODEL, build_judge_user, parse_verdict
from rag.ports import Completion, Verdict

from test_store import DOCS, FakeEmbedder  # reuse the M2 corpus fixtures


class ScriptedAnswerer:
    """Returns a routed reply when a needle appears in the user message, else IDK
    — lets one fake produce both grounded answers and abstentions across items."""

    model = ANSWER_MODEL  # stands in for the real answerer, so pricing is exercised too

    def __init__(self, routes: list[tuple[str, str]]) -> None:
        self.routes = routes

    def complete(self, system: str, user: str) -> Completion:
        question = user.split("\n", 1)[0]  # the "Question: ..." line, not the sources
        for needle, reply in self.routes:
            if needle in question:
                return Completion(text=reply, usage={"input_tokens": 3, "output_tokens": 2})
        return Completion(text=gen.IDK, usage={"input_tokens": 3, "output_tokens": 0})


class ScriptedJudge:
    """Faithful unless the answer contains a flagged needle (e.g. a fabricated claim)."""

    model = JUDGE_MODEL  # stands in for the real judge, keeping its spend in its own bucket

    def __init__(self, unfaithful_needles: tuple[str, ...] = ()) -> None:
        self.unfaithful_needles = unfaithful_needles

    def judge(self, question: str, answer: str, sources_block: str) -> Verdict:
        bad = any(n in answer for n in self.unfaithful_needles)
        return Verdict(
            faithful=not bad,
            unsupported_claims=["fabricated"] if bad else [],
            usage={"input_tokens": 4, "output_tokens": 1},
        )


def _build(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_mod.build(chunks, db, FakeEmbedder())
    return db


# --- pure helpers -----------------------------------------------------------

def test_recall_hit_matches_expected_page():
    item = eval_mod.EvalItem("i", "q", ["u/pc", "u/tu"], in_corpus=True)
    assert eval_mod.recall_hit(item, ["u/vs", "u/tu"]) is True
    assert eval_mod.recall_hit(item, ["u/vs"]) is False
    # an abstention item has no expected page → never a recall hit
    oob = eval_mod.EvalItem("o", "q", [], in_corpus=False)
    assert eval_mod.recall_hit(oob, ["u/vs"]) is False


def test_first_hit_rank_is_one_based_position_of_first_expected():
    item = eval_mod.EvalItem("i", "q", ["u/pc", "u/tu"], in_corpus=True)
    assert eval_mod.first_hit_rank(item, ["u/vs", "u/tu", "u/pc"]) == 2  # u/tu first
    assert eval_mod.first_hit_rank(item, ["u/pc"]) == 1
    assert eval_mod.first_hit_rank(item, ["u/vs", "u/x"]) == 0           # not retrieved


def test_parse_verdict_reads_json():
    v = parse_verdict('{"faithful": false, "unsupported_claims": ["x", "y"]}')
    assert v.faithful is False
    assert v.unsupported_claims == ["x", "y"]


def test_build_judge_user_carries_all_three_parts():
    msg = build_judge_user("Q?", "A [1].", "[1] (Path)\nbody")
    assert "QUESTION:\nQ?" in msg
    assert "ANSWER:\nA [1]." in msg
    assert "SOURCES:\n[1] (Path)" in msg


def test_load_items_skips_comments_and_filters_by_tag(tmp_path):
    f = tmp_path / "qa.jsonl"
    f.write_text(
        "# a comment\n"
        '{"id": "a", "question": "q1", "expected_source_urls": ["u/pc"], "in_corpus": true}\n'
        "\n"
        '{"id": "b", "question": "q2", "expected_source_urls": [], "in_corpus": false}\n'
        '{"id": "h", "question": "q3", "expected_source_urls": ["u/x"], "in_corpus": true, "tag": "hard"}\n',
        encoding="utf-8",
    )
    items = eval_mod.load_items(f)
    assert [i.id for i in items] == ["a", "b", "h"]
    assert items[0].expected_source_urls == ["u/pc"]
    assert items[1].in_corpus is False
    assert items[0].tag == "core"        # default when absent
    assert items[2].tag == "hard"
    # --tag filters to a single slice
    assert [i.id for i in eval_mod.load_items(f, tag="hard")] == ["h"]
    assert [i.id for i in eval_mod.load_items(f, tag="core")] == ["a", "b"]


# --- aggregation ------------------------------------------------------------

def test_summarize_computes_rank_and_llm_metrics():
    # rank is the trailing field; it drives recall@1/@3/@k + MRR (what rerank moves).
    rows = [
        # in-corpus, hit at rank 1, answered + faithful
        eval_mod.EvalRow(eval_mod.EvalItem("a", "", ["u/pc"], True), ["u/pc"], True, True, True, "", {}, rank=1),
        # in-corpus, hit at rank 1, answered but NOT faithful
        eval_mod.EvalRow(eval_mod.EvalItem("b", "", ["u/tu"], True), ["u/tu"], True, True, False, "", {}, rank=1),
        # in-corpus, hit only at rank 4 → counts for recall@k but not recall@3; abstained
        eval_mod.EvalRow(eval_mod.EvalItem("c", "", ["u/vs"], True), ["u/x", "u/y", "u/z", "u/vs"], True, False, None, "", {}, rank=4),
        # in-corpus, retrieval missed (rank 0)
        eval_mod.EvalRow(eval_mod.EvalItem("d", "", ["u/x"], True), ["u/pc"], False, True, True, "", {}, rank=0),
        # out-of-corpus, correctly abstained
        eval_mod.EvalRow(eval_mod.EvalItem("e", "", [], False), ["u/pc"], False, False, None, "", {}, rank=0),
    ]
    s = eval_mod.summarize(rows)
    assert s["n_in_corpus"] == 4 and s["n_oob"] == 1
    assert s["recall_at_1"] == 0.5             # a,b at rank 1
    assert s["recall_at_3"] == 0.5             # a,b ≤3; c is rank 4
    assert s["recall_at_k"] == 0.75            # a,b,c retrieved; d missed
    assert s["mrr"] == (1 + 1 + 0.25 + 0) / 4  # 0.5625
    assert s["n_answered"] == 3                # a,b,d
    assert s["faithfulness"] == 2 / 3          # a,d faithful; b not
    assert s["n_false_abstentions"] == 1       # c
    assert s["abstention_rate"] == 1.0         # e abstained


def test_retrieval_only_summary_leaves_llm_metrics_na():
    rows = [
        eval_mod.EvalRow(eval_mod.EvalItem("a", "", ["u/pc"], True), ["u/pc"], True, None, None, "", {}, rank=1),
        eval_mod.EvalRow(eval_mod.EvalItem("e", "", [], False), ["u/pc"], False, None, None, "", {}, rank=0),
    ]
    s = eval_mod.summarize(rows)
    assert s["recall_at_k"] == 1.0             # recall/MRR still measurable
    assert s["mrr"] == 1.0
    assert s["faithfulness"] is None           # nothing judged
    assert s["abstention_rate"] is None        # generation never ran


# --- end-to-end loop with fakes --------------------------------------------

def test_evaluate_full_loop(tmp_path):
    db = _build(tmp_path)
    items = [
        eval_mod.EvalItem("cache", "how do I cache the prompt prefix?", ["u/pc"], True),
        eval_mod.EvalItem("tool", "call a tool with a schema", ["u/tu"], True),
        eval_mod.EvalItem("abstain-in", "what does vision do here?", ["u/vs"], True),
        eval_mod.EvalItem("oob", "airspeed of an unladen swallow?", [], False),
    ]
    answerer = ScriptedAnswerer([
        ("cache", "Caching reuses the prompt prefix [1]."),
        ("tool", "Define a tool with a schema [1]. It can also teleport [1]."),
    ])
    judge = ScriptedJudge(unfaithful_needles=("teleport",))

    rows = eval_mod.evaluate(items, db, FakeEmbedder(), answerer, judge, k=3)
    by_id = {r.item.id: r for r in rows}

    assert by_id["cache"].recall_hit and by_id["cache"].grounded and by_id["cache"].faithful is True
    assert by_id["tool"].recall_hit and by_id["tool"].grounded and by_id["tool"].faithful is False
    assert by_id["abstain-in"].grounded is False and by_id["abstain-in"].faithful is None
    assert by_id["oob"].grounded is False and by_id["oob"].recall_hit is False

    s = eval_mod.summarize(rows)
    assert s["recall_at_k"] == 1.0             # all 3 in-corpus pages retrieved
    assert s["mrr"] == 1.0                     # each at rank 1
    assert s["faithfulness"] == 0.5            # cache faithful, tool not
    assert s["n_false_abstentions"] == 1       # abstain-in
    assert s["abstention_rate"] == 1.0         # oob declined
    # Usage is filed per model rather than summed into one bucket, which is what
    # makes the generation/judging cost split visible; cost comes from PRICING.
    assert s["usage_by_model"][ANSWER_MODEL]["input_tokens"] > 0
    assert s["usage_by_model"][JUDGE_MODEL]["input_tokens"] > 0
    assert s["cost"]["total"] > 0
    assert len(s["latencies"]) == len(rows)    # every question is timed


def test_evaluate_retrieval_only_makes_no_llm_calls(tmp_path):
    db = _build(tmp_path)
    items = [eval_mod.EvalItem("cache", "how do I cache the prompt prefix?", ["u/pc"], True)]
    rows = eval_mod.evaluate(items, db, FakeEmbedder(), answerer=None, judge=None, k=3)
    assert rows[0].recall_hit is True
    assert rows[0].grounded is None and rows[0].faithful is None
    # no LLM and a fake embedder with no usage meter → nothing billable recorded
    assert rows[0].usage_by_model == {}
