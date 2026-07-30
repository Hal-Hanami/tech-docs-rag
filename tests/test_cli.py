"""Offline tests for the flag surface and the orchestration it drives.

These two layers hold no pipeline logic, which is exactly why they were the last
place a bug hid: `--dense-only` was documented as the un-reranked baseline while
the code still passed a reranker, so a reproduction command in the README
reported a configuration nobody had run. Nothing failed, because the retrieval
tests call `search()` directly and never go through a flag.

So what is pinned here is the translation itself — one parsed namespace to one
call, and the ablation flags staying independent — plus the credential loading
that happens before any of it. No network, no key, no index.
"""

from __future__ import annotations

import argparse

import pytest

from rag import cli, commands, config


# --- the ablation flags are independent -----------------------------------------

@pytest.mark.parametrize("dense_only,no_rerank,expected", [
    (False, False, "hybrid (dense+BM25) + rerank"),
    (True,  False, "dense + rerank"),            # --dense-only alone still reranks
    (False, True,  "hybrid (dense+BM25)"),
    (True,  True,  "dense"),                     # both flags = the bare baseline
])
def test_mode_banner_names_every_combination(dense_only, no_rerank, expected):
    assert commands._mode(dense_only, no_rerank) == expected


def test_reranker_is_present_unless_switched_off(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(commands, "VoyageReranker", lambda *a, **kw: sentinel)
    assert commands._reranker(no_rerank=False) is sentinel
    assert commands._reranker(no_rerank=True) is None


# --- flags in, one call out ------------------------------------------------------

@pytest.fixture
def calls(monkeypatch):
    """Capture what the CLI asks `commands` to do, without doing it."""
    seen: dict[str, dict] = {}
    for name in ("run_index", "run_query", "run_ask", "run_eval"):
        def rec(*args, _n=name, **kw):
            seen["cmd"] = _n
            seen["args"] = args
            seen["kw"] = kw
        monkeypatch.setattr(commands, name, rec)
    return seen


def drive(calls, argv):
    args = cli.build_parser().parse_args(argv)
    args.run(args)
    return calls


def test_index_passes_the_model_and_limit(calls):
    c = drive(calls, ["index", "--limit", "10", "--model", "voyage-4"])
    assert c["cmd"] == "run_index" and c["kw"] == {"model": "voyage-4", "limit": 10}


def test_query_forwards_the_text_and_retrieval_flags(calls):
    c = drive(calls, ["query", "how do I cache a prompt?", "-k", "3", "--dense-only"])
    assert c["cmd"] == "run_query" and c["args"] == ("how do I cache a prompt?",)
    assert c["kw"]["k"] == 3
    assert c["kw"]["dense_only"] is True and c["kw"]["no_rerank"] is False


def test_ask_forwards_the_generation_cap(calls):
    c = drive(calls, ["ask", "q", "--max-tokens", "512", "--no-rerank"])
    assert c["cmd"] == "run_ask" and c["kw"]["max_tokens"] == 512
    assert c["kw"]["no_rerank"] is True and c["kw"]["dense_only"] is False


def test_eval_forwards_every_scoring_flag(calls):
    c = drive(calls, ["eval", "-k", "3", "--tag", "hard", "--limit", "5",
                      "--retrieval-only", "--no-judge", "--dense-only", "--no-rerank"])
    assert c["cmd"] == "run_eval"
    assert c["kw"] == {"k": 3, "tag": "hard", "limit": 5, "retrieval_only": True,
                       "no_judge": True, "max_tokens": 4096,
                       "dense_only": True, "no_rerank": True}


def test_eval_defaults_reproduce_the_headline_configuration(calls):
    """The bare command is the one the README quotes numbers for."""
    c = drive(calls, ["eval"])
    assert c["kw"] == {"k": 5, "tag": "", "limit": 0, "retrieval_only": False,
                       "no_judge": False, "max_tokens": 4096,
                       "dense_only": False, "no_rerank": False}


def test_dense_only_does_not_imply_no_rerank(calls):
    """The regression that produced a wrong reproduction command in the README."""
    c = drive(calls, ["eval", "--dense-only"])
    assert c["kw"]["dense_only"] is True
    assert c["kw"]["no_rerank"] is False


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_retrieval_flags_are_offered_on_every_retrieving_subcommand():
    parser = cli.build_parser()
    sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    for name in ("query", "ask", "eval"):
        flags = {o for a in sub.choices[name]._actions for o in a.option_strings}
        assert {"--dense-only", "--no-rerank"} <= flags, name


# --- credentials ------------------------------------------------------------------

def test_load_dotenv_reads_key_value_lines(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        '# a comment\n\nVOYAGE_API_KEY=abc\nANTHROPIC_API_KEY="quoted"\n'
        "SPACED = padded \nnot_a_pair\n", encoding="utf-8")
    for k in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY", "SPACED"):
        monkeypatch.delenv(k, raising=False)
    config.load_dotenv(tmp_path)
    import os
    assert os.environ["VOYAGE_API_KEY"] == "abc"
    assert os.environ["ANTHROPIC_API_KEY"] == "quoted"   # quotes stripped
    assert os.environ["SPACED"] == "padded"              # whitespace stripped


def test_an_exported_key_beats_the_file(tmp_path, monkeypatch):
    """Otherwise a stale .env silently overrides the key you just exported."""
    (tmp_path / ".env").write_text("VOYAGE_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("VOYAGE_API_KEY", "from_shell")
    config.load_dotenv(tmp_path)
    import os
    assert os.environ["VOYAGE_API_KEY"] == "from_shell"


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    config.load_dotenv(tmp_path)   # no .env here; keys may come from the shell


# --- run_eval: the function that produces the published numbers -------------------
# Everything above stops at the boundary of `commands`. This drives `run_eval`
# itself with stand-ins for the two paid clients, so the wiring that decides
# *what gets measured* -- hybrid on/off, reranker present, which questions, and
# whether the paid stages run at all -- is checked without paying for it.

class _Recorder:
    """Captures the arguments `evaluate` was called with."""

    def __init__(self):
        self.kw = None
        self.items = None

    def evaluate(self, items, db, embedder, answerer, judge, **kw):
        self.items = items
        self.kw = dict(kw, answerer=answerer, judge=judge, embedder=embedder)
        return []


@pytest.fixture
def eval_harness(monkeypatch, tmp_path):
    rec = _Recorder()
    items = [{"q": f"q{i}", "tag": "core"} for i in range(4)]

    monkeypatch.setattr(commands, "VoyageEmbedder", lambda *a, **kw: "EMBEDDER")
    monkeypatch.setattr(commands, "VoyageReranker", lambda *a, **kw: "RERANKER")
    monkeypatch.setattr(commands, "ClaudeAnswerer", lambda **kw: f"ANSWERER({kw})")
    monkeypatch.setattr(commands, "ClaudeJudge", lambda *a, **kw: "JUDGE")
    monkeypatch.setattr(commands.eval_mod, "load_items", lambda path, tag="": list(items))
    monkeypatch.setattr(commands.eval_mod, "evaluate", rec.evaluate)
    monkeypatch.setattr(commands.eval_mod, "summarize", lambda rows: {})
    monkeypatch.setattr(commands.report, "format_report", lambda rows, s, k=5: "")
    return rec


def test_run_eval_defaults_measure_hybrid_with_the_reranker(eval_harness):
    commands.run_eval()
    assert eval_harness.kw["hybrid"] is True
    assert eval_harness.kw["reranker"] == "RERANKER"
    assert eval_harness.kw["k"] == 5


def test_run_eval_dense_only_turns_off_bm25_but_keeps_the_reranker(eval_harness):
    commands.run_eval(dense_only=True)
    assert eval_harness.kw["hybrid"] is False
    assert eval_harness.kw["reranker"] == "RERANKER"


def test_run_eval_no_rerank_drops_the_reranker_only(eval_harness):
    commands.run_eval(no_rerank=True)
    assert eval_harness.kw["hybrid"] is True
    assert eval_harness.kw["reranker"] is None


def test_run_eval_retrieval_only_buys_no_generation_at_all(eval_harness):
    """The path that must stay free of Anthropic spend."""
    commands.run_eval(retrieval_only=True)
    assert eval_harness.kw["answerer"] is None
    assert eval_harness.kw["judge"] is None


def test_run_eval_no_judge_generates_but_skips_the_faithfulness_pass(eval_harness):
    commands.run_eval(no_judge=True)
    assert eval_harness.kw["answerer"] is not None
    assert eval_harness.kw["judge"] is None


def test_run_eval_limit_bounds_the_questions_scored(eval_harness):
    commands.run_eval(limit=2)
    assert len(eval_harness.items) == 2


def test_run_eval_zero_limit_scores_the_whole_set(eval_harness):
    commands.run_eval(limit=0)
    assert len(eval_harness.items) == 4


def test_run_eval_forwards_the_generation_cap(eval_harness):
    commands.run_eval(max_tokens=512)
    assert "512" in eval_harness.kw["answerer"]


# --- run_query / run_ask: the same wiring, on the display commands ----------------
# `run_eval` is where the published numbers come from, but the ablation flags are
# wired separately in each command, so a fix in one does not protect the others.

@pytest.fixture
def query_harness(monkeypatch):
    seen = {}

    def fake_search(text, db, embedder, k=5, *, hybrid=True, reranker=None, **kw):
        seen.update(text=text, k=k, hybrid=hybrid, reranker=reranker, embedder=embedder)
        return [{"text": "body", "score": 0.9, "section_path": "S", "url": "u"}]

    monkeypatch.setattr(commands, "VoyageEmbedder", lambda *a, **kw: "EMBEDDER")
    monkeypatch.setattr(commands, "VoyageReranker", lambda *a, **kw: "RERANKER")
    monkeypatch.setattr(commands.search_mod, "search", fake_search)
    return seen


def test_run_query_defaults_are_hybrid_with_the_reranker(query_harness):
    commands.run_query("q")
    assert query_harness["hybrid"] is True and query_harness["reranker"] == "RERANKER"


def test_run_query_dense_only_keeps_the_reranker(query_harness):
    """--dense-only drops BM25 and nothing else; the two flags stay independent."""
    commands.run_query("q", dense_only=True)
    assert query_harness["hybrid"] is False
    assert query_harness["reranker"] == "RERANKER"


def test_run_query_no_rerank_drops_only_the_reranker(query_harness):
    commands.run_query("q", no_rerank=True)
    assert query_harness["hybrid"] is True and query_harness["reranker"] is None


def test_run_query_both_flags_give_the_bare_dense_baseline(query_harness):
    commands.run_query("q", dense_only=True, no_rerank=True)
    assert query_harness["hybrid"] is False and query_harness["reranker"] is None


def test_run_query_forwards_k(query_harness):
    commands.run_query("q", k=3)
    assert query_harness["k"] == 3


@pytest.fixture
def ask_harness(monkeypatch):
    seen = {}

    class _Client:
        model = "fake-model"
        usage = {"total_tokens": 0}

        def __init__(self, **kw):
            self.kw = kw

    def fake_answer(text, db, embedder, answerer, k=5, *, hybrid=True,
                    reranker=None, trace=None, **kw):
        seen.update(k=k, hybrid=hybrid, reranker=reranker)
        return {"answer": "a", "sources": [], "grounded": True, "usage": {}}

    monkeypatch.setattr(commands, "VoyageEmbedder", lambda *a, **kw: _Client())
    monkeypatch.setattr(commands, "VoyageReranker", lambda *a, **kw: _Client())
    monkeypatch.setattr(commands, "ClaudeAnswerer", lambda **kw: _Client(**kw))
    monkeypatch.setattr(commands.generate_mod, "answer", fake_answer)
    return seen


def test_run_ask_dense_only_keeps_the_reranker(ask_harness):
    commands.run_ask("q", dense_only=True)
    assert ask_harness["hybrid"] is False and ask_harness["reranker"] is not None


def test_run_ask_no_rerank_drops_only_the_reranker(ask_harness):
    commands.run_ask("q", no_rerank=True)
    assert ask_harness["hybrid"] is True and ask_harness["reranker"] is None


def test_the_module_entry_point_resolves():
    # `python -m rag` is the command every doc prints. Nothing else imports
    # rag.__main__, so without this the one line it contains is never run.
    import rag.__main__ as entry
    assert entry.main is cli.main
