"""Offline tests for the demo baker — design §8.2 and §8.4, the step that writes what the
public serves.

`demo.bake` is the only module that produces `demo/examples.json`, and until this
file existed nothing exercised it: the artifact was checked after the fact, so a
bake that violated the contract was caught only if someone remembered to run the
suite before committing. Here the baker runs end to end against a temporary index
and fake clients, and its publishing guard is driven directly.

No key, no network: `bake_one` takes its clients and its index as arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo import bake
from demo.render import IDK, validate_examples
from rag import index as index_mod

from test_store import FakeEmbedder  # reuse the store fixture

# Same shape as the real corpus rows, but with the http URLs the demo contract
# requires (a citation the reader cannot click is not a citation).
DOCS = [
    {"id": "pc#0", "url": "https://docs.example/pc#a", "source_url": "https://docs.example/pc",
     "page_title": "Prompt caching", "section_path": "Prompt caching > Overview", "anchor": "a",
     "text": "Prompt cache reduces cost by caching the prompt prefix across requests."},
    {"id": "tu#0", "url": "https://docs.example/tu#b", "source_url": "https://docs.example/tu",
     "page_title": "Tool use", "section_path": "Tool use > Defining tools", "anchor": "b",
     "text": "Define a tool with a schema and let Claude call the tool."},
    {"id": "vs#0", "url": "https://docs.example/vs#c", "source_url": "https://docs.example/vs",
     "page_title": "Vision", "section_path": "Vision > Images", "anchor": "c",
     "text": "Vision lets the model read an image and answer questions about it."},
]


class MeteredEmbedder(FakeEmbedder):
    """FakeEmbedder plus the cumulative counter the trace snapshots before/after."""

    def __init__(self) -> None:
        self.usage = {"total_tokens": 0}

    def embed(self, texts, *, input_type):
        self.usage["total_tokens"] += 13
        return super().embed(texts, input_type=input_type)


class FakeReranker:
    model = "fake-reranker"

    def __init__(self) -> None:
        self.usage = {"total_tokens": 0}

    def rerank(self, query, documents, *, top_k=None):
        self.usage["total_tokens"] += 29
        n = len(documents) if top_k is None else min(top_k, len(documents))
        return [(i, 1.0 - i / 100) for i in range(n)]


class FakeAnswerer:
    model = "fake-answerer"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system: str, user: str):
        from rag.ports import Completion
        return Completion(text=self.reply, usage={"input_tokens": 11, "output_tokens": 7})


def _index(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_mod.build(chunks, db, FakeEmbedder())
    return db


def _bake(tmp_path: Path, reply: str) -> dict:
    return bake.bake_one("how do I cache the prompt prefix?", MeteredEmbedder(),
                         FakeReranker(), FakeAnswerer(reply), db_path=_index(tmp_path))


# --- what one baked entry has to contain ------------------------------------------

def test_a_baked_entry_carries_citations_and_no_corpus_body(tmp_path):
    ex = _bake(tmp_path, "Caching reuses the prefix [1].")
    assert ex["grounded"] is True
    assert [s["n"] for s in ex["sources"]] == [1, 2, 3]
    assert all(s["url"].startswith("https://") for s in ex["sources"])
    # the licensing invariant, enforced at the point the artifact is produced
    assert not any("text" in s for s in ex["sources"])


def test_a_baked_entry_carries_a_per_model_cost_breakdown(tmp_path):
    ex = _bake(tmp_path, "Caching reuses the prefix [1].")
    by_model = ex["trace"]["cost_by_model"]
    assert {"fake-bow", "fake-reranker", "fake-answerer"} == set(by_model)
    # every model needs a usd figure — app.py renders it without checking
    assert all("usd" in u for u in by_model.values())
    assert ex["trace"]["total_ms"] > 0 and ex["trace"]["stages"]


def test_an_abstention_bakes_as_not_grounded(tmp_path):
    ex = _bake(tmp_path, IDK)
    assert ex["grounded"] is False and ex["answer"] == IDK


def test_a_full_payload_meets_the_contract_it_will_be_published_under(tmp_path):
    grounded = _bake(tmp_path, "Caching reuses the prefix [1].")
    abstained = _bake(tmp_path, IDK)
    payload = bake.build_payload([grounded, abstained], 0.42)
    assert validate_examples(payload) == []
    assert payload["bake_cost_usd"] == 0.42


# --- §8.4 a failed batch keeps what it already paid for ---------------------------

def _entry(question: str) -> dict:
    return {"question": question, "answer": f"a [1] for {question}", "grounded": True,
            "sources": [{"n": 1, "url": "https://x", "section_path": "S"}],
            "trace": {"stages": [], "total_ms": 1.0, "total_usd": 0.02,
                      "cost_by_model": {"m": {"input_tokens": 1, "usd": 0.02}}}}


def test_a_failure_part_way_through_keeps_the_answers_already_bought(tmp_path):
    partial = tmp_path / "examples.json.partial"

    def flaky(q: str) -> dict:
        if q == "third":
            raise RuntimeError("overloaded")
        return _entry(q)

    with pytest.raises(RuntimeError):
        bake.bake_all(["first", "second", "third"], flaky, partial_file=partial)

    kept = json.loads(partial.read_text(encoding="utf-8"))
    assert [e["question"] for e in kept] == ["first", "second"]


def test_a_rerun_pays_only_for_what_is_missing(tmp_path):
    partial = tmp_path / "examples.json.partial"
    partial.write_text(json.dumps([_entry("first"), _entry("second")]), encoding="utf-8")
    charged: list[str] = []

    def bake_one_question(q: str) -> dict:
        charged.append(q)
        return _entry(q)

    out = bake.bake_all(["first", "second", "third"], bake_one_question,
                        partial_file=partial)

    assert charged == ["third"]                    # the two already bought are not re-bought
    assert [e["question"] for e in out] == ["first", "second", "third"]
    assert not partial.exists()                    # a finished run cleans up after itself


def test_a_run_that_fails_on_its_first_question_leaves_no_partial(tmp_path):
    partial = tmp_path / "examples.json.partial"

    def always_fails(q: str) -> dict:
        raise RuntimeError("overloaded")

    with pytest.raises(RuntimeError):
        bake.bake_all(["only"], always_fails, partial_file=partial)
    assert not partial.exists()  # nothing was paid for, so there is nothing to keep


# --- the publishing guard ---------------------------------------------------------

def test_publish_writes_an_artifact_that_meets_the_contract(tmp_path):
    out = tmp_path / "examples.json"
    payload = bake.build_payload([_bake(tmp_path, "Caching reuses the prefix [1]."),
                                  _bake(tmp_path, IDK)], 0.42)
    bake.publish(payload, out_file=out)
    assert json.loads(out.read_text(encoding="utf-8"))["examples"]


def test_publish_refuses_a_violating_artifact_and_leaves_the_old_one_alone(tmp_path):
    out = tmp_path / "examples.json"
    out.write_text('{"kept": true}\n', encoding="utf-8")
    bad = bake.build_payload([{"question": "q", "answer": "a [1]", "grounded": True,
                               "sources": [{"n": 1, "url": "https://x",
                                            "section_path": "S", "text": "leaked body"}],
                               "trace": {"stages": [], "total_ms": 1.0, "total_usd": 0.0,
                                         "cost_by_model": {}}}], 0.1)

    with pytest.raises(SystemExit) as e:
        bake.publish(bad, out_file=out)

    assert "corpus body" in str(e.value)
    # the published file is untouched, and the paid-for output is kept to inspect
    assert json.loads(out.read_text(encoding="utf-8")) == {"kept": True}
    rejected = tmp_path / "examples.json.rejected"
    assert json.loads(rejected.read_text(encoding="utf-8"))["examples"][0]["question"] == "q"
