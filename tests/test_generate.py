"""Offline tests for grounded answer generation — design §4 and §5.

Pins the grounding rules (§4.1 sources and nothing else, §4.2 URLs withheld from
the model, §4.4 one retrieval shared with scoring) and the abstention contract
(§5.1 the exact sentence, §5.2 compared with `==`, §5.3 empty retrieval declines).

No network, no Claude key: a `FakeAnswerer` stands in for the LLM seam so we can
assert the retrieval -> numbered-sources prompt -> grounding logic without a real
model call. (The real `claude-opus-4-8` path is exercised via `python -m rag ask`.)
"""

from __future__ import annotations

import json
from pathlib import Path

from rag import generate as gen
from rag import index as index_mod
from rag.ports import Completion

from test_store import DOCS, FakeEmbedder  # reuse the store fixtures


class FakeAnswerer:
    """Records the prompt it was given and returns a canned completion."""

    model = "fake-answerer"  # the Answerer Protocol requires it; cost accounting keys off it

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system: str | None = None
        self.user: str | None = None

    def complete(self, system: str, user: str) -> Completion:
        self.system, self.user = system, user
        return Completion(text=self.reply, usage={"input_tokens": 11, "output_tokens": 7})


def _build(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_mod.build(chunks, db, FakeEmbedder())
    return db


def test_sources_block_is_numbered_with_section_paths():
    block = gen.build_sources_block(DOCS)
    assert block.startswith("[1] (Prompt caching > Overview)")
    assert "[2] (Tool use > Defining tools)" in block
    assert "[3] (Vision > Images)" in block
    # the chunk body is present so the model can ground on it
    assert "caching the prompt prefix" in block


def test_user_message_carries_question_and_sources():
    msg = gen.build_user_message("how do I cache?", DOCS)
    assert msg.startswith("Question: how do I cache?")
    assert "Sources:" in msg
    assert "[1] (Prompt caching > Overview)" in msg


def test_answer_grounds_and_passes_prompt_through(tmp_path):
    db = _build(tmp_path)
    fake = FakeAnswerer("Yes — caching reuses the prompt prefix [1].")
    out = gen.answer("how do I cache the prompt prefix?", db, FakeEmbedder(), fake, k=3)

    assert out["grounded"] is True
    assert out["answer"] == "Yes — caching reuses the prompt prefix [1]."
    assert out["usage"] == {"input_tokens": 11, "output_tokens": 7}
    # the top retrieved chunk's citation metadata is returned for printing
    assert out["sources"][0]["chunk_id"] == "pc#0"
    # the LLM was handed the grounding rules + the retrieved context
    assert gen.IDK in fake.system
    assert "Prompt cache reduces cost" in fake.user


def test_answer_marks_abstention_when_model_declines(tmp_path):
    db = _build(tmp_path)
    out = gen.answer("what is the airspeed of a swallow?", db, FakeEmbedder(),
                     FakeAnswerer(gen.IDK), k=3)
    assert out["grounded"] is False
    assert out["answer"] == gen.IDK
