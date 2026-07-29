"""M7 demo tests — offline, no key, no Streamlit.

Guards the public demo's contract and the licensing invariant: the baked
`demo/examples.json` ships grounded + abstention examples that cite / decline
correctly, and carries **no corpus body**. Also covers the
citation-linking helper the UI renders with.
"""

from __future__ import annotations

import json
from pathlib import Path

from demo.render import IDK, link_citations, validate_examples

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = json.loads((ROOT / "demo" / "examples.json").read_text(encoding="utf-8"))


def test_examples_meet_the_contract():
    assert validate_examples(EXAMPLES) == []


def test_no_corpus_body_in_artifact():
    # The licensing-critical invariant: only section paths + URLs ship, never text.
    assert not any("text" in s for e in EXAMPLES["examples"] for s in e["sources"])


def test_has_grounded_and_one_abstention():
    grounded = [e for e in EXAMPLES["examples"] if e["grounded"]]
    abstained = [e for e in EXAMPLES["examples"] if not e["grounded"]]
    assert grounded, "demo should show at least one grounded answer"
    assert abstained, "demo should show abstention (hallucination suppression)"
    assert all(e["answer"] == IDK for e in abstained)


def test_link_citations_maps_each_number_to_its_source():
    sources = [{"n": 1, "url": "https://x/#a"}, {"n": 2, "url": "https://y/#b"}]
    out = link_citations("Caching is cheaper [1]; enable with cache_control [2]. See [9].",
                         sources)
    assert "(https://x/#a)" in out and "(https://y/#b)" in out
    assert "[9]" in out  # no matching source -> left untouched


def test_validate_flags_corpus_body_leak():
    bad = {"examples": [{"question": "q", "grounded": True, "answer": "a [1]",
                         "sources": [{"n": 1, "url": "https://x", "text": "leaked body"}]}]}
    assert any("corpus body" in p for p in validate_examples(bad))
