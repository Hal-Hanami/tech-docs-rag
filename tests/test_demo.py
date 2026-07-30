"""Demo tests — design §8, §4.3, §2.2. Offline, no key, no Streamlit.

Guards the published artifact's contract (§8.1 what it carries, §8.3 that the
contract covers everything the page reads unguarded) and the licensing invariant
(§2.2 no corpus body ever ships). Also pins §4.3: a `[n]` inside code is a
subscript, not a citation, and the linking helper must leave it alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from demo.render import IDK, cited_numbers, link_citations, validate_examples

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
    bad = _artifact(answer="a [1]", sources=[{"n": 1, "url": "https://x",
                                              "section_path": "S", "text": "leaked body"}])
    assert any("corpus body" in p for p in validate_examples(bad))


# --- citations vs code: identical to a regex, different to a reader ---------------

# The corpus is API documentation, so an answer that quotes `response.content[1]`
# is normal. Rewriting that subscript into a hyperlink corrupts the code sample on
# the public page. The committed artifact only survives today because the one
# subscript it quotes is `[0]`, which happens to fall outside the source range.
CODE_ANSWER = (
    "Read the text off the first block [1]:\n\n"
    "```python\n"
    "print(response.content[1].text)\n"
    "```\n\n"
    "The inline form is `response.content[2].text` [2]."
)


def test_link_citations_leaves_subscripts_inside_code_alone():
    out = link_citations(CODE_ANSWER, [{"n": 1, "url": "https://x"},
                                       {"n": 2, "url": "https://y"}])
    assert "print(response.content[1].text)" in out
    assert "`response.content[2].text`" in out


def test_link_citations_still_links_the_prose_around_that_code():
    out = link_citations(CODE_ANSWER, [{"n": 1, "url": "https://x"},
                                       {"n": 2, "url": "https://y"}])
    assert out.count("(https://x)") == 1 and out.count("(https://y)") == 1


def test_cited_numbers_ignores_code():
    assert cited_numbers(CODE_ANSWER) == [1, 2]


def test_an_unterminated_fence_is_code_to_the_end():
    # A truncated answer must not have its half-written code sample rewritten.
    out = link_citations("see [1]\n```python\nx = y[1]\n", [{"n": 1, "url": "https://x"}])
    assert "x = y[1]" in out and "(https://x)" in out


# --- the contract is everything app.py reads without guarding ---------------------

def _artifact(**overrides) -> dict:
    """A minimal valid artifact, so each test can break exactly one thing."""
    example = {
        "question": "q", "answer": "a [1]", "grounded": True,
        "sources": [{"n": 1, "url": "https://x", "section_path": "S"}],
        "trace": {"stages": [["embed", 1.0]], "total_ms": 1.0, "total_usd": 0.01,
                  "cost_by_model": {"m": {"input_tokens": 1, "usd": 0.01}}},
    }
    example.update(overrides)
    return {"generated_at": "2026-01-01", "k": 3, "generation_model": "m",
            "retrieval": "hybrid", "examples": [example, _abstention()]}


def _abstention() -> dict:
    return {"question": "oob", "answer": IDK, "grounded": False,
            "sources": [{"n": 1, "url": "https://x", "section_path": "S"}],
            "trace": {"stages": [], "total_ms": 1.0, "total_usd": 0.0,
                      "cost_by_model": {}}}


def test_the_baseline_artifact_is_valid():
    # Otherwise the negative cases below could pass for the wrong reason.
    assert validate_examples(_artifact()) == []


def test_validate_flags_a_citation_with_no_matching_source():
    assert any("cites missing sources" in p
               for p in validate_examples(_artifact(answer="a [1] and [4]")))


def test_validate_flags_sources_that_are_not_numbered_from_one():
    bad = _artifact(sources=[{"n": 2, "url": "https://x", "section_path": "S"}])
    assert any("numbered 1..n" in p for p in validate_examples(bad))


def test_validate_flags_a_trace_the_metrics_row_cannot_render():
    bad = _artifact(trace={"stages": [], "total_ms": 1.0,
                           "cost_by_model": {}})  # no total_usd
    assert any("total_usd" in p for p in validate_examples(bad))


def test_validate_flags_a_model_with_no_dollar_figure():
    bad = _artifact(trace={"stages": [], "total_ms": 1.0, "total_usd": 0.0,
                           "cost_by_model": {"m": {"input_tokens": 1}}})
    assert any("no usd" in p for p in validate_examples(bad))


def test_validate_flags_a_missing_caption_field():
    data = _artifact()
    del data["generation_model"]
    assert any("generation_model" in p for p in validate_examples(data))
