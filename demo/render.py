"""Pure helpers for the public demo (design §8) — no Streamlit import, so unit-testable
offline (`tests/test_demo.py`) alongside the rest of the suite.

`link_citations` is the visible payoff of grounding (each `[n]` becomes a link to
the cited section); `validate_examples` enforces the demo contract and, critically,
the licensing invariant: a baked example must carry **no corpus body**.

Both read `[n]` the same way, through `_split_code`, because a citation and a
Python subscript look identical to a regular expression. They must agree on which
brackets are citations or the page will link something the validator never saw.
"""

from __future__ import annotations

import re
from typing import Any

# Must match generate.IDK — the exact string the model uses to abstain.
IDK = "I don't know based on the provided documentation."

# Markdown code, in the order the alternatives must be tried: a closed fence, an
# unterminated fence (which runs to the end of the answer), then an inline span.
_CODE = re.compile(r"```.*?```|```.*|`[^`\n]+`", re.S)
_CITE = re.compile(r"\[(\d+)\]")


def _split_code(text: str) -> list[tuple[str, bool]]:
    """Split `text` into ordered `(segment, is_code)` pairs.

    The corpus is API documentation, so answers quote things like
    `response.content[1].text`. Those brackets are subscripts, not citations, and
    rewriting one into a hyperlink corrupts the code sample on the public page —
    in a project whose entire claim is that its citations are trustworthy.
    """
    parts: list[tuple[str, bool]] = []
    last = 0
    for m in _CODE.finditer(text):
        if m.start() > last:
            parts.append((text[last:m.start()], False))
        parts.append((m.group(0), True))
        last = m.end()
    parts.append((text[last:], False))
    return parts


def cited_numbers(answer: str) -> list[int]:
    """The `[n]` citation numbers in `answer`, ignoring anything inside code."""
    return [int(n) for seg, is_code in _split_code(answer) if not is_code
            for n in _CITE.findall(seg)]


def link_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    """Turn each inline `[n]` in `answer` into a markdown link to source n's URL.

    Code spans pass through untouched. Unknown numbers (no matching source) are
    left as-is. The brackets are escaped so the rendered link text reads `[n]`.
    """
    by_n = {s["n"]: s["url"] for s in sources}

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[\\[{n}\\]]({by_n[n]})" if n in by_n else m.group(0)

    return "".join(seg if is_code else _CITE.sub(repl, seg)
                   for seg, is_code in _split_code(answer))


def validate_examples(data: dict[str, Any]) -> list[str]:
    """Return a list of contract violations in a baked `examples.json` ([] = OK).

    The contract is everything `app.py` reads without checking first: a missing
    key there is a traceback on the public page. It also guards the demo's
    usefulness (grounded + abstention examples that really cite / decline) and
    the licensing posture (no chunk body ships).

    `demo.bake` runs this before writing, so a bad bake cannot reach the repo,
    and `tests/test_demo.py` runs it against the committed artifact.
    """
    problems: list[str] = []
    for key in ("generated_at", "k", "generation_model", "retrieval"):
        if key not in data:  # the caption under the demo reads all four
            problems.append(f"missing top-level key: {key!r}")

    examples = data.get("examples", [])
    if not examples:
        return problems + ["no examples"]
    if not any(e["grounded"] for e in examples):
        problems.append("no grounded example")
    if not any(not e["grounded"] for e in examples):
        problems.append("no abstention example")

    for e in examples:
        q = e["question"]
        numbers = [s.get("n") for s in e["sources"]]
        if numbers != list(range(1, len(e["sources"]) + 1)):
            # link_citations indexes sources by `n`; anything else silently
            # drops a citation or raises on lookup.
            problems.append(f"sources are not numbered 1..n: {q!r}")
        for s in e["sources"]:
            if "text" in s:  # the corpus body must never reach the demo artifact
                problems.append(f"source carries corpus body: {q!r}")
            if not str(s.get("url", "")).startswith("http"):
                problems.append(f"source missing a URL: {q!r}")
            if not s.get("section_path"):
                problems.append(f"source missing a section path: {q!r}")

        cited = cited_numbers(e["answer"])
        if e["grounded"]:
            if not cited:
                problems.append(f"grounded answer has no [n] citation: {q!r}")
            unresolved = sorted({n for n in cited if n not in set(numbers)})
            if unresolved:
                problems.append(f"answer cites missing sources {unresolved}: {q!r}")
        elif e["answer"] != IDK:
            problems.append(f"abstention answer is not the exact decline string: {q!r}")

        problems.extend(_trace_problems(e.get("trace"), q))
    return problems


def _trace_problems(trace: Any, question: str) -> list[str]:
    """Violations in one example's trace block — the metrics row and the cost table."""
    if not isinstance(trace, dict):
        return [f"missing trace: {question!r}"]
    problems = [f"trace missing {key!r}: {question!r}"
                for key in ("stages", "total_ms", "total_usd", "cost_by_model")
                if key not in trace]
    for model, usage in trace.get("cost_by_model", {}).items():
        if "usd" not in usage:  # app.py renders `${u['usd']:.6f}` per model
            problems.append(f"cost_by_model[{model!r}] has no usd: {question!r}")
    return problems
