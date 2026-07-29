"""Pure helpers for the M7 demo — no Streamlit import, so they're unit-testable
offline (`tests/test_demo.py`) alongside the rest of the suite.

`link_citations` is the visible payoff of grounding (each `[n]` becomes a link to
the cited section); `validate_examples` enforces the demo contract and, critically,
the licensing invariant: a baked example must carry **no corpus body**.
"""

from __future__ import annotations

import re
from typing import Any

# Must match generate.IDK — the exact string the model uses to abstain.
IDK = "I don't know based on the provided documentation."


def link_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    """Turn each inline `[n]` in `answer` into a markdown link to source n's URL.

    Unknown numbers (no matching source) are left as-is. The brackets are escaped
    so the rendered link text reads `[n]`.
    """
    by_n = {s["n"]: s["url"] for s in sources}

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[\\[{n}\\]]({by_n[n]})" if n in by_n else m.group(0)

    return re.sub(r"\[(\d+)\]", repl, answer)


def validate_examples(data: dict[str, Any]) -> list[str]:
    """Return a list of contract violations in a baked `examples.json` ([] = OK).

    Guards both the demo's usefulness (it has grounded + abstention examples that
    actually cite / decline) and the licensing posture (no chunk body ships).
    """
    problems: list[str] = []
    examples = data.get("examples", [])
    if not examples:
        return ["no examples"]
    if not any(e["grounded"] for e in examples):
        problems.append("no grounded example")
    if not any(not e["grounded"] for e in examples):
        problems.append("no abstention example")
    for e in examples:
        q = e["question"]
        for s in e["sources"]:
            if "text" in s:  # the corpus body must never reach the demo artifact
                problems.append(f"source carries corpus body: {q!r}")
            if not str(s.get("url", "")).startswith("http"):
                problems.append(f"source missing a URL: {q!r}")
        if e["grounded"]:
            if not re.search(r"\[\d+\]", e["answer"]):
                problems.append(f"grounded answer has no [n] citation: {q!r}")
        elif e["answer"] != IDK:
            problems.append(f"abstention answer is not the exact decline string: {q!r}")
    return problems
