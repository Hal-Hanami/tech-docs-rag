"""The README claims every number in it is reproducible. This checks the ones that can be.

Two kinds of number appear in that file. Most require a paid run — recall, MRR,
faithfulness — and those are out of scope here; their reproduction commands are
printed beside them. The rest are *derived*: the cost table follows from the token
counts and the rate table in `rag/observe.py`, and the question counts follow from
`eval/qa.jsonl`. Those can drift silently, and a stale price or an edited row would
leave the repository asserting something its own code no longer produces.

So the numbers are read out of `README.md` itself rather than copied here. Copying
them would let the two drift apart, which is the failure this exists to prevent.

If this fails: either a rate in `observe.PRICING` changed — in which case re-measure
and update the table with the new figures — or the table was edited by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag import observe
from rag.eval import load_items

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
# Prose wraps, so a sentence can straddle a line break; tables never do.
PROSE = re.sub(r"\s+", " ", README)

ANSWER_MODEL = "claude-opus-4-8"  # the model the cost table was measured on


def _row(label: str) -> list[str]:
    """The cells of the README table row whose first column contains `label`."""
    for line in README.splitlines():
        if line.startswith("|") and label in line:
            return [c.strip().strip("*").strip() for c in line.strip("|").split("|")]
    pytest.fail(f"no README table row for {label!r} — was the cost table renamed?")


def _num(cell: str) -> float:
    """A README figure: strips $ , % and the typographic minus the table uses."""
    return float(cell.replace("$", "").replace(",", "").replace("%", "")
                     .replace("−", "-").strip())


@pytest.fixture(scope="module")
def published() -> dict:
    tokens_in = _row("generation input tokens")
    tokens_out = _row("generation output tokens")
    cost = _row("total cost per query")
    rerank = re.search(r"its \$([0-9.]+) is identical in both runs", PROSE)
    assert rerank, "README no longer states the k-independent rerank spend"
    n_q = re.search(r"core slice, (\d+) q", PROSE)
    assert n_q, "README no longer states how many questions the cost run covered"
    return {
        "n_queries": int(n_q.group(1)),
        "rerank_usd": float(rerank.group(1)),
        "k5": {"in": _num(tokens_in[1]), "out": _num(tokens_out[1]), "usd": _num(cost[1])},
        "k3": {"in": _num(tokens_in[2]), "out": _num(tokens_out[2]), "usd": _num(cost[2])},
        "d_in": _num(tokens_in[3]), "d_out": _num(tokens_out[3]), "d_usd": _num(cost[3]),
    }


def _per_query(run: dict, published: dict) -> float:
    ledger = {ANSWER_MODEL: {"input_tokens": int(run["in"]), "output_tokens": int(run["out"])}}
    generation = observe.cost_usd(ledger)["total"]
    # The reranker scores the full candidate pool, so its spend does not move with k.
    return (generation + published["rerank_usd"]) / published["n_queries"]


@pytest.mark.parametrize("run", ["k5", "k3"])
def test_the_published_cost_reproduces_from_the_rate_table(published, run):
    derived = round(_per_query(published[run], published), 4)
    assert derived == published[run]["usd"], (
        f"README says ${published[run]['usd']:.4f}/query at {run}, but the token counts "
        f"in the same table and observe.PRICING give ${derived:.4f}. Re-measure, or fix the table."
    )


def test_the_published_deltas_match_the_published_figures(published):
    def pct(a: float, b: float) -> float:
        return (b - a) / a * 100

    assert round(pct(published["k5"]["in"], published["k3"]["in"])) == published["d_in"]
    assert round(pct(published["k5"]["out"], published["k3"]["out"])) == published["d_out"]
    assert round(pct(published["k5"]["usd"], published["k3"]["usd"])) == published["d_usd"]


def test_the_token_saving_and_the_cost_saving_are_not_the_same_number(published):
    # The paragraph under the table is built on this gap: quoting the token saving
    # as the cost saving would overstate the result. If they ever coincide, that
    # argument has to be rewritten rather than left standing.
    assert published["d_in"] != published["d_usd"]


def test_the_eval_set_matches_the_slice_sizes_the_readme_quotes(published):
    core = load_items(ROOT / "eval" / "qa.jsonl", tag="core")
    hard = load_items(ROOT / "eval" / "qa.jsonl", tag="hard")
    in_corpus = [i for i in core if i.in_corpus]

    assert len(core) == published["n_queries"]  # "core slice, 26 q" in the cost table
    assert len(in_corpus) == int(_row("**core** (")[0].split("(")[1].split(" ")[0])
    assert len(hard) == int(_row("**hard** (")[0].split("(")[1].split(" ")[0])
