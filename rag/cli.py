"""Command-line surface: flags in, `rag.commands` calls out.

Holds no pipeline logic. Everything this module does is describe the flags and
translate one parsed namespace into one function call, which keeps the
orchestration in `rag.commands` callable without argparse.

    python -m rag index [--limit N] [--model M]      # chunks.jsonl -> data/index.db
    python -m rag query "how do I cache a prompt?"   # top-k with citations
    python -m rag ask   "how do I cache a prompt?"   # grounded answer + trace
    python -m rag eval  [--retrieval-only]           # quality metrics

`index` and `query` need a Voyage key; `ask` and `eval` also need an Anthropic
key. `eval --retrieval-only` needs no Anthropic key — it still embeds the
questions, so Voyage is required either way.
"""

from __future__ import annotations

import argparse

from . import commands, config
from .clients.voyage import DEFAULT_EMBED_MODEL


def _add_retrieval_flags(p: argparse.ArgumentParser) -> None:
    """The two ablation toggles, each dropping exactly one retrieval stage.

    They are independent by design, which makes all four combinations
    measurable on one index. The M2-era baseline (no BM25, no reranker) is both
    flags together — `--dense-only` on its own still reranks, and the help text
    says so because the earlier wording implied otherwise and produced a
    reproduction command that reported the wrong configuration.
    """
    p.add_argument("--dense-only", action="store_true",
                   help="skip the BM25 hybrid — dense vectors only "
                        "(still reranks; add --no-rerank for the un-reranked baseline)")
    p.add_argument("--no-rerank", action="store_true",
                   help="skip the reranker (measure retrieval without it)")


def _add_generation_flags(p: argparse.ArgumentParser) -> None:
    """Generation-side knobs shared by `ask` and `eval`.

    `--max-tokens` caps thinking *and* answer text together, which is why a
    truncated answer is a `--max-tokens` problem rather than a prompt problem.
    """
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="output cap for the answer, thinking included (default 4096)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="build the vector index from chunks.jsonl")
    p_index.add_argument("--limit", type=int, default=0, help="index only the first N chunks")
    p_index.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    p_index.set_defaults(
        run=lambda a: commands.run_index(model=a.model, limit=a.limit))

    p_query = sub.add_parser("query", help="top-k retrieval for a query string")
    p_query.add_argument("text", help="the query string")
    p_query.add_argument("-k", type=int, default=5, help="number of results")
    p_query.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    _add_retrieval_flags(p_query)
    p_query.set_defaults(
        run=lambda a: commands.run_query(a.text, k=a.k, model=a.model,
                                         dense_only=a.dense_only, no_rerank=a.no_rerank))

    p_ask = sub.add_parser("ask", help="grounded answer with citations")
    p_ask.add_argument("text", help="the question")
    p_ask.add_argument("-k", type=int, default=5, help="number of chunks to ground on")
    _add_generation_flags(p_ask)
    _add_retrieval_flags(p_ask)
    p_ask.set_defaults(
        run=lambda a: commands.run_ask(a.text, k=a.k, max_tokens=a.max_tokens,
                                       dense_only=a.dense_only, no_rerank=a.no_rerank))

    p_eval = sub.add_parser("eval", help="score the eval set (recall/MRR, faithfulness, abstention)")
    p_eval.add_argument("-k", type=int, default=5, help="top-k for retrieval and grounding")
    p_eval.add_argument("--tag", default="", help="score only this slice (e.g. core, hard)")
    p_eval.add_argument("--limit", type=int, default=0, help="score only the first N questions")
    p_eval.add_argument("--retrieval-only", action="store_true",
                        help="rank metrics only — no LLM calls")
    p_eval.add_argument("--no-judge", action="store_true",
                        help="generate and score abstention, but skip the faithfulness judge")
    _add_generation_flags(p_eval)
    _add_retrieval_flags(p_eval)
    p_eval.set_defaults(
        run=lambda a: commands.run_eval(k=a.k, tag=a.tag, limit=a.limit,
                                        retrieval_only=a.retrieval_only, no_judge=a.no_judge,
                                        max_tokens=a.max_tokens,
                                        dense_only=a.dense_only, no_rerank=a.no_rerank))
    return parser


def main(argv: list[str] | None = None) -> None:
    # Load `.env` before parsing so a missing key is reported by the client that
    # needs it, with its own instructions, rather than by argparse.
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    args.run(args)
