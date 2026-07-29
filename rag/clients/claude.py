"""Anthropic adapters — answer generation (`Answerer`) and faithfulness judging (`Judge`).

Two models, chosen for different jobs: generation gets the strongest model
available because a wrong answer is the failure this project exists to prevent,
and judging gets the cheapest capable one because it runs once per answered
question and its output is a boolean plus a list of strings.

Why the judge's prompt lives here and the answerer's does not: the `Answerer`
port receives an already-assembled system prompt, because *what to tell the
model about grounding* is policy (`rag.generate` owns it). The `Judge` port is
defined in domain terms instead — question, answer, sources — so turning those
into a ruling is this implementation's business. A different `Judge` (a human
reviewer, a local model, a rules engine) would share the interface and share
none of the machinery below.

Both clients import `anthropic` lazily and check for a key in `__init__`, so
retrieval-only commands need neither the SDK installed nor a key present.
"""

from __future__ import annotations

import json
import os

from ..ports import Completion, Verdict

ANSWER_MODEL = "claude-opus-4-8"
JUDGE_MODEL = "claude-haiku-4-5"

# Structured-outputs schema for the verdict. Deliberately inside the documented
# json_schema subset (object / boolean / array-of-string / additionalProperties
# false) — a schema the API cannot compile fails the whole request.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithful", "unsupported_claims"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You are a strict grader checking whether an ANSWER is faithful to its "
    "SOURCES. You are given a QUESTION, the ANSWER, and the numbered SOURCES the "
    "answer was supposed to be grounded in.\n"
    "Decide whether every factual claim in the ANSWER is directly supported by "
    "the SOURCES. Judge the substance, not the citation markers like [1]. "
    "Reasonable paraphrase counts as supported; a claim that adds facts, numbers, "
    "names, or values not present in the SOURCES does not.\n"
    'Set "faithful" to false if ANY claim is unsupported, and list those claims. '
    "Output only the JSON object."
)


def _require_key() -> None:
    """Fail early and legibly when no Anthropic credential is present.

    Checked in the constructor rather than at call time so that a long eval run
    cannot get thirty questions in before discovering it cannot generate.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    raise SystemExit(
        "ANTHROPIC_API_KEY not set.\n"
        "  Put  ANTHROPIC_API_KEY=...  in .env  (gitignored), or export it;\n"
        "  or run retrieval-only, which needs no Anthropic key."
    )


def build_judge_user(question: str, answer: str, sources_block: str) -> str:
    """Assemble the judge's user message. Pure, so the wording is testable."""
    return f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{sources_block}"


def parse_verdict(text: str) -> Verdict:
    """Parse the judge's JSON ruling.

    Strict on purpose. Structured outputs guarantee well-formed JSON matching the
    schema, so a parse failure means an assumption broke — surfacing it beats
    swallowing it and silently scoring the run as unfaithful.
    """
    data = json.loads(text)
    return Verdict(
        faithful=bool(data["faithful"]),
        unsupported_claims=list(data.get("unsupported_claims", [])),
    )


class ClaudeAnswerer:
    """Answer generation. Implements `rag.ports.Answerer`."""

    def __init__(self, model: str = ANSWER_MODEL, max_tokens: int = 4096) -> None:
        import anthropic  # lazy: only generation paths need the SDK

        _require_key()
        self.model = model
        # Caps thinking *and* answer text together, so this needs headroom for
        # both; the truncation marker below exists because it is easy to set low.
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> Completion:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},  # the only on-mode for this model family
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        if msg.stop_reason == "max_tokens":
            # A truncated answer looks like a confident short one. Say so, rather
            # than letting a cut-off sentence be scored as a real answer.
            text += "\n[truncated: hit max_tokens — raise --max-tokens]"
        return Completion(text=text, usage=usage)


class ClaudeJudge:
    """Faithfulness judging. Implements `rag.ports.Judge`.

    `effort` is not passed: it is unsupported on this model, and sending it would
    fail the request rather than degrade gracefully.
    """

    def __init__(self, model: str = JUDGE_MODEL, max_tokens: int = 1024) -> None:
        import anthropic  # lazy: only judged eval runs need the SDK

        _require_key()
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def judge(self, question: str, answer: str, sources_block: str) -> Verdict:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user",
                       "content": build_judge_user(question, answer, sources_block)}],
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        verdict = parse_verdict(text)
        verdict.usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        return verdict
