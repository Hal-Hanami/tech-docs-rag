# tech-docs-rag

**Source-grounded RAG over technical documentation.** Ask a question, get an
answer with a citation to the exact section behind every claim — or an explicit
*"I don't know"* when the documentation does not support one.

[![tests](https://github.com/Hal-Hanami/tech-docs-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Hal-Hanami/tech-docs-rag/actions/workflows/ci.yml)

Retrieval-augmented generation is a commodity. What is not commodity is the
engineering around it: knowing whether retrieval actually works, knowing whether
the model is inventing things, and being able to put a number on what a query
costs.

```
┌─ tech-docs-rag ─ ask the docs ───────────────────────────┐
│ Q: how does prompt caching reduce cost?                  │
├──────────────────────────────────────────────────────────┤
│ Prompt caching reuses a cached prefix, so repeated input  │
│ tokens are billed at a fraction of the standard price [1] │
│ ; you enable it with cache_control on the block to cache  │
│ [2].                                                      │
│                                                          │
│ Sources  ✓ grounded                                      │
│  [1] Pricing > … > Prompt caching        → platform.cla… │
│  [3] Prompt caching > How it works       → platform.cla… │
│ ── Cost $0.0335 · Latency 10.8s · trace ▸ ───────────────│
│   generation   in=2098 out=893   $0.0328                 │
│   rerank       total=34526       $0.0007                 │
│   embedding    total=13          $0.0000                 │
└──────────────────────────────────────────────────────────┘
```

## What it measures

Four figures, each reproducible with a command printed beside it in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

| | | |
|---|---|---|
| recall@1 on the hard slice | **91.7 %** | up from 66.7 % without the reranker |
| recall@3 on the core slice | **100 %** | the right page is where the generator reads |
| abstention rate | **100 %** | with 0 false abstentions |
| cost per query | **$0.0207** | **−23 %** after trimming the generation context |

The measurements, their dates, their weaknesses, and the two claims this project
has withdrawn are all in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Documentation

| | |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | what the system must do and why — the invariants, numbered so code and tests can cite them |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | what measuring them returned, with the command that reproduces each figure |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | commit and comment conventions, and the five rules the build enforces |

## Running it

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv -e '.[dev]'
printf 'VOYAGE_API_KEY=...\nANTHROPIC_API_KEY=...\n' > .env   # gitignored

python -m ingest all              # fetch docs        → data/chunks.jsonl
python -m rag index               # embed             → data/index.db (~20s, 1,290 chunks)
python -m rag ask "how do I cache a prompt?" -k 3
```

`index` and `query` need a Voyage key; `ask` and `eval` also need an Anthropic
key. `eval --retrieval-only` needs no Anthropic key.

The demo UI runs without keys or an index, serving precomputed answers:

```bash
pip install -r requirements.txt   # just streamlit
streamlit run app.py
```

### Tests

```bash
python -m pytest -q               # 146 tests, no key, no network
```

## Corpus and licensing

The corpus is the Claude developer documentation, fetched from
`https://platform.claude.com/llms.txt`; each page is published as clean Markdown
at its `.md` URL, so there is no HTML scraping. Scope and the reasoning behind it
are in [`docs/DESIGN.md`](docs/DESIGN.md) §2.

The code is [MIT-licensed](LICENSE). The documentation text is Anthropic's
copyrighted material — this project **fetches it for local retrieval and does not
redistribute it**. The fetched Markdown, `chunks.jsonl`, and `index.db` are all
git-ignored, and the demo artifact ships **no corpus body** (only model-written
cited prose plus source URLs, enforced by a test).

## Built on this

[incident-triage-agent](https://github.com/Hal-Hanami/incident-triage-agent)
reuses the `rag` package here as the runbook search behind a read-only incident
triage agent on the Claude Agent SDK. It carries the same posture one rung up the
autonomy ladder: where this system declines to *answer* what its sources do not
support, that one declines to *act* when confidence is low, and escalates to a
human instead. The interface it depends on is pinned — see
[`docs/DESIGN.md`](docs/DESIGN.md) §9.
