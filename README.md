# tech-docs-rag

**Source-grounded RAG over technical documentation.** Ask a question, get an
answer with a citation to the exact section behind every claim — or an explicit
*"I don't know"* when the documentation does not support one.

[![tests](https://github.com/Hal-Hanami/tech-docs-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Hal-Hanami/tech-docs-rag/actions/workflows/ci.yml)

Retrieval-augmented generation is a commodity. What is not commodity is the
engineering around it: knowing whether retrieval actually works, knowing whether
the model is inventing things, and being able to put a number on what a query
costs. Every claim in this README is a number the evaluation harness in this
repository produces, and every one of them can be reproduced with a command
printed below it.

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
│  [1] Pricing > Prompt caching            → platform.cla… │
│  [2] Prompt caching > How it works       → platform.cla… │
│ ── Cost $0.0341 · Latency 11.3s · trace ▸ ───────────────│
│   generation   in=2098 out=917   $0.0334                 │
│   rerank       total=34526       $0.0007                 │
│   embedding    total=13          $0.0000                 │
└──────────────────────────────────────────────────────────┘
```

---

## Not hallucinating

Four layers, and the last one is a number.

1. **Grounded prompt.** The model receives the retrieved sources and nothing
   else. The system prompt forbids outside knowledge and requires a `[n]`
   citation on every claim (`rag/generate.py`).
2. **Abstention as a first-class outcome.** When the sources do not support an
   answer, the model must reply with one exact sentence rather than guess.
   Because it is exact, abstention is *detectable*, which is what makes it
   measurable.
3. **Better evidence.** Hybrid retrieval and a cross-encoder reranker put the
   right section in front of the model to begin with (table below).
4. **Measured.** On the evaluation set: **abstention rate 100 %** (4/4
   out-of-corpus questions declined) with **0 false abstentions** (no in-corpus
   question was wrongly refused).

The citations map one-to-one back to retrieval, which is what makes faithfulness
scorable at all — see the honesty note below for why that score is not used to
support any claim here.

## Costing what it costs

You cannot optimise what you cannot see. Every request carries a trace that
splits tokens and dollars **by model** — generation, judging, embedding and
reranking are priced separately from a sourced rate table (`rag/observe.py`).
That breakdown showed generation was ~90 % of spend, which pointed at exactly one
lever: how much context the generator receives.

Reranking saturates recall at rank 3 (`recall@3 = 100 %`), so chunks 4 and 5 cost
tokens without adding coverage. Cutting the generation context from **k = 5 to
k = 3**:

| metric (evaluation set, core slice, one session) | k = 5 | k = 3 | Δ |
|---|---|---|---|
| recall@1 / recall@3 / MRR | 86.4 % / 100 % / 0.932 | **identical** | **0** |
| abstention / false abstentions | 100 % / 0 | 100 % / 0 | 0 |
| **total cost per query** | $0.0303 | **$0.0219** | **−28 %** |
| generation input tokens | 88,499 | 54,197 | **−39 %** |
| latency p95 | 13.0 s | 10.2 s | −22 % |

The quality claim rests on the **deterministic** metrics, which are unchanged —
not on faithfulness, which is too noisy to carry it. The reranker scores the full
candidate pool regardless of `k`, so this trims generation cost without touching
retrieval quality.

---

## Retrieval quality

Same index, dense baseline versus the product configuration. `recall@5` is 100 %
for every configuration — the right page is essentially always *in* the pool — so
the reranker earns its keep on **rank**, which is what the generator actually
reads.

| slice | metric | dense | hybrid (RRF only) | **hybrid + rerank** |
|---|---|---|---|---|
| **core** (22 q) | recall@1 | 86.4 % | 72.7 % | **86.4 %** |
| | recall@3 | 100 % | 95.5 % | **100 %** |
| | MRR | 0.932 | 0.835 | **0.932** |
| **hard** (12 q) | recall@1 | 66.7 % | 58.3 % | **91.7 %** |
| | recall@3 | 100 % | 100 % | **100 %** |
| | MRR | 0.833 | 0.764 | **0.944** |

Reproduce any column — no LLM calls, no Anthropic key:

```bash
python -m rag eval --tag core --retrieval-only --dense-only --no-rerank  # dense
python -m rag eval --tag core --retrieval-only --no-rerank               # RRF only
python -m rag eval --tag core --retrieval-only                           # product
```

Three findings, all from the harness:

- **Plain RRF *hurts* rank.** On a corpus where dense retrieval is already
  strong, fusing in BM25's keyword hits pushes the right page off rank 1 —
  86.4 → 72.7 % on core, 66.7 → 58.3 % on hard. **The reranker is what makes
  hybrid safe:** it recovers that loss on both slices, back to parity on core and
  past the baseline on hard.
- **Reranking pays off on hard queries.** Exact-keyword lookups (`bash_20`,
  `str_replace`, `pause_turn`) and sibling-page disambiguation (web-fetch versus
  web-search) lift **recall@1 from 66.7 % to 91.7 %** on the hard slice. On the
  easier core slice dense retrieval is already at the ceiling, so the win appears
  exactly where the harder questions are.
- **Once you rerank, the BM25 half earns nothing here.** Reranking dense results
  alone (`--dense-only`, reranker on) scores *identically* to the full hybrid on
  every metric above — 86.4 % / 0.932 on core, 91.7 % / 0.944 on hard. On this
  corpus the measured win belongs entirely to the cross-encoder, not to the
  fusion. BM25 stays in because it is nearly free and is cheap insurance against
  a query the embedder misses, but the honest reading is: **this table does not
  demonstrate value from RRF.**

The RRF-only column is also the only one that moves between runs: core MRR
returns 0.835 rather than the 0.827 first recorded, and recall@3 95.5 % rather
than 100 %. Both are a single question crossing the rank-3/4 boundary
(1 ÷ 22 ≈ 0.008 of MRR). The dense and product columns are stable across runs;
the un-reranked fusion is the configuration that sits on a knife edge.

**Honesty note.** The faithfulness judge is **±20 points noisy run to run** — the
same pipeline scored 86.4 % one day and 63.6 % the next. This project therefore
treats faithfulness as a *direction*, never a headline, and rests every claim on
the deterministic metrics: recall, MRR, abstention rate, false abstentions.
Knowing which of your own metrics you cannot trust is part of the work.

---

## How it is built

One SQLite file is the entire datastore — dense vectors (sqlite-vec), BM25
(FTS5), and citation metadata all live in `index.db`, so the project deploys as a
single service with no additional infrastructure.

```
question
  → embed (voyage-4-lite)                ┐
  → dense kNN (sqlite-vec) ∪ BM25 (FTS5) → RRF fuse
  → rerank (rerank-2.5-lite)             ┘  cross-encoder, top-k by rank
  → generate (claude-opus-4-8): answer only from sources, cite [n], or decline
  → answer + citations + trace (per-stage latency, per-model token cost)
```

### Architecture

Dependencies point inward. Policy declares the interfaces it needs; the clients
conform to them.

```
rag/
  cli.py, commands.py   flags in, wiring out — the only layer that knows both
                        which client implements what and which policy wants it
  ports.py              the four seams: Embedder, Reranker, Answerer, Judge
  clients/              the only modules that open a network connection
    voyage.py             embeddings + reranking
    claude.py             generation + judging
  search.py             dense / hybrid / reranked retrieval
  generate.py           numbered-sources prompt, citation and abstention rules
  eval.py               scoring: recall@1/@3, MRR, faithfulness, abstention
  report.py             rendering the results as a table
  observe.py            per-stage tracing, per-model cost, latency percentiles
  store.py, index.py    the SQLite index and how it is built
ingest/                 published docs → cleaned, heading-aware, citable chunks
demo/                   precomputed answers for a keyless public demo
```

That inversion is not decoration: it is why the entire
retrieval → generate → judge → aggregate path runs under test with **no API key
and no network**. The fakes in `tests/` are simply other implementations of the
same four Protocols, and no policy module can tell the difference.

## Running it

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv -e '.[dev]'
printf 'VOYAGE_API_KEY=...\nANTHROPIC_API_KEY=...\n' > .env   # gitignored

python -m ingest all              # fetch docs        → data/chunks.jsonl
python -m rag index               # embed             → data/index.db (~20s, 1,290 chunks)
python -m rag ask "how do I cache a prompt?" -k 3
```

The demo UI runs without keys or an index, serving precomputed answers:

```bash
pip install -r requirements.txt   # just streamlit
streamlit run app.py
```

### Tests

```bash
python -m pytest -q               # 67 tests, no key, no network
```

## Corpus and licensing

The corpus is the Claude developer documentation, fetched from
`https://platform.claude.com/llms.txt`; each page is published as clean Markdown
at its `.md` URL, so there is no HTML scraping. Scope is the core feature and
tool-use documentation (`ingest/scope.py`); the per-language API reference
(~1,429 pages) is excluded to keep the corpus in the hundreds-to-few-thousand
chunk range.

The code is [MIT-licensed](LICENSE). The documentation text is Anthropic's
copyrighted material — this project **fetches it for local retrieval and does not
redistribute it**. The fetched Markdown, `chunks.jsonl`, and `index.db` are all
git-ignored, and the demo artifact ships **no corpus body** (only model-written
cited prose plus source URLs, enforced by a test in `demo/render.py`). Only code,
the URL manifest, the evaluation set, and metrics are versioned.

## Built on this

[incident-triage-agent](https://github.com/Hal-Hanami/incident-triage-agent)
reuses the `rag` package here as the runbook search behind a read-only incident
triage agent on the Claude Agent SDK. It carries the same posture one rung up the
autonomy ladder: where this system declines to *answer* what its sources do not
support, that one declines to *act* when confidence is low, and escalates to a
human instead.

## Contributing

Commit and comment conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).
