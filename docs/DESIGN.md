# tech-docs-rag — design

> This document is the **spec**: the properties the system must hold and why.
> [`EVALUATION.md`](EVALUATION.md) is the **record** of what measuring them
> returned, including where the numbers are weak. Nothing here is a measurement;
> nothing there is a decision.
>
> Numbered sections state invariants, and code and tests cite them (`design §4.3`).
> Every `§N` below is pinned by at least one test — CI fails otherwise, because a
> written rule nobody checks is a rule the next change will quietly break.
> Sections without a number are context, not contract.

## Purpose

Answer questions about a body of published technical documentation, with a
citation to the exact section behind every claim — or an explicit refusal when
the documentation does not support an answer.

Retrieval-augmented generation is a commodity; the engineering around it is not.
Three properties carry this system, and each is a numbered section below:

| Property | Where | The claim |
|---|---|---|
| Retrieval you can measure | §3, §7 | rank metrics on a versioned question set, per configuration |
| Answers that decline | §4, §5 | grounded or refused, and the refusal is detectable so it can be counted |
| Cost you can explain | §6 | tokens and dollars split per model, on every request |

---

## §1 Architecture — dependencies point inward

Policy declares the interfaces it needs; the adapters conform to them.

```
rag/
  cli.py, commands.py   flags in, wiring out — the only layer that knows both
                        which client implements what and which policy wants it
  ports.py              the four seams: Embedder, Reranker, Answerer, Judge
  clients/              the only modules that open a network connection
  search, generate,     pure policy — retrieval strategy, prompt assembly,
  eval, report, observe scoring, rendering, tracing
  store.py, index.py    the SQLite index and how it is built
ingest/                 published docs → cleaned, heading-aware, citable chunks
demo/                   precomputed answers for a keyless public demo
```

**§1.1** Only modules under `rag/clients/` may open a network connection. Every
other module takes what it needs as one of the four Protocols in `rag/ports.py`.

**§1.2** The whole retrieval → generate → judge → aggregate path must run with no
API key and no network. The fakes in `tests/` are other implementations of the
same Protocols, and no policy module can tell the difference.

This is not decoration: §1.2 is what makes the scoring logic testable at all, and
it is why a test that needs a key is a sign that policy and adapter have become
entangled.

## §2 Corpus and licensing

The corpus is published documentation fetched from its own `.md` URLs, so there is
no HTML scraping. Scope is the core feature and tool-use documentation; the
per-language API reference is excluded — it is the large majority of the pages and
adds no distinct content, so admitting it would multiply the corpus for nothing.

**§2.1** The documentation text is not ours to redistribute. The fetched Markdown,
the chunked corpus, and the built index are all untracked. Only code, the URL
manifest, the evaluation set, and measurements are versioned.

**§2.2** No published artifact may carry corpus body text. The demo ships section
paths and URLs only (§8), and this is enforced rather than remembered.

**§2.3** Every chunk keeps the citation trail it was built with — source page,
section path, and a deep-link anchor — so an answer can name where a claim came
from rather than merely that it came from somewhere.

## §3 Retrieval

```
question → embed → dense kNN (sqlite-vec) ∪ BM25 (FTS5) → RRF fuse → rerank → top-k
```

One SQLite file is the entire datastore: dense vectors, the keyword index, and the
citation metadata live together, so the system deploys as a single service with no
additional infrastructure.

**§3.1** Fusion is rank-based (Reciprocal Rank Fusion). Cosine distances and BM25
scores are on unrelated scales; combining them by rank avoids inventing a
normalization that would need its own justification.

**§3.2** The two ablation toggles are **independent**. `--dense-only` removes the
BM25 half; `--no-rerank` removes the reranker; neither implies the other, and the
un-reranked baseline is both together. Each run prints the configuration it
actually used, so a reader cannot mistake one ablation for another.

> This was once false in a way nothing caught: the help text described
> `--dense-only` as the un-reranked baseline while the code still passed a
> reranker, and a reproduction command in the documentation therefore reported a
> configuration nobody had run.

**§3.3** The reranker scores the whole candidate pool, and the pool size does not
depend on `k`. **Reranking cost is therefore independent of `k`** — which is why
cutting `k` saves less money than it saves tokens (§6.3).

**§3.4** A query that produces no keyword tokens yields no BM25 rows rather than an
error. Query text is turned into quoted terms before it reaches FTS5, so
punctuation cannot reach the query parser.

## §4 Grounding and citations

**§4.1** The model receives the retrieved sources and nothing else. The system
prompt forbids outside knowledge and requires a `[n]` citation on every claim.

**§4.2** Source **URLs are withheld from the model** and re-attached to each number
at print time. A model that never sees a link cannot fabricate one, and citation
integrity is the property this system exists to provide.

**§4.3** **An answer may contain code, and code contains brackets.** The corpus is
API documentation, so `response.content[1].text` is ordinary prose here. A `[n]`
inside a fenced block or an inline span is a subscript, not a citation, and must
be left exactly as written. Anything that reads citations — rendering them as
links, or checking that they resolve — reads them through one shared helper, so
the two cannot disagree about what counts as a citation.

**§4.4** Retrieval and generation see the same evidence. A caller that has already
retrieved passes those results to generation rather than searching again, so what
is scored for rank is what the answer was grounded in.

## §5 Abstention

**§5.1** When the sources do not support an answer, the model replies with one
**exact sentence** and nothing else.

**§5.2** That sentence is compared with `==`, never a substring or case-insensitive
match. Abstention has to be a *detectable* outcome to be a countable one, and a
fuzzy check would score a hedged answer ("I don't know if that's right, but…") as a
clean refusal.

**§5.3** Empty retrieval short-circuits to a refusal. There is nothing to ground
on, and asking the model anyway invites it to answer from memory.

**§5.4** Refusing an in-corpus question is a **false abstention** and is counted
separately from correct refusals. Collapsing the two would let a system that
declines everything look perfect.

## §6 Observability and cost

**§6.1** Every request carries a trace: one span per stage, and token usage filed
under the model that produced it. Attribution happens at the point of spend, so
the per-model split is real rather than reconstructed afterwards.

**§6.2** Token counts are authoritative — each API returns them. Dollars are an
estimate from a rate table that records **where each rate came from and when it was
checked**. A model absent from the table contributes zero rather than a guess.

**§6.3** Cost is split by model because that split is the optimization lever. Most
spend is generation, which is what makes the generation context size the thing
worth tuning — and §3.3 is why the saving is smaller than the token reduction.

**§6.4** **An error names the boundary that failed.** The two network endpoints
share one HTTP helper, so without an explicit label a failure reads identically
whether embedding or reranking broke — and which one broke is the first thing
anyone needs to know.

**§6.5** "Not measured" and "measured as zero" never render the same way. A
retrieval-only run cannot observe abstention, so it reports `n/a`; printing `0.0%`
would claim the system failed to decline a single out-of-corpus question.

## §7 Evaluation

The evaluation set is versioned alongside the code: authored questions plus the
pages that should answer them, sliced so configurations can be compared without
averaging away the effect being measured.

**§7.1** Recall is scored at **page** level. A question answered from a different
chunk of the right page is a success, not a miss.

**§7.2** Rank is the metric that matters. Recall at the full `k` saturates — the
right page is essentially always *in* the pool — so what distinguishes
configurations is whether it lands where the generator will read it.

**§7.3** One retrieval per question, reused for both rank and generation, so the
two are measured against identical evidence (§4.4).

**§7.4** Only a real answer to an in-corpus question is judged for faithfulness. An
abstention has no claims to check, and an out-of-corpus question has no right
answer to be faithful to.

**§7.5** Claims rest on the **deterministic** metrics — recall, MRR, abstention
rate, false abstentions. Faithfulness is an LLM judgement and too noisy run to run
to carry a claim; it is reported as a direction, never as a headline. Knowing which
of your own metrics you cannot trust is part of the work.

## §8 The published demo

The public deployment serves **precomputed** answers: no LLM call, no key, no
index. It costs nothing to run continuously and redistributes none of the corpus.

**§8.1** The baked artifact carries model-written cited prose, section paths, URLs,
and the measured trace — and **no corpus body** (§2.2).

**§8.2** The artifact's contract is **enforced where it is produced**, not only
where it is consumed. The baker validates before writing; a bake that violates the
contract leaves the published file untouched and keeps its own output aside for
inspection. A run costs real money, so it is preserved rather than discarded.

**§8.3** The contract covers everything the application reads without guarding —
the caption fields, source numbering, citation resolution, and the trace fields the
metrics row and cost table render. A missing key is a traceback in front of a
visitor, not a test failure.

## §9 The `rag` package as a published interface

**§9.1** The module paths, call signatures, and result fields that another project
imports are a **published interface**, pinned by `tests/test_public_api.py`. Moving
any of it is allowed; moving it without updating that file in the same commit is
not.

> This was once unwritten, and the cost was concrete: two modules were folded into
> a new package, and a downstream consumer's live path stopped resolving while both
> test suites stayed green. A stub of this package answers whatever the caller asks
> and cannot notice that the real thing moved — so the check belongs here, on the
> side that owns the interface.

## Boundaries and non-goals

- No re-ranking of the corpus itself, no fine-tuning, no multi-hop reasoning.
- No agentic behaviour: this system answers questions, it does not take actions.
- No authentication, multi-tenancy, or persistence beyond the index file.
- The demo is a demonstration surface, not a product UI.

## Honesty notes

- The reranker's measured win is real; the RRF fusion's is not demonstrated on this
  corpus (see `EVALUATION.md`). The fusion stays because it is nearly free
  insurance against a query the embedder misses, not because the numbers argue
  for it.
- The faithfulness judge is too noisy to support a claim (§7.5).
- Latency is not part of any claim here; the sample is small enough that the
  percentiles move with whatever the API was doing that afternoon.
