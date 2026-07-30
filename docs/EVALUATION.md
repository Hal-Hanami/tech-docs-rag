# Evaluation

Every number this repository claims, with the date it was measured, the set it was
measured over, and the command that reproduces it. Where a number is weak, noisy,
or an artifact of the corpus, it says so here rather than in a footnote.

This file is the **record**. [`DESIGN.md`](DESIGN.md) is the spec that says what
was supposed to happen; a decision belongs there, a measurement belongs here.

The offline suite (no key, no network) covers the logic that produces these
numbers — see [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for what the build
enforces. The figures below are the part that needs a paid run.

## The question set

`eval/qa.jsonl` — authored questions with the pages that should answer them,
versioned so a run is reproducible. Recall is scored at page level (§7.1).

| slice | in-corpus | out-of-corpus | total | what it stresses |
|---|---|---|---|---|
| `core` | 22 | 4 | 26 | general questions across the corpus, plus the refusal cases |
| `hard` | 12 | 0 | 12 | keyword-exact lookups and near-sibling pages |

The four out-of-corpus questions have no right answer; the correct behaviour is a
refusal (§5). They are why the cost run below covers a **core slice, 26 q** while
the rank metrics cover 22.

## Retrieval quality

Measured **2026-07-30**, one session, one index. All eight cells were re-run and
every published figure reproduced exactly.

| slice | metric | dense | hybrid (RRF only) | **hybrid + rerank** | dense + rerank |
|---|---|---|---|---|---|
| **core** (22 q) | recall@1 | 86.4 % | 72.7 % | **86.4 %** | 86.4 % |
| | recall@3 | 100 % | 95.5 % | **100 %** | 100 % |
| | MRR | 0.932 | 0.835 | **0.932** | 0.932 |
| **hard** (12 q) | recall@1 | 66.7 % | 58.3 % | **91.7 %** | 91.7 % |
| | recall@3 | 100 % | 100 % | **100 %** | 100 % |
| | MRR | 0.833 | 0.764 | **0.944** | 0.944 |

`recall@5` is 100 % in every configuration — the right page is essentially always
*in* the pool — so the reranker earns its keep on **rank**, which is what the
generator actually reads (§7.2).

Reproduce any column. No LLM calls and no Anthropic key; the first two need no
Voyage spend either, since they never call the reranker:

```bash
python -m rag eval --tag core --retrieval-only --dense-only --no-rerank  # dense
python -m rag eval --tag core --retrieval-only --no-rerank               # RRF only
python -m rag eval --tag core --retrieval-only                           # product
python -m rag eval --tag core --retrieval-only --dense-only              # dense + rerank
```

Three findings, all from the harness:

- **Plain RRF *hurts* rank.** On a corpus where dense retrieval is already strong,
  fusing in BM25's keyword hits pushes the right page off rank 1 — 86.4 → 72.7 % on
  core, 66.7 → 58.3 % on hard. **The reranker is what makes hybrid safe:** it
  recovers that loss on both slices, back to parity on core and past the baseline
  on hard.
- **Reranking pays off on hard queries.** Exact-keyword lookups (`bash_20`,
  `str_replace`, `pause_turn`) and sibling-page disambiguation lift **recall@1 from
  66.7 % to 91.7 %**. On the easier core slice dense retrieval is already at the
  ceiling, so the win appears exactly where the harder questions are.
- **Once you rerank, the BM25 half earns nothing here.** Reranking dense results
  alone scores *identically* to the full hybrid on every metric above. On this
  corpus the measured win belongs entirely to the cross-encoder, not to the fusion.
  BM25 stays in because it is nearly free insurance against a query the embedder
  misses, but the honest reading is: **this table does not demonstrate value from
  RRF.**

**Run-to-run stability.** The dense and product columns are stable. The RRF-only
column is the one that moves: core MRR returned 0.835 here against 0.827 first
recorded, and recall@3 95.5 % rather than 100 %. Both are a single question
crossing the rank-3/4 boundary (1 ÷ 22 ≈ 0.008 of MRR). The un-reranked fusion
sits on a knife edge; the configurations that carry claims do not.

## Generation cost

Measured **2026-07-30**, both runs in one session against one index, $1.24 total.

The per-model trace showed generation at ~90 % of spend, which pointed at one
lever: how much context the generator receives. Reranking saturates recall at rank
3, so chunks 4 and 5 cost tokens without adding coverage. Cutting the generation
context from **k = 5 to k = 3**:

| metric (evaluation set, core slice, 26 q, one session, 2026-07-30) | k = 5 | k = 3 | Δ |
|---|---|---|---|
| recall@1 / recall@3 / MRR | 86.4 % / 100 % / 0.932 | **identical** | **0** |
| abstention / false abstentions | 100 % / 0 | 100 % / 0 | 0 |
| generation input tokens | 88,499 | 54,197 | **−39 %** |
| generation output tokens | 9,619 | 9,963 | +4 % |
| **total cost per query** | $0.0270 | **$0.0207** | **−23 %** |

```bash
python -m rag eval --tag core --no-judge        # k = 5
python -m rag eval --tag core --no-judge -k 3   # k = 3
```

**The cost falls by less than the tokens do, and the gap is the interesting part.**
Cutting `k` removes 39 % of the *input* tokens but only 23 % of the bill, because
two components do not shrink with it: the answers get slightly *longer* with less
context to quote from (+4 % output tokens), and the reranker scores the full
candidate pool regardless of `k`, so its $0.0193 is identical in both runs (§3.3).
A token count is not a cost, and quoting the token saving as if it were the cost
saving would overstate this result by 16 points.

The quality claim rests on the deterministic metrics, which are unchanged — not on
faithfulness, which is too noisy to carry it (§7.5).

## Abstention

On the same run: **abstention rate 100 %** — all four out-of-corpus questions were
declined — with **0 false abstentions**, meaning no in-corpus question was wrongly
refused (§5.4). Unchanged at both `k = 5` and `k = 3`.

This is the deterministic half of the hallucination claim. Because the refusal is
one exact sentence (§5.1), it is detectable, and because it is detectable it is
countable.

## Corrections

Kept rather than quietly edited, because a record that only ever agreed with itself
is not evidence of anything.

- **Cost saving: −28 % → −23 %.** The first figure came from confusing the token
  reduction with the cost reduction. Re-measurement gave −23 %, and §3.3 explains
  why the two differ by 16 points.
- **The latency claim is withdrawn.** An earlier run recorded p95 13.0 s → 10.2 s
  and read that as a win. Re-measuring gave 10.41 s → 10.32 s — no movement. p95
  over 26 questions is essentially the second-slowest question, so it moves with
  whatever the API was doing that afternoon. Trimming `k` buys cost, not speed.
- **Test count: the documentation used to state a number nobody checked.** The
  build now fails if the count quoted anywhere disagrees with the count that ran.

## The faithfulness judge

**±20 points noisy run to run** — the same pipeline scored 86.4 % one day and
63.6 % the next. This project therefore treats faithfulness as a *direction*, never
a headline, and rests every claim on the deterministic metrics (§7.5).

## What these numbers do not show

- **Generalization beyond this corpus.** Every figure is measured on one body of
  documentation with one embedding model. The finding that RRF does not help is a
  finding about *this* corpus, where dense retrieval is already strong.
- **A question set anyone else authored.** The evaluation set is written by the
  same person who built the system. It is versioned and reproducible, which makes
  it honest, not unbiased.
- **The live error paths.** Retry and backoff against a failing API are not
  exercised by any measurement here.
- **Answer quality as a reader would judge it.** Rank metrics say the right page
  was retrieved; abstention says the system knew when to stop. Neither says the
  prose was good.
