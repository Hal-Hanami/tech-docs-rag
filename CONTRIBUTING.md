# Contributing

## Commit messages

This repository uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <summary in the imperative, lower case, no trailing period>

<body: why this change is needed, and anything a reader would otherwise
have to reconstruct from the diff>
```

**Types**

| Type | Use for |
|---|---|
| `feat` | new capability |
| `fix` | corrected behaviour |
| `docs` | documentation only |
| `test` | tests only |
| `refactor` | restructuring with no behavioural change |
| `perf` | measured performance or cost improvement |
| `build` | packaging, dependencies, CI |
| `chore` | housekeeping that fits nothing above |

**Scopes** follow the package layout: `ingest`, `store`, `retrieval`, `generate`,
`eval`, `report`, `observe`, `cli`, `demo`, `clients`.

```
feat(retrieval): fuse BM25 candidates into the dense ranking with RRF
fix(cli): stop --dense-only from implying --no-rerank
docs(readme): correct RRF-only MRR to the re-measured 0.835
test(ingest): cover the scope include/exclude rules
refactor(eval): move report rendering out of the scoring module
```

**The body explains why.** The diff already shows what changed; a reader a year
from now needs the reason. Constraints discovered, alternatives rejected, and
measurements that motivated the change all belong here.

**Keep it about the system.** Commit messages describe the software, not the
process that produced it and not the author's circumstances. If a sentence would
not make sense to someone who has never met the author, it belongs in a personal
note rather than in the history of a public repository.

## Code comments

Comments explain **why**, not what. The code already states what it does, and a
comment restating it will drift out of date and start lying.

Worth a comment: a constraint that is not visible locally, a non-obvious ordering
requirement, a rejected alternative, a value that came from a measurement.
Not worth a comment: anything a reader can get from the line itself.

Module and function docstrings carry the reasoning; inline `#` comments are for
the specific line that would otherwise make a reader stop and squint.

## Tests

Every pure function is expected to have direct tests. Network boundaries sit
behind the Protocols in `rag/ports.py`, so the retrieval, generation, scoring,
and reporting paths are all exercised offline with fakes — a test that needs an
API key or a network connection is a sign that policy and adapter have become
entangled.

```bash
python -m pytest -q
```

Tests must pass without `VOYAGE_API_KEY` or `ANTHROPIC_API_KEY` set.

### Five rules the build enforces

Most defects this project has shipped were not coding errors. They were
**invariants nobody had written down** — that an answer may contain code and code
contains brackets; that `rag` is an interface another repository imports; that the
two ablation flags are independent. When a rule is unwritten, the implementation
and its tests get written from the same missing understanding, and the tests agree
with the bug. So the rules below are checked by CI, not by good intentions.

1. **Every numbered section of [`docs/DESIGN.md`](docs/DESIGN.md) is pinned by a
   test.** That document states what must be true; a test naming the section is
   what keeps it true. A section nothing cites is a rule the next change will break
   without noticing. Citing `§4.3` counts as citing `§4`.
2. **No module sits at 0% coverage.** A file nothing imports is a file nothing
   checks. Entry points count: `python -m rag` is a documented command, so
   something must prove it resolves.
3. **A published number is tied to something that produces it.** Either a
   reproduction command printed beside it in
   [`docs/EVALUATION.md`](docs/EVALUATION.md), or a test that re-derives it —
   `tests/test_published_numbers.py` reads the cost table out of that file and
   rebuilds it from `rag/observe.py`, so a rate change cannot quietly make it wrong.
   The README quotes a few headline figures; those are checked against the record
   for the same reason, as is the test count.
4. **`tests/test_public_api.py` is the contract with other repositories.** Anything
   another project imports from `rag` is pinned there. Moving it is fine; moving it
   without updating that file in the same commit is how a consumer breaks silently.
5. **A restructuring must be shown not to change behaviour.** Parse both trees,
   strip docstrings, and compare the moved functions with `ast.dump`. Identical
   output is evidence; "it was only a move" is not. What survives that check can be
   committed as `refactor`; what does not is a `feat` or a `fix` and needs its own
   reasoning in the body.

### Where a sentence belongs

One concern per file. When in doubt: **the reason one module needs to be read**
goes in its docstring; **a promise that spans modules** goes in `docs/DESIGN.md`
with a section number; **a number that came out of a run** goes in
`docs/EVALUATION.md` with its date and reproduction command. The README is the
entry point and quotes the other three rather than repeating them.
