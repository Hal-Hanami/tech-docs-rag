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
