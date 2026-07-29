"""Which documentation pages belong in the corpus.

The Claude docs `llms.txt` lists 1,557 English pages, but 1,429 of those are the
per-language API reference (Python / TypeScript / Go / ...). M1 deliberately
narrows to the *core* "Build with Claude" feature docs plus the tool-use family,
so the corpus lands in the design's "hundreds–few thousand chunks" target and
stays a single, deployable service.

Scope is expressed as path prefixes relative to the docs root
`https://platform.claude.com/docs/en/`. Keeping it as data (not code) means the
scope can be reviewed and adjusted without touching the pipeline.

Note on the manifest gap: the conceptual model/pricing pages (Models overview,
Pricing, Glossary, ...) live under `about-claude/` and are NOT listed in the
developer `llms.txt`, yet they resolve as clean `.md` and are squarely in scope.
They are added explicitly via EXTRA_URLS (each verified to return 200).
"""

from __future__ import annotations

DOCS_ROOT = "https://platform.claude.com/docs/en/"

# Whole categories to include. Verified to exist via llms.txt.
INCLUDE_PREFIXES: tuple[str, ...] = (
    "build-with-claude/",          # Messages, streaming, prompt caching, thinking,
                                   # effort, structured outputs, batch, files, tokens...
    "agents-and-tools/tool-use/",  # tool use family
    "about-claude/models/",        # models overview / comparison / migration (via EXTRA_URLS)
)

# Specific extra pages inside DOCS_ROOT to include even though they sit outside
# the prefixes above (path relative to DOCS_ROOT, without the `.md` suffix).
INCLUDE_PAGES: tuple[str, ...] = (
    "get-started",
    "intro",
    "about-claude/glossary",
    "about-claude/pricing",
    "about-claude/model-deprecations",
)

# Hard excludes — these win over the includes above. The per-language API
# reference is the big one to keep out: huge, and duplicated per language.
EXCLUDE_PREFIXES: tuple[str, ...] = (
    "api",            # per-language endpoint reference (huge, duplicated)
    "managed-agents",
    "manage-claude",
)

# High-value pages that are in scope but absent from llms.txt. Each verified
# to return 200 as `.md` on 2026-06-08. Injected into the manifest in manifest.py.
EXTRA_URLS: tuple[str, ...] = (
    DOCS_ROOT + "about-claude/models/overview.md",
    DOCS_ROOT + "about-claude/models/choosing-a-model.md",
    DOCS_ROOT + "about-claude/models/migration-guide.md",
    DOCS_ROOT + "about-claude/models/model-ids-and-versions.md",
    DOCS_ROOT + "about-claude/model-deprecations.md",
    DOCS_ROOT + "about-claude/pricing.md",
    DOCS_ROOT + "about-claude/glossary.md",
)


def page_path(url: str) -> str | None:
    """Return the docs path (relative to DOCS_ROOT, no `.md`) for an en docs URL.

    Returns None for URLs that are not English docs pages.
    """
    if not url.startswith(DOCS_ROOT):
        return None
    path = url[len(DOCS_ROOT):]
    if path.endswith(".md"):
        path = path[: -len(".md")]
    return path.strip("/") or None


def in_scope(url: str) -> bool:
    """True if `url` is an English docs page inside the M1 corpus scope."""
    path = page_path(url)
    if path is None:
        return False
    top = path.split("/", 1)[0]
    if top in EXCLUDE_PREFIXES:
        return False
    if any(path.startswith(prefix) for prefix in INCLUDE_PREFIXES):
        return True
    return path in INCLUDE_PAGES
