"""Stage 1 — parse llms.txt into the in-scope corpus URL list.

`https://platform.claude.com/llms.txt` is a Markdown sitemap whose links already
carry the `.md` suffix, e.g.

    - [Streaming](https://platform.claude.com/docs/en/build-with-claude/streaming.md) - ...

so we just parse the links, keep English docs pages, apply the M1 scope
(`scope.in_scope`), dedupe, and emit a plain URL list. No `.md` needs to be
appended — the links are already in `.md` form.
"""

from __future__ import annotations

import re
from collections import Counter
from urllib.request import Request, urlopen

from . import scope

LLMS_TXT_URL = "https://platform.claude.com/llms.txt"
_LINK_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
_USER_AGENT = "tech-docs-rag/0.1 (personal RAG ingest; +https://platform.claude.com/llms.txt)"


def fetch_llms_txt(url: str = LLMS_TXT_URL, timeout: float = 30.0) -> str:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_links(text: str) -> list[str]:
    """All `.md` URLs referenced as Markdown links, in order, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for url in _LINK_RE.findall(text):
        url = url.rstrip(".,)")  # tolerate trailing punctuation
        if not url.endswith(".md"):
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def in_scope_urls(text: str) -> list[str]:
    """In-scope llms.txt links, plus the verified pages missing from the manifest."""
    urls = [u for u in parse_links(text) if scope.in_scope(u)]
    urls.extend(scope.EXTRA_URLS)
    return sorted(set(urls))


def category_breakdown(urls: list[str]) -> Counter[str]:
    """Top-level category -> page count, for eyeballing that scope is sane."""
    counts: Counter[str] = Counter()
    for u in urls:
        path = scope.page_path(u) or ""
        top = path.split("/", 1)[0] if path else "(root)"
        counts[top] += 1
    return counts


def all_en_category_breakdown(text: str) -> Counter[str]:
    """Category -> count across *all* en docs (ignoring scope) for transparency.

    Lets us see what categories exist so the scope allowlist can be checked
    against reality without re-reading the whole llms.txt by hand.
    """
    counts: Counter[str] = Counter()
    for u in parse_links(text):
        path = scope.page_path(u)
        if path is None:
            continue
        top = path.split("/", 1)[0]
        counts[top] += 1
    return counts
