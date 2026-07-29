"""Tests for manifest parsing — `llms.txt` markdown sitemap -> corpus URL list.

Parsing is pure and network-free; only `fetch_llms_txt` touches the network and
it is not exercised here. Everything below feeds the parser a literal document,
which is also how the awkward cases (trailing punctuation, duplicates, non-`.md`
links) stay pinned down.
"""

from __future__ import annotations

from ingest import manifest, scope

SAMPLE = f"""\
# Claude Docs

- [Streaming]({scope.DOCS_ROOT}build-with-claude/streaming.md) - Stream responses
- [Tool use]({scope.DOCS_ROOT}agents-and-tools/tool-use/overview.md) - Tools
- [Messages API]({scope.DOCS_ROOT}api/messages.md) - Endpoint reference
- [Streaming again]({scope.DOCS_ROOT}build-with-claude/streaming.md) - duplicate
- [Home]({scope.DOCS_ROOT.rstrip('/')}) - not a .md link
- [External](https://example.com/page.md) - outside the docs root
"""


def test_parse_links_keeps_only_md_urls_in_document_order():
    links = manifest.parse_links(SAMPLE)
    assert links[0] == scope.DOCS_ROOT + "build-with-claude/streaming.md"
    assert links[1] == scope.DOCS_ROOT + "agents-and-tools/tool-use/overview.md"
    assert all(u.endswith(".md") for u in links)


def test_parse_links_dedupes_while_preserving_first_position():
    links = manifest.parse_links(SAMPLE)
    streaming = scope.DOCS_ROOT + "build-with-claude/streaming.md"
    assert links.count(streaming) == 1
    assert links.index(streaming) == 0


def test_parse_links_keeps_urls_outside_the_docs_root():
    """Parsing and scoping are separate steps: the parser reports every `.md`
    link it sees, and `in_scope_urls` is what narrows the set. Filtering here
    too would hide out-of-scope links from the category breakdown."""
    assert "https://example.com/page.md" in manifest.parse_links(SAMPLE)


def test_parse_links_strips_trailing_punctuation():
    text = f"See [Streaming]({scope.DOCS_ROOT}build-with-claude/streaming.md),"
    assert manifest.parse_links(text) == [scope.DOCS_ROOT + "build-with-claude/streaming.md"]


def test_in_scope_urls_drops_out_of_scope_pages():
    urls = manifest.in_scope_urls(SAMPLE)
    assert scope.DOCS_ROOT + "build-with-claude/streaming.md" in urls
    assert scope.DOCS_ROOT + "api/messages.md" not in urls
    assert "https://example.com/page.md" not in urls


def test_in_scope_urls_injects_the_pages_missing_from_the_manifest():
    """The conceptual model/pricing pages resolve as clean `.md` but are absent
    from `llms.txt`, so they are added regardless of what the document lists."""
    urls = manifest.in_scope_urls("")
    assert set(scope.EXTRA_URLS) <= set(urls)


def test_in_scope_urls_is_sorted_and_deduped():
    """A stable, deduped list keeps the downstream fetch cache from depending on
    the order links happened to appear in."""
    urls = manifest.in_scope_urls(SAMPLE + SAMPLE)
    assert urls == sorted(set(urls))


def test_category_breakdown_counts_by_top_level_segment():
    counts = manifest.category_breakdown(manifest.in_scope_urls(SAMPLE))
    assert counts["build-with-claude"] == 1
    assert counts["agents-and-tools"] == 1
    assert "api" not in counts  # excluded before it reaches the breakdown


def test_all_en_category_breakdown_ignores_scope():
    """The unscoped breakdown is how the include/exclude lists get checked
    against what the site actually publishes, so it must count the excluded
    categories too."""
    counts = manifest.all_en_category_breakdown(SAMPLE)
    assert counts["api"] == 1
    assert counts["build-with-claude"] == 1
