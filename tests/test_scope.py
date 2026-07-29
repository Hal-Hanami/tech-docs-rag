"""Tests for corpus scope — which documentation pages are in and which are out.

Scope is the decision that keeps this project a single deployable service: the
source `llms.txt` lists ~1,557 English pages, and ~1,429 of them are the
per-language API reference. Letting those in would multiply the corpus without
adding distinct content, so the exclusion rules below are load-bearing rather
than cosmetic, and worth asserting.
"""

from __future__ import annotations

from ingest import scope


def test_page_path_strips_root_and_md_suffix():
    url = scope.DOCS_ROOT + "build-with-claude/streaming.md"
    assert scope.page_path(url) == "build-with-claude/streaming"


def test_page_path_rejects_urls_outside_the_docs_root():
    assert scope.page_path("https://example.com/build-with-claude/streaming.md") is None


def test_page_path_returns_none_for_the_bare_root():
    """The root is not a page; returning "" would make it compare equal to nothing
    and quietly slip through the `path in INCLUDE_PAGES` check."""
    assert scope.page_path(scope.DOCS_ROOT) is None


def test_included_prefix_is_in_scope():
    assert scope.in_scope(scope.DOCS_ROOT + "build-with-claude/prompt-caching.md")
    assert scope.in_scope(scope.DOCS_ROOT + "agents-and-tools/tool-use/overview.md")


def test_explicitly_listed_page_is_in_scope():
    """Pages named in INCLUDE_PAGES sit outside every include prefix, so they are
    in scope only because they are listed by name."""
    assert scope.in_scope(scope.DOCS_ROOT + "get-started.md")
    assert scope.in_scope(scope.DOCS_ROOT + "about-claude/pricing.md")


def test_api_reference_is_excluded():
    """The single most important rule: the per-language endpoint reference is the
    bulk of the source site and must stay out of the corpus."""
    assert not scope.in_scope(scope.DOCS_ROOT + "api/messages.md")
    assert not scope.in_scope(scope.DOCS_ROOT + "api/client-sdks.md")


def test_exclusion_matches_a_whole_path_segment_not_a_prefix():
    """`api` must not exclude a hypothetical `apis/...` category.

    The check compares the first path segment exactly; a `startswith` would make
    the exclusion silently broader than intended.
    """
    assert scope.page_path(scope.DOCS_ROOT + "apis/overview.md") == "apis/overview"
    # Not excluded by the "api" rule — it simply isn't in any include list either.
    assert not scope.in_scope(scope.DOCS_ROOT + "apis/overview.md")


def test_unlisted_page_is_out_of_scope():
    assert not scope.in_scope(scope.DOCS_ROOT + "release-notes/whatever.md")


def test_url_outside_the_docs_root_is_out_of_scope():
    assert not scope.in_scope("https://example.com/anything.md")


def test_every_extra_url_would_also_pass_the_scope_rules():
    """EXTRA_URLS are injected into the manifest because they are missing from
    `llms.txt`, not because they are exceptions to the rules. If one stops
    satisfying `in_scope`, the two definitions of "in the corpus" have drifted.
    """
    for url in scope.EXTRA_URLS:
        assert url.startswith(scope.DOCS_ROOT), url
        assert url.endswith(".md"), url
        assert scope.in_scope(url), url
