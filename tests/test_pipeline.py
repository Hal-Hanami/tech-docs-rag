"""Offline tests for clean + chunk (no network).

The fixture is hand-authored Markdown (not Anthropic docs) so it can be
committed without redistributing the corpus, while still exercising every MDX
pattern the cleaner must handle.
"""

from __future__ import annotations

from ingest.chunk import chunk_page, slugify
from ingest.clean import clean

FIXTURE = """---
title: Sample page
description: frontmatter that must be stripped
---

# Sample Page

Intro paragraph with a <Tooltip tooltipContent="hidden hint">visible term</Tooltip> in it.

## Using widgets

<Note>Keep this sentence; drop the tags.</Note>

A line with a break<br/>and a footnote<sup>1</sup>.

| Feature | Value |
|---------|-------|
| A       | 1     |
| B       | 2     |

```python
# code with comparisons that must NOT be tag-stripped
if a < b and c > d:
    print("<not a tag>")
```

<section title="Legacy notes">

This section's title should become a heading.

</section>
"""

URL = "https://platform.claude.com/docs/en/build-with-claude/sample.md"


def test_clean_removes_mdx_keeps_inner_text():
    out = clean(FIXTURE)
    assert "<Tooltip" not in out and "</Tooltip>" not in out
    assert "visible term" in out
    assert "<Note>" not in out and "Keep this sentence; drop the tags." in out
    assert "<sup>" not in out and "<br" not in out


def test_clean_preserves_table_and_code():
    out = clean(FIXTURE)
    assert "| Feature | Value |" in out
    assert 'if a < b and c > d:' in out          # comparisons survive
    assert 'print("<not a tag>")' in out          # angle brackets in code survive


def test_clean_promotes_section_title_to_heading():
    out = clean(FIXTURE)
    assert "## Legacy notes" in out
    assert "<section" not in out


def test_frontmatter_stripped():
    out = clean(FIXTURE)
    assert "description: frontmatter" not in out


def test_chunk_metadata_and_links():
    chunks = chunk_page(clean(FIXTURE), URL)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c["source_url"] == URL[:-3]
        assert c["page_title"] == "Sample Page"
        assert c["section_path"].startswith("Sample Page")
        assert c["token_estimate"] >= 1
        assert c["text"].strip()
        if c["anchor"]:
            assert c["url"].endswith("#" + c["anchor"])
    # ids are unique and indexed
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_slugify():
    assert slugify("How to use **fine-grained** streaming") == "how-to-use-fine-grained-streaming"
    assert slugify("Choosing the right model!") == "choosing-the-right-model"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
