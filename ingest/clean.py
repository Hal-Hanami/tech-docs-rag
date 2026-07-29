"""MDX/JSX cleanup of a fetched `.md` page.

The docs `.md` is mostly clean Markdown but still carries MDX components
(`<Note>`, `<Tooltip>`, `<Warning>`, `<section title="...">`, `<Card>`, `<sup>`,
`<br/>`, ...). We strip the component *tags* while keeping their inner text, so
the prose a reader sees survives. Markdown tables and fenced code are preserved
verbatim (a naive `<...>` strip would corrupt code containing generics/comparisons,
so code and inline-code are masked before tag removal and restored afterwards).
"""

from __future__ import annotations

import re

# --- patterns ---------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCED_CODE_RE = re.compile(r"```.*?\n.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MDX_EXPR_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)

# <section title="Legacy models"> -> promote the title to a heading so the
# chunker still sees the structural boundary.
_SECTION_OPEN_RE = re.compile(r'<section\b[^>]*\btitle="([^"]*)"[^>]*>', re.IGNORECASE)
_SECTION_OPEN_NOTITLE_RE = re.compile(r"<section\b[^>]*>", re.IGNORECASE)
_SECTION_CLOSE_RE = re.compile(r"</section\s*>", re.IGNORECASE)

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Capitalised JSX component tags (<Note>, <Tooltip ...>, </CardGroup>, <Card .../>)
# and a few lowercase inline wrappers we want gone but whose text we keep.
_COMPONENT_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9]*\b[^>]*?/?>")
_INLINE_WRAPPER_RE = re.compile(r"</?(?:sup|sub|u|kbd|abbr)\b[^>]*?/?>", re.IGNORECASE)

_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _mask(text: str, pattern: re.Pattern[str], store: list[str], tag: str) -> str:
    def repl(m: re.Match[str]) -> str:
        store.append(m.group(0))
        return f"\x00{tag}{len(store) - 1}\x00"

    return pattern.sub(repl, text)


def _unmask(text: str, store: list[str], tag: str) -> str:
    for i, original in enumerate(store):
        text = text.replace(f"\x00{tag}{i}\x00", original)
    return text


def clean(md: str) -> str:
    """Return reader-facing Markdown with MDX components removed."""
    text = _FRONTMATTER_RE.sub("", md, count=1)

    # Mask code first so tag-stripping can't touch it.
    fenced: list[str] = []
    inline: list[str] = []
    text = _mask(text, _FENCED_CODE_RE, fenced, "FENCE")
    text = _mask(text, _INLINE_CODE_RE, inline, "CODE")

    # Comments.
    text = _HTML_COMMENT_RE.sub("", text)
    text = _MDX_EXPR_COMMENT_RE.sub("", text)

    # <section title="X"> -> "## X" boundary; other sections just unwrap.
    text = _SECTION_OPEN_RE.sub(lambda m: f"\n## {m.group(1).strip()}\n", text)
    text = _SECTION_OPEN_NOTITLE_RE.sub("", text)
    text = _SECTION_CLOSE_RE.sub("", text)

    # Line breaks -> newline, then drop remaining component / inline wrapper tags.
    text = _BR_RE.sub("\n", text)
    text = _COMPONENT_TAG_RE.sub("", text)
    text = _INLINE_WRAPPER_RE.sub("", text)

    # Restore code.
    text = _unmask(text, inline, "CODE")
    text = _unmask(text, fenced, "FENCE")

    # Tidy whitespace.
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip() + "\n"
