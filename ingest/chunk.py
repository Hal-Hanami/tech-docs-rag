"""Stage 3 — structure-aware chunking + metadata, emitted as JSONL.

Split at H2/H3 boundaries (not fixed-length), keep each chunk to ~300–800
tokens, pack too-small sections into their neighbour and split oversize ones at
paragraph/sentence boundaries (never mid code-block or mid-table). Every chunk
carries metadata that traces back to the source: source_url / section_path /
anchor, so an answer in M2+ can cite "which doc, which section".

token_estimate is a chars/4 heuristic — accurate enough to control chunk size.
M2 can swap in the official token-counting API if exact counts are needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import fetch, scope
from .clean import clean

CHARS_PER_TOKEN = 4
MIN_TOKENS = 120
MAX_TOKENS = 800
_TARGET_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def est_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def slugify(text: str) -> str:
    t = re.sub(r"`+|\*+|_+", "", text.strip().lower())
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    t = re.sub(r"\s+", "-", t)
    return re.sub(r"-+", "-", t).strip("-")


# --- block parsing ----------------------------------------------------------

@dataclass
class Block:
    kind: str  # "heading" | "code" | "table" | "text"
    text: str
    level: int = 0
    heading_text: str = ""


def parse_blocks(md: str) -> list[Block]:
    lines = md.split("\n")
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        stripped = line.lstrip()
        if stripped.startswith("```"):  # fenced code — keep intact
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                closed = lines[i].lstrip().startswith("```")
                i += 1
                if closed:
                    break
            blocks.append(Block("code", "\n".join(buf)))
            continue
        m = _HEADING_RE.match(line)
        if m:
            blocks.append(Block("heading", line.strip(), level=len(m.group(1)),
                                heading_text=m.group(2).strip()))
            i += 1
            continue
        if stripped.startswith("|"):  # markdown table — keep intact
            buf = [line]
            i += 1
            while i < n and lines[i].lstrip().startswith("|"):
                buf.append(lines[i])
                i += 1
            blocks.append(Block("table", "\n".join(buf)))
            continue
        buf = [line]  # paragraph / list run
        i += 1
        while i < n and lines[i].strip():
            nxt = lines[i].lstrip()
            if nxt.startswith(("```", "|")) or _HEADING_RE.match(lines[i]):
                break
            buf.append(lines[i])
            i += 1
        blocks.append(Block("text", "\n".join(buf)))
    return blocks


# --- sectioning at H1/H2/H3 -------------------------------------------------

@dataclass
class Section:
    path: list[str]      # breadcrumb of headings (levels <= 3), e.g. [H1, H2, H3]
    anchor: str          # slug of this section's own heading ("" = page top)
    blocks: list[Block] = field(default_factory=list)


def split_into_sections(blocks: list[Block], page_title: str) -> list[Section]:
    sections: list[Section] = []
    path_stack: list[tuple[int, str]] = []
    cur: Section | None = None
    for b in blocks:
        if b.kind == "heading" and b.level <= 3:
            path_stack = [(lvl, t) for (lvl, t) in path_stack if lvl < b.level]
            path_stack.append((b.level, b.heading_text))
            anchor = "" if b.level == 1 else slugify(b.heading_text)
            cur = Section(path=[t for _, t in path_stack], anchor=anchor, blocks=[b])
            sections.append(cur)
        else:
            if cur is None:
                cur = Section(path=[page_title], anchor="", blocks=[])
                sections.append(cur)
            cur.blocks.append(b)
    return sections


# --- size control -----------------------------------------------------------

def _split_long_text(text: str, target: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        if len(para) > target:
            for sent in _SENTENCE_SPLIT_RE.split(para):
                if buf and len(buf) + len(sent) + 1 > target:
                    out.append(buf.strip())
                    buf = ""
                buf = f"{buf} {sent}".strip()
        else:
            if buf and len(buf) + len(para) + 2 > target:
                out.append(buf.strip())
                buf = ""
            buf = f"{buf}\n\n{para}".strip()
    if buf.strip():
        out.append(buf.strip())
    return out


def _section_pieces(sec: Section) -> list[dict]:
    """Render a section into one or more text pieces, each <= ~MAX tokens."""
    pieces: list[str] = []
    buf: list[str] = []
    chars = 0

    def emit() -> None:
        nonlocal buf, chars
        if buf:
            pieces.append("\n\n".join(buf).strip())
            buf, chars = [], 0

    for b in sec.blocks:
        if b.kind in ("code", "table"):  # never split these
            if chars and chars + len(b.text) > _TARGET_CHARS:
                emit()
            buf.append(b.text)
            chars += len(b.text) + 2
        elif len(b.text) > _TARGET_CHARS:
            emit()
            pieces.extend(_split_long_text(b.text, _TARGET_CHARS))
        else:
            if chars and chars + len(b.text) > _TARGET_CHARS:
                emit()
            buf.append(b.text)
            chars += len(b.text) + 2
    emit()
    return [{"path": sec.path, "anchor": sec.anchor, "text": t} for t in pieces if t]


def _merge_small(units: list[dict]) -> list[dict]:
    """Fold under-sized units into the previous chunk (design: small -> parent)."""
    merged: list[dict] = []
    for u in units:
        if merged:
            prev = merged[-1]
            combined = est_tokens(prev["text"]) + est_tokens(u["text"])
            prev_small = est_tokens(prev["text"]) < MIN_TOKENS
            cur_small = est_tokens(u["text"]) < MIN_TOKENS
            if (prev_small or cur_small) and combined <= MAX_TOKENS:
                prev["text"] += "\n\n" + u["text"]  # keep prev's path/anchor
                continue
        merged.append(dict(u))
    return merged


# --- public API -------------------------------------------------------------

def chunk_page(cleaned_md: str, source_md_url: str) -> list[dict]:
    source_url = source_md_url[:-3] if source_md_url.endswith(".md") else source_md_url
    page_slug = scope.page_path(source_md_url) or source_url
    blocks = parse_blocks(cleaned_md)
    page_title = next((b.heading_text for b in blocks
                       if b.kind == "heading" and b.level == 1), "")
    if not page_title:
        page_title = page_slug.rsplit("/", 1)[-1].replace("-", " ").title()

    units: list[dict] = []
    for sec in split_into_sections(blocks, page_title):
        units.extend(_section_pieces(sec))
    units = _merge_small(units)

    chunks: list[dict] = []
    for idx, u in enumerate(units):
        anchor = u["anchor"]
        url = f"{source_url}#{anchor}" if anchor else source_url
        chunks.append({
            "id": f"{page_slug}#{idx}",
            "source_url": source_url,
            "source_md_url": source_md_url,
            "url": url,
            "page_title": page_title,
            "section_path": " > ".join(u["path"]),
            "anchor": anchor,
            "chunk_index": idx,
            "token_estimate": est_tokens(u["text"]),
            "char_count": len(u["text"]),
            "text": u["text"],
        })
    return chunks


def build_jsonl(urls: list[str], raw_dir: Path, out_path: Path) -> dict:
    """Clean + chunk every cached page, write JSONL, return summary stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_tokens: list[int] = []
    pages = 0
    with out_path.open("w", encoding="utf-8") as f:
        for url in urls:
            raw_path = fetch.cache_path(raw_dir, url)
            if not raw_path.exists():
                continue
            cleaned = clean(raw_path.read_text(encoding="utf-8"))
            for chunk in chunk_page(cleaned, url):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                all_tokens.append(chunk["token_estimate"])
            pages += 1

    all_tokens.sort()
    n = len(all_tokens)
    median = all_tokens[n // 2] if n else 0
    return {
        "pages": pages,
        "chunks": n,
        "tokens_min": all_tokens[0] if n else 0,
        "tokens_median": median,
        "tokens_max": all_tokens[-1] if n else 0,
        "tokens_total": sum(all_tokens),
        "out_path": str(out_path),
    }
