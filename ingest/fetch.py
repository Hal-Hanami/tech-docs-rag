"""Stage 2 — download each in-scope page as `.md` into a local cache.

Polite + cache-first: a page already on disk is not re-downloaded, and there's a
small delay between network requests. We never *commit* these files (see
.gitignore) — the corpus body is referenced by source_url, not redistributed.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.request import Request, urlopen

from . import scope

_USER_AGENT = "tech-docs-rag/0.1 (personal RAG ingest; +https://platform.claude.com/llms.txt)"


def cache_path(raw_dir: Path, url: str) -> Path:
    """Local path for a page's `.md`, mirroring its docs path under raw_dir."""
    path = scope.page_path(url) or "unknown"
    return raw_dir / f"{path}.md"


def fetch_page(url: str, timeout: float = 30.0) -> str:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def download_all(
    urls: list[str],
    raw_dir: Path,
    *,
    delay: float = 0.25,
    force: bool = False,
) -> list[Path]:
    """Download every URL into raw_dir (cached). Returns the local paths."""
    paths: list[Path] = []
    for i, url in enumerate(urls, 1):
        dest = cache_path(raw_dir, url)
        if dest.exists() and not force:
            paths.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = fetch_page(url)
        dest.write_text(text, encoding="utf-8")
        paths.append(dest)
        print(f"  [{i}/{len(urls)}] fetched {url}")
        time.sleep(delay)
    return paths
