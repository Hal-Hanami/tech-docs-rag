"""CLI for the M1 ingestion pipeline.

    python -m ingest manifest          # Stage 1 -> corpus_urls.txt + category report
    python -m ingest categories        # show ALL en docs categories (scope sanity check)
    python -m ingest fetch [--limit N] [--force]   # Stage 2 -> data/raw/*.md (cached)
    python -m ingest build                         # Stage 3 -> data/chunks.jsonl (+ stats)
    python -m ingest all  [--limit N] [--force]    # Stages 1-3 end to end
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import fetch, manifest
from .chunk import build_jsonl

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
URLS_FILE = ROOT / "corpus_urls.txt"
CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"


def _load_urls() -> list[str]:
    if not URLS_FILE.exists():
        raise SystemExit("corpus_urls.txt not found — run `python -m ingest manifest` first.")
    return [ln.strip() for ln in URLS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def cmd_manifest(_: argparse.Namespace) -> None:
    text = manifest.fetch_llms_txt()
    urls = manifest.in_scope_urls(text)
    URLS_FILE.write_text("\n".join(urls) + "\n", encoding="utf-8")
    print(f"in-scope pages: {len(urls)}  ->  {URLS_FILE.relative_to(ROOT)}")
    print("by category:")
    for cat, count in manifest.category_breakdown(urls).most_common():
        print(f"  {count:4d}  {cat}")


def cmd_categories(_: argparse.Namespace) -> None:
    text = manifest.fetch_llms_txt()
    counts = manifest.all_en_category_breakdown(text)
    print(f"all en docs categories ({sum(counts.values())} pages total):")
    for cat, count in counts.most_common():
        print(f"  {count:4d}  {cat}")


def cmd_fetch(args: argparse.Namespace) -> None:
    urls = _load_urls()
    if args.limit:
        urls = urls[: args.limit]
    print(f"fetching {len(urls)} pages into {RAW_DIR.relative_to(ROOT)} (cached) ...")
    paths = fetch.download_all(urls, RAW_DIR, force=args.force)
    print(f"done: {len(paths)} pages on disk")


def cmd_build(_: argparse.Namespace) -> None:
    urls = _load_urls()
    stats = build_jsonl(urls, RAW_DIR, CHUNKS_FILE)
    _print_stats(stats)


def cmd_all(args: argparse.Namespace) -> None:
    cmd_manifest(args)
    cmd_fetch(args)
    cmd_build(args)


def _print_stats(stats: dict) -> None:
    print(f"\nwrote {stats['chunks']} chunks from {stats['pages']} pages "
          f"-> {Path(stats['out_path']).name}")
    print(f"  tokens/chunk  min={stats['tokens_min']}  "
          f"median={stats['tokens_median']}  max={stats['tokens_max']}")
    print(f"  total tokens  ~{stats['tokens_total']:,}")
    lo, hi = 200, 3000
    ok = lo <= stats["chunks"] <= hi
    print(f"  DoD check (hundreds–few thousand): "
          f"{'OK' if ok else 'OUT OF RANGE'} ({stats['chunks']})")


def build_parser() -> argparse.ArgumentParser:
    """The flag surface, separated from `main` so tests can drive it with an argv.

    `all` re-declares `--limit` and `--force` because it runs `fetch` in the
    middle of the chain and hands it the same namespace: a flag missing here
    would not fail loudly, it would silently fetch the whole corpus.
    """
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").set_defaults(func=cmd_manifest)
    sub.add_parser("categories").set_defaults(func=cmd_categories)
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--limit", type=int, default=0)
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch)
    sub.add_parser("build").set_defaults(func=cmd_build)
    p_all = sub.add_parser("all")
    p_all.add_argument("--limit", type=int, default=0)
    p_all.add_argument("--force", action="store_true")
    p_all.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
