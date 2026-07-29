"""Where files live, and how credentials get into the process.

Kept in one module so that "what path does this write to" is answerable without
reading the CLI, and so tests can point at temporary directories without
monkey-patching module internals.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root, derived from this file's location rather than the working
# directory — the CLI must behave the same whether it is run from the repo root
# or from anywhere else.
ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"
DB_FILE = ROOT / "data" / "index.db"
EVAL_FILE = ROOT / "eval" / "qa.jsonl"


def load_dotenv(root: Path = ROOT) -> None:
    """Populate `os.environ` from a gitignored `.env` of `KEY=VALUE` lines.

    Existing environment variables win (`setdefault`), so an explicitly exported
    key always beats the file — otherwise a stale `.env` would silently override
    the key you just exported to debug something.

    This is intentionally ~10 lines rather than a dependency: the file format we
    need is one `=` per line, and adding a package for that would enlarge the
    install footprint of a project whose whole point is a small dependency set.
    """
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
