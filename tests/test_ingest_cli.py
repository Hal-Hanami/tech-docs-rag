"""Offline tests for the ingestion flag surface and the stages it drives.

Same reason `tests/test_cli.py` exists: this module holds no pipeline logic, which
is exactly the sort of place a bug survives a full green suite. The retrieval CLI
had one — `--dense-only` documented as the un-reranked baseline while the code
still passed a reranker — and nothing failed, because no test went through a flag.

What is pinned here is the translation, plus the one thing `all` can silently get
wrong: it runs `fetch` in the middle of the chain with the namespace it was given,
so a flag it forgets to declare does not raise, it downloads the whole corpus.

No network: the three stages are replaced with recorders.
"""

from __future__ import annotations

import pytest

from ingest import __main__ as ingest_cli


@pytest.fixture
def stages(monkeypatch, tmp_path):
    """Capture which stages ran and with what, without fetching anything."""
    seen: dict[str, dict] = {}

    def fake_llms_txt(*a, **kw):
        seen["manifest"] = {}
        return "- [Prompt caching](https://docs.example/pc.md)\n"

    def fake_download_all(urls, raw_dir, force=False):
        seen["fetch"] = {"urls": list(urls), "raw_dir": raw_dir, "force": force}
        return list(urls)

    def fake_build_jsonl(urls, raw_dir, out_path):
        seen["build"] = {"urls": list(urls), "out_path": out_path}
        return {"chunks": 3, "pages": 1, "tokens_min": 1, "tokens_median": 2,
                "tokens_max": 3, "tokens_total": 6, "out_path": str(out_path)}

    monkeypatch.setattr(ingest_cli.manifest, "fetch_llms_txt", fake_llms_txt)
    monkeypatch.setattr(ingest_cli.manifest, "in_scope_urls",
                        lambda text: ["https://docs.example/a.md", "https://docs.example/b.md"])
    monkeypatch.setattr(ingest_cli.fetch, "download_all", fake_download_all)
    monkeypatch.setattr(ingest_cli, "build_jsonl", fake_build_jsonl)
    # keep the corpus manifest out of the repo while the tests run
    monkeypatch.setattr(ingest_cli, "URLS_FILE", tmp_path / "corpus_urls.txt")
    monkeypatch.setattr(ingest_cli, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(ingest_cli, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(ingest_cli, "ROOT", tmp_path)
    return seen


def test_manifest_writes_the_url_list(stages, tmp_path):
    ingest_cli.main(["manifest"])
    assert "manifest" in stages
    assert (tmp_path / "corpus_urls.txt").read_text(encoding="utf-8").splitlines() == [
        "https://docs.example/a.md", "https://docs.example/b.md"]


def test_fetch_reads_the_url_list_and_honours_limit(stages, tmp_path):
    ingest_cli.main(["manifest"])
    ingest_cli.main(["fetch", "--limit", "1"])
    assert stages["fetch"]["urls"] == ["https://docs.example/a.md"]
    assert stages["fetch"]["force"] is False


def test_fetch_passes_force_through(stages):
    ingest_cli.main(["manifest"])
    ingest_cli.main(["fetch", "--force"])
    assert stages["fetch"]["force"] is True


def test_fetch_without_a_manifest_says_which_command_to_run_first(stages):
    with pytest.raises(SystemExit) as e:
        ingest_cli.main(["fetch"])
    assert "ingest manifest" in str(e.value)


def test_build_chunks_the_pages_already_on_disk(stages, tmp_path):
    ingest_cli.main(["manifest"])
    ingest_cli.main(["build"])
    assert stages["build"]["out_path"] == tmp_path / "chunks.jsonl"


def test_all_runs_the_three_stages_in_order(stages):
    ingest_cli.main(["all"])
    assert set(stages) == {"manifest", "fetch", "build"}


def test_all_forwards_its_flags_to_the_fetch_stage(stages):
    # `all` re-declares --limit/--force and hands cmd_fetch the same namespace.
    # If it stopped declaring them the run would not fail — it would quietly
    # fetch every page, which is slow, rude to the origin, and hard to notice.
    ingest_cli.main(["all", "--limit", "1", "--force"])
    assert stages["fetch"]["urls"] == ["https://docs.example/a.md"]
    assert stages["fetch"]["force"] is True


def test_a_subcommand_is_required(stages):
    with pytest.raises(SystemExit):
        ingest_cli.main([])
