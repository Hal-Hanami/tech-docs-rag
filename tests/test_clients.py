"""Offline tests for the network adapters — design §1.1, §6.4.

Nothing here reaches the network. What is pinned is the part of the client that
is reached only when something goes wrong, and which no measurement exercises:
the message a failure produces.

Both endpoints share one HTTP helper (§1.1 keeps that helper the only place a
socket is opened), and sharing it once cost the error message its most useful
word. §6.4 says a failure names the boundary that failed, so that is asserted
here rather than left to be noticed during an outage.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from rag.clients import voyage


def _raises(exc: Exception):
    def fake_urlopen(request, timeout=None):
        raise exc
    return fake_urlopen


def _http_error(code: int, body: str = "bad request") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.voyageai.com/v1/x", code, "err", {}, io.BytesIO(body.encode()))


# --- §6.4 a failure names the boundary that failed --------------------------------

@pytest.mark.parametrize("url,label", [
    (voyage.EMBED_URL, "Voyage embedding API"),
    (voyage.RERANK_URL, "Voyage rerank API"),
])
def test_a_non_retryable_error_names_its_endpoint(monkeypatch, url, label):
    monkeypatch.setattr(voyage.urllib.request, "urlopen", _raises(_http_error(400)))
    with pytest.raises(SystemExit) as e:
        voyage._post(url, {}, "key", label=label)
    assert str(e.value).startswith(f"{label} error 400")


def test_the_servers_own_explanation_is_passed_through(monkeypatch):
    monkeypatch.setattr(voyage.urllib.request, "urlopen",
                        _raises(_http_error(400, "model 'nope' does not exist")))
    with pytest.raises(SystemExit) as e:
        voyage._post(voyage.EMBED_URL, {}, "key", label="Voyage embedding API")
    # the API's own words beat any paraphrase of ours
    assert "model 'nope' does not exist" in str(e.value)


def test_a_connection_failure_names_its_endpoint(monkeypatch):
    monkeypatch.setattr(voyage.urllib.request, "urlopen",
                        _raises(urllib.error.URLError("no route to host")))
    with pytest.raises(SystemExit) as e:
        voyage._post(voyage.RERANK_URL, {}, "key", label="Voyage rerank API", retries=1)
    assert str(e.value).startswith("Voyage rerank API connection error")


def test_a_retryable_status_still_fails_by_name_once_retries_run_out(monkeypatch):
    # retries=1 so the last attempt is the first one — no backoff sleep in the suite
    monkeypatch.setattr(voyage.urllib.request, "urlopen", _raises(_http_error(429)))
    with pytest.raises(SystemExit) as e:
        voyage._post(voyage.EMBED_URL, {}, "key", label="Voyage embedding API", retries=1)
    assert "Voyage embedding API error 429" in str(e.value)


# --- the seams the callers depend on ----------------------------------------------

def test_an_explicit_key_is_preferred_over_the_environment(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "from-env")
    assert voyage._require_key("explicit") == "explicit"
    assert voyage._require_key(None) == "from-env"


def test_a_missing_key_explains_how_to_supply_one(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        voyage._require_key(None, disable_hint="\n  Or disable reranking.")
    assert "VOYAGE_API_KEY not set" in str(e.value)
    assert "Or disable reranking." in str(e.value)  # the hint reaches the reader


def test_reranking_nothing_costs_nothing(monkeypatch):
    # An empty candidate pool must not become a billed request.
    monkeypatch.setattr(voyage.urllib.request, "urlopen",
                        _raises(AssertionError("should not be called")))
    reranker = voyage.VoyageReranker(api_key="k")
    assert reranker.rerank("q", []) == []
    assert reranker.usage["total_tokens"] == 0
