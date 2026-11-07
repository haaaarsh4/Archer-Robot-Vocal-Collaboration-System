"""Web search parsing, caching and query selection, with no network.

    PYTHONPATH=. .venv/bin/python tests/test_web_search.py
    PYTHONPATH=. .venv/bin/python -m pytest tests/test_web_search.py -q
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chat import web_search

_HTML = """
<html><body>
<div class="result results_links">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fcree">Plains Cree language</a>
  <a class="result__snippet">Plains Cree is an Algonquian language spoken in Canada.</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://example.com/words">Cree words</a>
  <a class="result__snippet">A short vocabulary.</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://example.com/words">Duplicate</a>
</div>
</body></html>
"""


class _Response:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {}


def test_results_are_parsed_and_deduplicated(monkeypatch):
    monkeypatch.setattr(web_search.requests, "get",
                        lambda *args, **kwargs: _Response(_HTML))
    search = web_search.WebSearch(provider="duckduckgo")
    rows = search.search("plains cree", max_results=5)
    assert len(rows) == 2
    assert rows[0].title == "Plains Cree language"
    assert rows[0].url == "https://example.org/cree"
    assert "Algonquian" in rows[0].snippet
    assert rows[1].url == "https://example.com/words"
    assert web_search.sources(rows) == [
        "https://example.org/cree", "https://example.com/words",
    ]


def test_the_same_query_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return _Response(_HTML)

    monkeypatch.setattr(web_search.requests, "get", fake_get)
    search = web_search.WebSearch(cache_seconds=60, provider="duckduckgo")
    search.search("plains cree")
    after_first = calls["n"]
    search.search("plains cree")
    assert calls["n"] == after_first
    assert after_first >= 1


def test_query_selection():
    assert web_search.query_for("search the web for plains cree") == "plains cree"
    assert web_search.query_for("look up the latest Cree dictionary") == "the latest Cree dictionary"
    assert web_search.query_for("what is the latest news about Cree?") == \
        "what is the latest news about Cree?"
    assert web_search.query_for("what does maskisin mean?") is None
    assert web_search.query_for("explain Horse", has_notes=True) is None
    assert web_search.query_for("search for Horse", mode="off") is None
    assert web_search.query_for("hello", mode="always") == "hello"


def test_a_network_error_is_an_empty_result(monkeypatch):
    def boom(*args, **kwargs):
        raise web_search.requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(web_search.requests, "get", boom)
    search = web_search.WebSearch(provider="duckduckgo")
    assert search.search("plains cree") == []


class _JsonResponse(_Response):
    def __init__(self, data: dict):
        super().__init__("")
        self._data = data

    def json(self) -> dict:
        return self._data


_INSTANT = {
    "Heading": "Plains Cree language",
    "AbstractText": "Plains Cree is a dialect of the Algonquian language Cree.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Plains_Cree_language",
    "RelatedTopics": [],
}

_WIKI = {"query": {"search": [{
    "title": "Plains Cree language",
    "snippet": '<span class="searchmatch">Plains Cree</span> is an Algonquian language.',
}]}}


def test_instant_answer_is_parsed(monkeypatch):
    monkeypatch.setattr(web_search.requests, "get",
                        lambda *args, **kwargs: _JsonResponse(_INSTANT))
    rows = web_search.WebSearch(provider="duckduckgo")._duckduckgo_instant(
        "plains cree", 5, 8)
    assert rows[0].url == "https://en.wikipedia.org/wiki/Plains_Cree_language"
    assert "Algonquian" in rows[0].snippet


def test_wikipedia_search_is_parsed(monkeypatch):
    monkeypatch.setattr(web_search.requests, "get",
                        lambda *args, **kwargs: _JsonResponse(_WIKI))
    rows = web_search.WebSearch(provider="duckduckgo")._wikipedia(
        "plains cree", 5, 8)
    assert rows[0].title == "Plains Cree language"
    assert rows[0].url == "https://en.wikipedia.org/wiki/Plains_Cree_language"
    assert "Algonquian" in rows[0].snippet
    assert "<span" not in rows[0].snippet


def test_keyless_chain_falls_through_to_wikipedia(monkeypatch):
    for key in ("EXA_API_KEY", "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
                "ARCHER_SEARCH_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(web_search.WebSearch, "_duckduckgo", lambda *args: [])
    monkeypatch.setattr(web_search.WebSearch, "_duckduckgo_instant", lambda *args: [])
    monkeypatch.setattr(
        web_search.WebSearch, "_wikipedia",
        lambda *args: [web_search.SearchResult(
            "Plains Cree language",
            "https://en.wikipedia.org/wiki/Plains_Cree_language",
            "Algonquian language",
        )],
    )
    rows = web_search.WebSearch(provider="auto").search("plains cree")
    assert rows[0].url == "https://en.wikipedia.org/wiki/Plains_Cree_language"


def test_a_failing_keyed_provider_falls_back(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(
        web_search.WebSearch, "_exa",
        lambda *args: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    monkeypatch.setattr(
        web_search.WebSearch, "_duckduckgo",
        lambda *args: [web_search.SearchResult("P", "https://example.org/p", "s")],
    )
    monkeypatch.setattr(web_search.WebSearch, "_duckduckgo_instant", lambda *args: [])
    monkeypatch.setattr(web_search.WebSearch, "_wikipedia", lambda *args: [])
    rows = web_search.WebSearch(provider="auto").search("anything")
    assert rows[0].url == "https://example.org/p"


def _main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    ran = 0
    for test in tests:
        if test.__code__.co_argcount:
            continue
        test()
        ran += 1
        print(f"ok  {test.__name__}")
    print(f"{ran} passed; the monkeypatch tests run under pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
