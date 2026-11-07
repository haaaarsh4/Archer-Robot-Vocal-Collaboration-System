"""Web search for the assistant, with the source kept beside every fact.

The default provider is DuckDuckGo's HTML endpoint: no account, no key, and the
same privacy posture as the rest of the app, which runs on the visitor's
machine. Tavily and Brave are used automatically when their keys are present,
because a free HTML endpoint can rate-limit an enthusiastic user. If the HTML
endpoint serves a bot challenge, the instant-answer JSON API and Wikipedia are
tried in turn. They are narrower than a full web index, but they are real
sources, and a narrower answer that names its source is better than a silent
zero or an invented one.

Only result titles, URLs and snippets are used. The result pages themselves are
never fetched: that would turn a search box into an uncontrolled request for
arbitrary URLs, and a snippet chosen by the search engine is enough to ground a
short answer. Results are cached in memory so asking the same question twice
does not hit the network twice.

Everything here fails soft. A search that times out or returns a block page is
an empty list, and the caller tells the model the search found nothing rather
than pretending it did.
"""
from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urlparse

import requests
from loguru import logger

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ArcherAssistant/1.0"
)

# Requests for a search, as opposed to a question that happens to mention one.
_EXPLICIT = re.compile(
    r"^\s*(?:please\s+)?(?:search(?:\s+the\s+web)?|web\s*search|look\s+up|"
    r"google|browse)\s+(?:for\s+)?(.+?)\s*[?.!]*$",
    re.IGNORECASE | re.DOTALL,
)

# A question whose answer changes, so a stale model memory is not good enough.
_FRESH = re.compile(
    r"\b(?:today|tonight|yesterday|latest|newest|current|currently|recent|"
    r"recently|news|headline|price|cost|weather|forecast|stock|score|"
    r"who\s+won|release\s+date|released|this\s+year|this\s+month|"
    r"right\s+now|live|2025|2026|2027)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


_TAG = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Search APIs return highlighted snippets as HTML. Keep the words."""
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def _dedupe(rows: list[SearchResult], limit: int) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for row in rows:
        if not row.url or row.url in seen:
            continue
        seen.add(row.url)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _clean_url(href: str) -> str:
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            href = target[0]
    return href if href.startswith(("http://", "https://")) else ""


class _DuckParser(HTMLParser):
    """Read DuckDuckGo's HTML results without a scraping dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._mode: str | None = None
        self._buf: list[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if tag == "a" and any(c.startswith("result__a") for c in classes):
            self._flush_title()
            self._mode, self._buf = "title", []
            self._href = values.get("href") or ""
        elif any(c.startswith("result__snippet") for c in classes):
            self._mode, self._buf = "snippet", []
        elif tag == "a" and any(c == "result-link" for c in classes):
            self._flush_title()
            self._mode, self._buf = "title", []
            self._href = values.get("href") or ""
        elif tag == "td" and any(c == "result-snippet" for c in classes):
            self._mode, self._buf = "snippet", []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._mode == "title" and tag == "a":
            self._flush_title()
        elif self._mode == "snippet" and tag in {"a", "div", "td"}:
            self._flush_snippet()

    def _flush_title(self) -> None:
        if self._mode != "title":
            return
        title = " ".join("".join(self._buf).split())
        url = _clean_url(self._href)
        self._mode, self._buf = None, []
        if title and url:
            self.results.append(SearchResult(title, url, ""))

    def _flush_snippet(self) -> None:
        if self._mode != "snippet":
            return
        snippet = " ".join("".join(self._buf).split())
        self._mode, self._buf = None, []
        if snippet and self.results:
            last = self.results[-1]
            if not last.snippet:
                self.results[-1] = SearchResult(last.title, last.url, snippet)

    def close(self) -> None:
        super().close()
        self._flush_title()
        self._flush_snippet()


def _parse_duck(html: str) -> list[SearchResult]:
    parser = _DuckParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML is an empty result
        pass
    seen: set[str] = set()
    out: list[SearchResult] = []
    for result in parser.results:
        if result.url in seen:
            continue
        seen.add(result.url)
        out.append(result)
    return out


class WebSearch:
    """Search with a small cache, so repeats and retries do not hit the network."""

    def __init__(self, cache_seconds: float = 600.0, provider: str | None = None):
        self.cache_seconds = float(cache_seconds)
        self.provider = (provider or os.environ.get("ARCHER_SEARCH_PROVIDER")
                         or "auto").lower()
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}

    def search(self, query: str, max_results: int = 5,
               timeout: float = 8.0) -> list[SearchResult]:
        query = " ".join(str(query).split())[:300]
        if not query:
            return []
        key = query.lower()
        cached = self._cache.get(key)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1][:max_results]

        results = _dedupe(self._run(query, max_results, timeout), max_results)
        self._cache[key] = (now + self.cache_seconds, results)
        return results[:max_results]

    def _run(self, query: str, max_results: int,
             timeout: float) -> list[SearchResult]:
        provider = self.provider
        try:
            keyed = (
                ("exa", "EXA_API_KEY", self._exa),
                ("tavily", "TAVILY_API_KEY", self._tavily),
                ("brave", "BRAVE_SEARCH_API_KEY", self._brave),
            )
            for name, env_name, finder in keyed:
                if provider not in {"auto", name} or not os.environ.get(env_name):
                    continue
                try:
                    found = finder(query, max_results, timeout)
                except Exception as exc:  # noqa: BLE001 - a bad key must not kill search
                    logger.warning(f"[web] {name} failed: {exc}")
                    continue
                if found:
                    return found
            # Each keyless source gets its own try: a blocked HTML endpoint
            # must not stop the JSON API or Wikipedia from answering. Results
            # are combined, because an abstract and a few encyclopedia entries
            # are a better answer together than whichever came first.
            keyless: list[SearchResult] = []
            for finder in (self._duckduckgo, self._duckduckgo_instant,
                           self._wikipedia):
                try:
                    found = finder(query, max_results, timeout)
                except Exception as exc:  # noqa: BLE001 - try the next source
                    logger.warning(f"[web] {finder.__name__} failed: {exc}")
                    continue
                if found:
                    keyless.extend(found)
                    if len(_dedupe(keyless, max_results)) >= max_results:
                        break
            if keyless:
                return _dedupe(keyless, max_results)
        except requests.exceptions.RequestException as exc:
            logger.warning(f"[web] search failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - a search must never break chat
            logger.warning(f"[web] search failed: {exc}")
        return []

    def _duckduckgo(self, query: str, max_results: int,
                    timeout: float) -> list[SearchResult]:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.8"},
            timeout=timeout,
        )
        response.raise_for_status()
        results = _parse_duck(response.text)
        if not results:
            # The lite endpoint is a different markup and often survives when
            # the main one serves a challenge page.
            response = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers={"User-Agent": _UA},
                timeout=timeout,
            )
            response.raise_for_status()
            results = _parse_duck(response.text)
        return results[:max_results]

    def _duckduckgo_instant(self, query: str, max_results: int,
                            timeout: float) -> list[SearchResult]:
        """DuckDuckGo's JSON answer API, which still answers when HTML is blocked.

        It is an encyclopedia and a disambiguation service rather than a web
        index: an abstract, a few instant results, and related topics. That is
        enough to ground a definition or a name, and every item carries the URL
        it came from.
        """
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1,
                    "skip_disambig": 1},
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.8"},
            timeout=timeout,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            return []
        if not isinstance(data, dict):
            return []

        rows: list[SearchResult] = []
        abstract = _strip_html(str(data.get("AbstractText") or ""))
        abstract_url = _clean_url(str(data.get("AbstractURL") or ""))
        if abstract and abstract_url:
            heading = _strip_html(str(data.get("Heading") or "")) or "answer"
            rows.append(SearchResult(heading, abstract_url, abstract))

        for item in data.get("Results") or []:
            if not isinstance(item, dict):
                continue
            text = _strip_html(str(item.get("Text") or ""))
            url = _clean_url(str(item.get("FirstURL") or ""))
            if url:
                rows.append(SearchResult(text[:140] or url, url, text))

        def walk(items: list) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("Topics"):
                    walk(list(item["Topics"]))
                    continue
                text = _strip_html(str(item.get("Text") or ""))
                url = _clean_url(str(item.get("FirstURL") or ""))
                if not url or "duckduckgo.com" in urlparse(url).netloc:
                    continue
                rows.append(SearchResult(text[:140] or url, url, text))

        walk(list(data.get("RelatedTopics") or []))
        return _dedupe(rows, max_results)

    def _wikipedia(self, query: str, max_results: int,
                   timeout: float) -> list[SearchResult]:
        """Wikipedia's search API: a keyless, attributable last resort."""
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": max_results, "format": "json", "srprop": "snippet",
                "utf8": 1,
            },
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.8"},
            timeout=timeout,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            return []
        if not isinstance(data, dict):
            return []
        rows: list[SearchResult] = []
        for item in (data.get("query") or {}).get("search") or []:
            if not isinstance(item, dict):
                continue
            title = _strip_html(str(item.get("title") or ""))
            if not title:
                continue
            url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            snippet = _strip_html(str(item.get("snippet") or ""))
            rows.append(SearchResult(title, url, snippet))
        return _dedupe(rows, max_results)

    def _exa(self, query: str, max_results: int,
             timeout: float) -> list[SearchResult]:
        """Exa's REST search, used when EXA_API_KEY is set.

        Semantic ranking is a better fit than a keyword SERP for a question
        written as a sentence, and its free tier is enough for a personal
        assistant. No SDK is needed: this is one POST.
        """
        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": os.environ["EXA_API_KEY"],
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": max_results,
                "type": "auto",
                "contents": {"text": {"maxCharacters": 1000}},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return [
            SearchResult(
                str(row.get("title") or row.get("url") or "result"),
                str(row.get("url") or ""),
                str(row.get("text") or row.get("summary") or ""),
            )
            for row in response.json().get("results", [])
            if row.get("url")
        ]

    def _tavily(self, query: str, max_results: int,
                timeout: float) -> list[SearchResult]:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return [
            SearchResult(
                str(row.get("title") or row.get("url") or "result"),
                str(row.get("url") or ""),
                str(row.get("content") or ""),
            )
            for row in response.json().get("results", [])
            if row.get("url")
        ]

    def _brave(self, query: str, max_results: int,
               timeout: float) -> list[SearchResult]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        rows = response.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                str(row.get("title") or "result"),
                str(row.get("url") or ""),
                str(row.get("description") or ""),
            )
            for row in rows
            if row.get("url")
        ]


def query_for(message: str, has_notes: bool = False, mode: str = "auto") -> str | None:
    """The text to search for, or None when this message does not need the web.

    ``mode`` is ``off``, ``auto`` or ``always``. In auto, an explicit request
    always searches, a question about something that changes searches, and a
    question already answered by the song notes does not.
    """
    mode = (mode or "auto").lower()
    if mode == "off":
        return None
    text = " ".join(str(message).split()).strip()
    if not text:
        return None
    explicit = _EXPLICIT.match(text)
    if explicit:
        return explicit.group(1).strip() or None
    if mode == "always":
        return text
    if has_notes:
        return None
    if _FRESH.search(text):
        return text
    return None


def format_results(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, 1):
        lines.append(f"{index}. {result.title} ({result.url})")
        if result.snippet:
            lines.append(f"   {result.snippet}")
    return "\n".join(lines)


def sources(results: list[SearchResult]) -> list[str]:
    return [result.url for result in results if result.url]
