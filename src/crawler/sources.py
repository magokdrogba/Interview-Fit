"""Interview-context collection: Wikipedia summary + user-supplied URLs + manual notes.

What this module **does**:
  * Asks Wikipedia's public REST API (no auth, ToS-friendly) for a one-paragraph
    company summary in Korean first, then English.
  * Fetches additional public URLs the user provides (news articles, the
    company's own About/Careers pages, etc.) using
    :func:`src.crawler.base.polite_fetch`, then strips boilerplate to plain text.
  * Accepts a free-form ``manual_input`` string — exactly what the user pasted
    after reading reviews on Jobplanet / Catch / Blind themselves. The
    ``ai-mock-interview`` codebase never scrapes those sites.

What this module **does not do**: log in to anything, solve CAPTCHAs, retry past
a single failure, or pretend to be a browser. Every fetch may return nothing,
and the rest of the pipeline continues with whatever did come back.

Result is cached to disk under ``data/cache/`` keyed by a hash of
``(company, sorted(extra_urls), manual_input)`` so re-runs are free; pass
``force_refresh=True`` to bypass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from config import CACHE_DIR
from src.crawler.base import polite_fetch

logger = logging.getLogger(__name__)

WIKIPEDIA_LANGS: tuple[str, ...] = ("ko", "en")
_MAX_SNIPPET_CHARS = 4_000  # cap each extracted snippet to keep prompts lean

# --- Auto web-search (Issue 1) ---------------------------------------------
# We build search queries from the company name alone, so users never need to
# paste URLs. DuckDuckGo's HTML endpoint needs no API key.
_SEARCH_HEADERS = {"User-Agent": "Mozilla/5.0"}
_SEARCH_TIMEOUT_S = 5.0
_RESULTS_PER_QUERY = 3
_SEARCH_SNIPPET_CHARS = 1_500     # per-page truncation for search results
_MIN_PARAGRAPH_CHARS = 30         # keep only substantial <p>/<li> text


def _search_queries(company: str) -> list[str]:
    return [
        f"{company} 면접 후기",
        f"{company} 면접 질문",
        f"{company} 기업문화",
        f"{company} 인재상",
    ]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class InterviewContext:
    """Everything we know about the target company + role, post-collection."""

    company: str
    job_posting: str = ""
    company_summary: str = ""           # one-paragraph Wikipedia-style blurb
    public_snippets: list[dict] = field(default_factory=list)  # [{url, text}]
    manual_notes: str = ""              # user-pasted text (e.g. paraphrased reviews)
    sources: list[str] = field(default_factory=list)  # human-readable trail
    generated_at: str = ""

    def to_prompt_block(self) -> str:
        """Render as a single block suitable for inclusion in an LLM prompt."""
        parts: list[str] = [f"Company: {self.company}"]
        if self.company_summary:
            parts.append(f"\n# Company summary\n{self.company_summary}")
        if self.job_posting:
            parts.append(f"\n# Job posting\n{self.job_posting}")
        if self.public_snippets:
            parts.append("\n# Public snippets")
            for s in self.public_snippets:
                parts.append(f"\n[{s['url']}]\n{s['text']}")
        if self.manual_notes:
            parts.append(f"\n# User notes (manually collected)\n{self.manual_notes}")
        return "\n".join(parts).strip()

    def is_empty(self) -> bool:
        return not any(
            (self.company_summary, self.public_snippets, self.manual_notes, self.job_posting)
        )


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------
def fetch_wikipedia_summary(company: str, langs: Iterable[str] = WIKIPEDIA_LANGS) -> tuple[str, str] | None:
    """Return ``(summary_text, source_url)`` for the first language that has an
    article, or None if nothing is found / fetch is blocked."""
    for lang in langs:
        title = quote(company.replace(" ", "_"))
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
        body = polite_fetch(url, accept="application/json")
        if not body:
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "disambiguation":
            continue
        extract = (payload.get("extract") or "").strip()
        if extract:
            page_url = (
                (payload.get("content_urls") or {}).get("desktop", {}).get("page")
                or f"https://{lang}.wikipedia.org/wiki/{title}"
            )
            return extract, page_url
    return None


# ---------------------------------------------------------------------------
# Generic public URL
# ---------------------------------------------------------------------------
_BOILERPLATE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")
_WS_RUN = re.compile(r"\s+")


def fetch_public_url(url: str) -> str | None:
    """Fetch an arbitrary URL and return its main text (de-boilerplated).

    Returns None if the fetch was blocked or returned nothing useful.
    """
    html = polite_fetch(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in _BOILERPLATE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Prefer <main> or <article> if present; else fall back to body.
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    text = _WS_RUN.sub(lambda m: "\n" if "\n" in m.group(0) else " ", text)
    text = text.strip()
    if not text:
        return None
    return text[:_MAX_SNIPPET_CHARS]


# ---------------------------------------------------------------------------
# Auto web-search (DuckDuckGo HTML) — no API key required
# ---------------------------------------------------------------------------
def _ddg_real_url(href: str) -> str | None:
    """Unwrap a DuckDuckGo result href into the real target URL."""
    if not href:
        return None
    parsed = urlparse(href if href.startswith("http") else "https:" + href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        return unquote(target[0]) if target else None
    return href if href.startswith("http") else None


def search_duckduckgo(query: str, max_results: int = _RESULTS_PER_QUERY) -> list[str]:
    """Return up to ``max_results`` result URLs for ``query``. Never raises."""
    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    try:
        resp = requests.get(url, headers=_SEARCH_HEADERS, timeout=_SEARCH_TIMEOUT_S)
    except requests.RequestException as exc:
        logger.info("DuckDuckGo search failed for %r: %s", query, exc)
        return []
    if not resp.ok:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    urls: list[str] = []
    for a in soup.select("a.result__a") or soup.find_all("a"):
        real = _ddg_real_url(a.get("href", ""))
        if real and real not in urls:
            urls.append(real)
        if len(urls) >= max_results:
            break
    return urls


def _same_host(a: str, b: str) -> bool:
    def host(u: str) -> str:
        return urlparse(u).netloc.lower().removeprefix("www.")
    return host(a) == host(b)


def fetch_search_result_text(url: str) -> str | None:
    """Fetch one search-result page and extract substantial <p>/<li> text.

    Returns None if the fetch fails, redirects to a different host (likely a
    login wall), or the page reads as a login page. Output is truncated to
    ``_SEARCH_SNIPPET_CHARS``.
    """
    try:
        resp = requests.get(
            url, headers=_SEARCH_HEADERS, timeout=_SEARCH_TIMEOUT_S, allow_redirects=True
        )
    except requests.RequestException as exc:
        logger.info("fetch failed for %s: %s", url, exc)
        return None
    if not resp.ok:
        return None

    # Login-wall detection: cross-host redirect, or login keywords up front.
    head = resp.text[:500]
    if not _same_host(resp.url, url):
        logger.info("skip %s — redirected to %s (login wall?)", url, resp.url)
        return None
    if "로그인" in head or "login" in head.lower():
        logger.info("skip %s — login keyword in first 500 chars", url)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    parts: list[str] = []
    for tag in soup.find_all(["p", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) >= _MIN_PARAGRAPH_CHARS:
            parts.append(text)
    combined = "\n".join(parts).strip()
    if not combined:
        return None
    return combined[:_SEARCH_SNIPPET_CHARS]


def _collect_web_snippets(company: str) -> list[dict]:
    """Search the company name and return de-duplicated {url, text} snippets."""
    seen_urls: set[str] = set()
    candidate_urls: list[str] = []
    for query in _search_queries(company):
        for url in search_duckduckgo(query):
            if url not in seen_urls:
                seen_urls.add(url)
                candidate_urls.append(url)

    snippets: list[dict] = []
    seen_text: set[str] = set()
    for url in candidate_urls:
        text = fetch_search_result_text(url)
        if not text:
            continue
        key = text[:200]
        if key in seen_text:
            continue
        seen_text.add(key)
        snippets.append({"url": url, "text": text})
    return snippets


# ---------------------------------------------------------------------------
# Orchestrator + cache
# ---------------------------------------------------------------------------
def _slug(company: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", company.strip().lower()).strip("-")
    return s or "company"


def _cache_path(company: str) -> "Path":  # noqa: F821 — Path imported lazily below
    from pathlib import Path  # local import keeps top of file uncluttered

    return Path(CACHE_DIR) / f"context_{_slug(company)}.json"


def _cache_key(company: str, extra_urls: list[str], manual_input: str, job_posting: str) -> str:
    payload = json.dumps(
        {"c": company, "u": sorted(extra_urls), "m": manual_input, "j": job_posting},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_interview_context(
    company: str,
    *,
    job_posting: str = "",
    manual_input: str = "",
    extra_urls: list[str] | None = None,
    auto_search: bool = True,
    force_refresh: bool = False,
) -> InterviewContext:
    """Collect public info + manual notes into a single InterviewContext.

    When ``auto_search`` is True (the default), interview-relevant pages are
    found automatically from the company name alone via DuckDuckGo — no URL
    input required. Always succeeds; if everything was blocked and the user
    provided no manual input, the result holds only the company name and any
    job_posting the caller passed in.
    """
    extra_urls = list(extra_urls or [])
    company = company.strip()
    if not company:
        raise ValueError("company must be non-empty")

    key = _cache_key(company, extra_urls, manual_input, job_posting)
    path = _cache_path(company)

    if not force_refresh and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("_key") == key:
                cached.pop("_key", None)
                return InterviewContext(**cached)
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Cache at %s is unreadable; refetching", path)

    ctx = InterviewContext(
        company=company,
        job_posting=job_posting.strip(),
        manual_notes=manual_input.strip(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Wikipedia (always attempted — Issue 1 step 3)
    wiki = fetch_wikipedia_summary(company)
    if wiki:
        ctx.company_summary, src = wiki
        ctx.sources.append(f"wikipedia:{src}")

    # Auto web-search from the company name (Issue 1 steps 1–2).
    if auto_search:
        for snip in _collect_web_snippets(company):
            ctx.public_snippets.append(snip)
            ctx.sources.append(f"search:{snip['url']}")

    # User-supplied public URLs (still supported, but no longer required).
    for url in extra_urls:
        text = fetch_public_url(url)
        if text:
            ctx.public_snippets.append({"url": url, "text": text})
            ctx.sources.append(f"public-url:{url}")
        else:
            ctx.sources.append(f"blocked-or-empty:{url}")

    if ctx.manual_notes:
        ctx.sources.append("manual-input")

    # Persist
    try:
        payload = asdict(ctx)
        payload["_key"] = key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - cache failure shouldn't break user
        logger.warning("Failed to write context cache %s: %s", path, exc)

    return ctx
