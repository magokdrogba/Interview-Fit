"""Tests for ``src.crawler.sources``.

Network is mocked via ``polite_fetch``. We test the orchestration logic and
the cache contract; the underlying HTTP layer is covered separately.
"""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.parse import quote

import pytest

from config import CACHE_DIR
from src.crawler import sources


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path, monkeypatch):
    """Redirect the cache dir into a tmp path so tests stay hermetic."""
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
    yield


@pytest.fixture(autouse=True)
def _no_real_web_search(monkeypatch):
    """Stub ``requests.get`` so the auto web-search never hits the network.

    The real ``search_duckduckgo`` / ``fetch_search_result_text`` still run,
    but against an empty 200 response (no result anchors → no snippets), so
    existing tests stay hermetic. Tests that exercise the search path patch
    ``requests.get`` themselves, which overrides this for their duration.
    """
    class _Empty:
        text = "<html><body></body></html>"
        status_code = 200
        ok = True
        url = ""

    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: _Empty())
    yield


def _patch_polite(mapping: dict):
    """Replace polite_fetch so each URL returns a fixed string or None."""
    def fake(url, **kw):
        return mapping.get(url)

    return patch("src.crawler.sources.polite_fetch", side_effect=fake)


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------
def test_wikipedia_summary_picks_first_lang_with_extract():
    payload_ko = json.dumps({"type": "standard", "extract": "토스는 핀테크 회사다.",
                             "content_urls": {"desktop": {"page": "https://ko.wikipedia.org/wiki/Toss"}}})
    payload_en = json.dumps({"type": "standard", "extract": "Toss is a fintech app."})

    mapping = {
        "https://ko.wikipedia.org/api/rest_v1/page/summary/Toss": payload_ko,
        "https://en.wikipedia.org/api/rest_v1/page/summary/Toss": payload_en,
    }
    with _patch_polite(mapping):
        got = sources.fetch_wikipedia_summary("Toss")
    assert got is not None
    text, url = got
    assert "토스" in text
    assert "ko.wikipedia.org" in url


def test_wikipedia_falls_back_to_english_when_ko_missing():
    mapping = {
        "https://en.wikipedia.org/api/rest_v1/page/summary/Acme":
            json.dumps({"type": "standard", "extract": "Acme Corp makes anvils."}),
    }
    with _patch_polite(mapping):
        got = sources.fetch_wikipedia_summary("Acme")
    assert got is not None
    text, _ = got
    assert "anvils" in text


def test_wikipedia_skips_disambiguation_pages():
    mapping = {
        "https://ko.wikipedia.org/api/rest_v1/page/summary/Mercury":
            json.dumps({"type": "disambiguation", "extract": "may refer to"}),
        "https://en.wikipedia.org/api/rest_v1/page/summary/Mercury":
            json.dumps({"type": "standard", "extract": "Mercury is the smallest planet."}),
    }
    with _patch_polite(mapping):
        got = sources.fetch_wikipedia_summary("Mercury")
    assert got is not None and "smallest" in got[0]


def test_wikipedia_returns_none_when_all_blocked():
    with _patch_polite({}):  # everything returns None
        assert sources.fetch_wikipedia_summary("WhoeverCorp") is None


# ---------------------------------------------------------------------------
# Public URL extraction
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Auto web-search (Issue 1) — requests is mocked, no real network
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, text="", status=200, url=""):
        self.text = text
        self.status_code = status
        self.ok = 200 <= status < 300
        self.url = url


def _ddg_html(urls):
    anchors = "".join(
        f'<a class="result__a" href="//duckduckgo.com/l/?uddg={quote(u, safe="")}">r</a>'
        for u in urls
    )
    return f"<html><body>{anchors}</body></html>"


def test_search_duckduckgo_unwraps_links():
    targets = ["https://blog.example/bcg-interview", "https://news.example/bcg"]

    def fake_get(url, **kw):
        return _FakeResp(text=_ddg_html(targets), url=url)

    with patch("src.crawler.sources.requests.get", side_effect=fake_get):
        out = sources.search_duckduckgo("BCG 면접 후기", max_results=3)
    assert out == targets


def test_search_duckduckgo_handles_network_error():
    import requests as _rq

    def boom(url, **kw):
        raise _rq.RequestException("down")

    with patch("src.crawler.sources.requests.get", side_effect=boom):
        assert sources.search_duckduckgo("BCG 면접") == []


def test_fetch_search_result_text_keeps_long_paragraphs():
    html = ("<html><body>"
            "<p>짧음</p>"
            "<p>" + ("가" * 40) + "</p>"
            "<li>" + ("나" * 35) + "</li>"
            "</body></html>")
    url = "https://blog.example/post"

    def fake_get(url_, **kw):
        return _FakeResp(text=html, url=url_)

    with patch("src.crawler.sources.requests.get", side_effect=fake_get):
        text = sources.fetch_search_result_text(url)
    assert text is not None
    assert "가" * 40 in text
    assert "나" * 35 in text
    assert "짧음" not in text  # < 30 chars dropped


def test_fetch_search_result_text_skips_login_wall_keyword():
    html = "<html><body><p>" + ("x" * 40) + " 로그인이 필요합니다</p></body></html>"

    def fake_get(url_, **kw):
        return _FakeResp(text=html, url=url_)

    with patch("src.crawler.sources.requests.get", side_effect=fake_get):
        assert sources.fetch_search_result_text("https://site/login") is None


def test_fetch_search_result_text_skips_cross_host_redirect():
    def fake_get(url_, **kw):
        # Redirected to a different host (login wall pattern).
        return _FakeResp(text="<p>" + ("y" * 40) + "</p>",
                         url="https://accounts.other.com/login")

    with patch("src.crawler.sources.requests.get", side_effect=fake_get):
        assert sources.fetch_search_result_text("https://blog.example/post") is None


def test_fetch_search_result_text_truncates_to_1500():
    html = "<html><body><p>" + ("자" * 3000) + "</p></body></html>"

    def fake_get(url_, **kw):
        return _FakeResp(text=html, url=url_)

    with patch("src.crawler.sources.requests.get", side_effect=fake_get):
        text = sources.fetch_search_result_text("https://blog.example/post")
    assert text is not None and len(text) <= sources._SEARCH_SNIPPET_CHARS


def test_collect_web_snippets_dedupes_urls(monkeypatch):
    # Same URL returned by multiple queries → fetched once.
    monkeypatch.setattr(sources, "search_duckduckgo",
                        lambda q, max_results=3: ["https://dup.example/a"])
    calls = {"n": 0}

    def fake_fetch(url):
        calls["n"] += 1
        return "충분히 긴 본문 내용입니다 " * 3

    monkeypatch.setattr(sources, "fetch_search_result_text", fake_fetch)
    out = sources._collect_web_snippets("BCG")
    assert len(out) == 1
    assert calls["n"] == 1  # deduped across the 4 queries


def test_collect_interview_context_uses_auto_search(monkeypatch):
    monkeypatch.setattr(
        sources, "_collect_web_snippets",
        lambda company: [{"url": "https://blog.example/bcg", "text": "BCG 면접은 케이스 중심입니다."}],
    )
    with _patch_polite({}):  # no wikipedia
        ctx = sources.collect_interview_context("BCG")
    assert any("BCG 면접은 케이스" in s["text"] for s in ctx.public_snippets)
    assert any(s.startswith("search:") for s in ctx.sources)
    assert not ctx.is_empty()


def test_fetch_public_url_strips_boilerplate():
    html = """
    <html><head><script>track()</script></head>
    <body>
      <nav>menu menu</nav>
      <main>
        <h1>Our values</h1>
        <p>We obsess over the customer.</p>
        <p>We move fast and write tests.</p>
      </main>
      <footer>© 2026</footer>
    </body></html>
    """
    with _patch_polite({"https://acme.com/about": html}):
        text = sources.fetch_public_url("https://acme.com/about")
    assert text is not None
    assert "Our values" in text
    assert "customer" in text
    assert "menu menu" not in text  # nav stripped
    assert "© 2026" not in text     # footer stripped
    assert "track()" not in text    # script stripped


def test_fetch_public_url_truncates_long_pages():
    huge = "<html><body><main>" + ("AAAA " * 5_000) + "</main></body></html>"
    with _patch_polite({"https://big.example/page": huge}):
        text = sources.fetch_public_url("https://big.example/page")
    assert text is not None
    assert len(text) <= sources._MAX_SNIPPET_CHARS


def test_fetch_public_url_returns_none_on_block():
    with _patch_polite({}):
        assert sources.fetch_public_url("https://blocked.example/page") is None


# ---------------------------------------------------------------------------
# Orchestrator + cache
# ---------------------------------------------------------------------------
def test_collect_returns_context_with_manual_only_offline():
    with _patch_polite({}):
        ctx = sources.collect_interview_context(
            "Toss",
            job_posting="Backend engineer, payments",
            manual_input="They asked about distributed transactions.",
            extra_urls=[],
        )
    assert ctx.company == "Toss"
    assert ctx.manual_notes.startswith("They asked")
    assert ctx.company_summary == ""
    assert ctx.public_snippets == []
    assert "manual-input" in ctx.sources
    assert not ctx.is_empty()  # job_posting + manual notes count


def test_collect_merges_all_sources():
    wiki_ko = json.dumps({"type": "standard", "extract": "토스는 핀테크 회사."})
    page = "<html><body><main><p>We obsess over the customer.</p></main></body></html>"
    mapping = {
        "https://ko.wikipedia.org/api/rest_v1/page/summary/Toss": wiki_ko,
        "https://acme.com/about": page,
    }
    with _patch_polite(mapping):
        ctx = sources.collect_interview_context(
            "Toss",
            manual_input="Note.",
            extra_urls=["https://acme.com/about"],
        )
    assert "핀테크" in ctx.company_summary
    assert len(ctx.public_snippets) == 1
    assert "customer" in ctx.public_snippets[0]["text"]
    # Source trail records each origin distinctly.
    assert any(s.startswith("wikipedia:") for s in ctx.sources)
    assert any(s.startswith("public-url:") for s in ctx.sources)
    assert "manual-input" in ctx.sources


def test_collect_records_blocked_urls_in_source_trail():
    with _patch_polite({}):
        ctx = sources.collect_interview_context(
            "Acme",
            extra_urls=["https://blocked.example/x"],
        )
    assert any(s.startswith("blocked-or-empty:") for s in ctx.sources)
    assert ctx.public_snippets == []


def test_collect_caches_and_reuses(monkeypatch):
    calls = {"n": 0}

    def fake(url, **kw):
        calls["n"] += 1
        if "wikipedia" in url:
            return json.dumps({"type": "standard", "extract": "Acme blurb."})
        return None

    with patch("src.crawler.sources.polite_fetch", side_effect=fake):
        ctx1 = sources.collect_interview_context("Acme")
        first_calls = calls["n"]
        ctx2 = sources.collect_interview_context("Acme")  # should hit cache

    assert ctx1.company_summary == ctx2.company_summary
    # The second call must not have issued *any* additional polite_fetch.
    assert calls["n"] == first_calls


def test_collect_force_refresh_bypasses_cache():
    def fake(url, **kw):
        if "wikipedia" in url:
            return json.dumps({"type": "standard", "extract": "v1"})
        return None

    # Seed cache
    with patch("src.crawler.sources.polite_fetch", side_effect=fake):
        sources.collect_interview_context("Acme")

    # New polite returns different content; force_refresh must pick it up.
    def fake2(url, **kw):
        if "wikipedia" in url:
            return json.dumps({"type": "standard", "extract": "v2"})
        return None

    with patch("src.crawler.sources.polite_fetch", side_effect=fake2):
        ctx = sources.collect_interview_context("Acme", force_refresh=True)
    assert "v2" in ctx.company_summary


def test_collect_cache_invalidates_on_manual_change():
    with patch("src.crawler.sources.polite_fetch", return_value=None):
        a = sources.collect_interview_context("Acme", manual_input="first")
        b = sources.collect_interview_context("Acme", manual_input="second")
    assert a.manual_notes == "first"
    assert b.manual_notes == "second"


# ---------------------------------------------------------------------------
# InterviewContext helpers
# ---------------------------------------------------------------------------
def test_context_to_prompt_block_includes_present_sections():
    ctx = sources.InterviewContext(
        company="Toss",
        job_posting="Backend",
        company_summary="Fintech.",
        public_snippets=[{"url": "https://x", "text": "About us."}],
        manual_notes="Asked SQL.",
    )
    block = ctx.to_prompt_block()
    for needle in ("Toss", "Backend", "Fintech.", "https://x", "About us.", "Asked SQL."):
        assert needle in block


def test_context_is_empty_flag():
    bare = sources.InterviewContext(company="X")
    assert bare.is_empty()
    with_summary = sources.InterviewContext(company="X", company_summary="hi")
    assert not with_summary.is_empty()


def test_collect_rejects_empty_company():
    with pytest.raises(ValueError):
        sources.collect_interview_context("   ")
