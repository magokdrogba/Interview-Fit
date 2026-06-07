"""Tests for ``src.crawler.base``.

All HTTP is mocked. We never touch the network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from src.crawler import base


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.ok = 200 <= status_code < 300


@pytest.fixture(autouse=True)
def _reset_caches():
    base._clear_caches()
    yield
    base._clear_caches()


# ---------------------------------------------------------------------------
# is_allowed_by_robots
# ---------------------------------------------------------------------------
def test_robots_allow_when_no_rules():
    def fake_get(url, **kw):
        return _FakeResponse(404)  # no robots.txt published

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        assert base.is_allowed_by_robots("https://example.com/anything") is True


def test_robots_disallow_blocks_path():
    body = "User-agent: *\nDisallow: /private\n"

    def fake_get(url, **kw):
        assert url.endswith("/robots.txt")
        return _FakeResponse(200, body)

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        assert base.is_allowed_by_robots("https://example.com/private/secret") is False
        assert base.is_allowed_by_robots("https://example.com/public/ok") is True


def test_robots_total_disallow_on_401_403():
    def fake_get(url, **kw):
        return _FakeResponse(403)

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        assert base.is_allowed_by_robots("https://example.com/anywhere") is False


def test_robots_allow_on_network_error():
    def fake_get(url, **kw):
        raise requests.ConnectionError("boom")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        # Per RFC 9309 §2.3 / Google convention: unreachable robots → allow.
        assert base.is_allowed_by_robots("https://example.com/anything") is True


def test_robots_cache_avoids_refetch():
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return _FakeResponse(200, "User-agent: *\nAllow: /\n")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        base.is_allowed_by_robots("https://example.com/a")
        base.is_allowed_by_robots("https://example.com/b")
        base.is_allowed_by_robots("https://example.com/c")

    assert calls["n"] == 1  # one robots fetch shared across 3 checks


def test_non_http_url_rejected():
    assert base.is_allowed_by_robots("file:///etc/passwd") is False
    assert base.is_allowed_by_robots("ftp://example.com/x") is False
    assert base.is_allowed_by_robots("not-a-url") is False


# ---------------------------------------------------------------------------
# polite_fetch
# ---------------------------------------------------------------------------
def test_polite_fetch_returns_text_on_success():
    def fake_get(url, **kw):
        if url.endswith("/robots.txt"):
            return _FakeResponse(404)
        return _FakeResponse(200, "<html>hi</html>")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        out = base.polite_fetch("https://example.com/page")
    assert out == "<html>hi</html>"


def test_polite_fetch_blocked_by_robots_returns_none():
    def fake_get(url, **kw):
        if url.endswith("/robots.txt"):
            return _FakeResponse(200, "User-agent: *\nDisallow: /\n")
        raise AssertionError("Should not fetch content when robots blocks it")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        assert base.polite_fetch("https://example.com/anything") is None


def test_polite_fetch_swallows_network_errors():
    def fake_get(url, **kw):
        if url.endswith("/robots.txt"):
            return _FakeResponse(404)
        raise requests.Timeout("slow")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        assert base.polite_fetch("https://example.com/page") is None


def test_polite_fetch_swallows_non_2xx():
    def fake_get(url, **kw):
        if url.endswith("/robots.txt"):
            return _FakeResponse(404)
        return _FakeResponse(500, "oops")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        assert base.polite_fetch("https://example.com/page") is None


def test_polite_fetch_rejects_non_http():
    # Should not touch the network at all.
    with patch("src.crawler.base.requests.get") as g:
        assert base.polite_fetch("file:///etc/passwd") is None
        g.assert_not_called()


def test_polite_fetch_sets_user_agent():
    seen: dict = {}

    def fake_get(url, **kw):
        seen[url] = kw.get("headers", {})
        if url.endswith("/robots.txt"):
            return _FakeResponse(404)
        return _FakeResponse(200, "ok")

    with patch("src.crawler.base.requests.get", side_effect=fake_get):
        base.polite_fetch("https://example.com/page")

    ua = seen["https://example.com/page"]["User-Agent"]
    assert "ai-mock-interview" in ua
