"""Shared HTTP utilities for the collection phase.

Two public functions:
  * :func:`is_allowed_by_robots` — robots.txt check with in-memory caching
    and a soft TTL. Failures are interpreted as *allow* (Google's policy) so
    that a temporarily unreachable robots.txt does not silently block
    everything, but real disallows are honored.
  * :func:`polite_fetch` — robots-respecting GET with a per-host minimum
    delay, explicit User-Agent, sane timeout. **Never raises**: returns the
    response text on success and ``None`` on any failure (block, network
    error, non-2xx). Caller decides what to do with that.

This module intentionally has no knowledge of any specific site. Site- or
API-specific logic lives in :mod:`src.crawler.sources`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Final
from urllib import robotparser
from urllib.parse import urlparse, urlunparse

import requests

from config import HTTP_REQUEST_DELAY_SECONDS, HTTP_USER_AGENT

logger = logging.getLogger(__name__)

# Robots cache: host -> (parsed_robots, fetched_at_epoch). Cached for 24h.
_ROBOTS_TTL_SECONDS: Final[float] = 24 * 60 * 60
_robots_cache: dict[str, tuple[robotparser.RobotFileParser, float]] = {}
_robots_lock = threading.Lock()

# Per-host last-fetch timestamps, so polite_fetch can rate-limit per origin.
_last_fetch_at: dict[str, float] = {}
_fetch_lock = threading.Lock()

# Default HTTP timeout for both robots and content fetches.
_HTTP_TIMEOUT_SECONDS: Final[float] = 10.0


@dataclass(frozen=True)
class _Origin:
    scheme: str
    netloc: str

    @property
    def base(self) -> str:
        return f"{self.scheme}://{self.netloc}"


def _origin_of(url: str) -> _Origin | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return _Origin(parsed.scheme, parsed.netloc)


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------
def _load_robots(origin: _Origin) -> robotparser.RobotFileParser:
    """Fetch and parse robots.txt for ``origin``, caching for 24h.

    On any fetch failure we return an empty parser; ``can_fetch`` on an
    empty parser returns True, matching the documented Google convention
    of allowing access when robots.txt is unreachable.
    """
    now = time.time()
    with _robots_lock:
        cached = _robots_cache.get(origin.netloc)
        if cached and (now - cached[1]) < _ROBOTS_TTL_SECONDS:
            return cached[0]

    rp = robotparser.RobotFileParser()
    robots_url = urlunparse((origin.scheme, origin.netloc, "/robots.txt", "", "", ""))
    rp.set_url(robots_url)
    try:
        # Avoid robotparser.read() (uses urllib directly with no UA / timeout).
        resp = requests.get(
            robots_url,
            headers={"User-Agent": HTTP_USER_AGENT},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            # 4xx auth-style responses → treat site as fully disallowed,
            # per RFC 9309 §2.3.
            rp.parse(["User-agent: *", "Disallow: /"])
        else:
            # 404 or any other status → allow (no rules published).
            rp.parse([])
    except requests.RequestException as exc:
        logger.warning("robots.txt fetch failed for %s: %s — allowing", origin.netloc, exc)
        rp.parse([])

    with _robots_lock:
        _robots_cache[origin.netloc] = (rp, now)
    return rp


def is_allowed_by_robots(url: str, user_agent: str = HTTP_USER_AGENT) -> bool:
    """Return True if robots.txt permits ``user_agent`` to fetch ``url``."""
    origin = _origin_of(url)
    if origin is None:
        return False
    rp = _load_robots(origin)
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:  # pragma: no cover - robotparser is permissive but be safe
        return True


# ---------------------------------------------------------------------------
# polite GET
# ---------------------------------------------------------------------------
def _wait_for_rate_limit(host: str) -> None:
    """Block until at least HTTP_REQUEST_DELAY_SECONDS have elapsed since
    the last fetch to ``host``. Updates the timestamp before returning."""
    now = time.monotonic()
    with _fetch_lock:
        last = _last_fetch_at.get(host)
        wait = 0.0 if last is None else max(0.0, HTTP_REQUEST_DELAY_SECONDS - (now - last))
    if wait > 0:
        time.sleep(wait)
    with _fetch_lock:
        _last_fetch_at[host] = time.monotonic()


def polite_fetch(
    url: str,
    *,
    accept: str = "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5",
    headers: dict[str, str] | None = None,
) -> str | None:
    """Fetch ``url`` politely. Returns response text on success, else None.

    Order of operations:
      1. Reject non-http(s) URLs.
      2. Check robots.txt — if disallowed, return None and log.
      3. Sleep to satisfy the per-host rate limit.
      4. Issue the GET with explicit UA and a 10s timeout.
      5. Return ``resp.text`` if 2xx, else None.
    """
    origin = _origin_of(url)
    if origin is None:
        logger.info("polite_fetch: rejecting non-http(s) URL %r", url)
        return None

    if not is_allowed_by_robots(url):
        logger.info("polite_fetch: robots.txt disallows %s", url)
        return None

    _wait_for_rate_limit(origin.netloc)

    merged_headers = {"User-Agent": HTTP_USER_AGENT, "Accept": accept}
    if headers:
        merged_headers.update(headers)

    try:
        resp = requests.get(url, headers=merged_headers, timeout=_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.info("polite_fetch: network error for %s: %s", url, exc)
        return None

    if not resp.ok:
        logger.info("polite_fetch: %s returned HTTP %s", url, resp.status_code)
        return None

    return resp.text


# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------
def _clear_caches() -> None:
    """Reset both in-memory caches. Used by tests; not part of the public API."""
    with _robots_lock:
        _robots_cache.clear()
    with _fetch_lock:
        _last_fetch_at.clear()
