"""Tests for the community feature's pure logic (no Supabase/Streamlit runtime).

UI rendering (render_write_form / render_feed) needs a live Streamlit session,
so we cover only the deterministic, side-effect-free helpers here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.community import db
from src.community.feed import _relative_time
from src.community.write import _author_hash


# ---------------------------------------------------------------------------
# db.get_client — graceful degradation
# ---------------------------------------------------------------------------
def test_get_client_returns_none_without_config(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    # _secret() also tries st.secrets, which raises outside a Streamlit run and
    # is swallowed — so the result must be None, never an exception.
    assert db.get_client() is None


# ---------------------------------------------------------------------------
# write._author_hash — anonymous, deterministic identifier
# ---------------------------------------------------------------------------
def test_author_hash_is_deterministic_and_short():
    h1 = _author_hash("지원자")
    h2 = _author_hash("지원자")
    assert h1 == h2
    assert len(h1) == 12
    assert all(c in "0123456789abcdef" for c in h1)


def test_author_hash_differs_by_nickname():
    assert _author_hash("alice") != _author_hash("bob")


def test_author_hash_does_not_leak_nickname():
    # The raw nickname must not appear in the hash.
    nick = "secretname"
    assert nick not in _author_hash(nick)


# ---------------------------------------------------------------------------
# feed._relative_time — Korean relative timestamps
# ---------------------------------------------------------------------------
def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) - delta).isoformat()


def test_relative_time_just_now():
    assert _relative_time(_iso(timedelta(seconds=5))) == "방금 전"


def test_relative_time_minutes():
    assert _relative_time(_iso(timedelta(minutes=10))) == "10분 전"


def test_relative_time_hours():
    assert _relative_time(_iso(timedelta(hours=3))) == "3시간 전"


def test_relative_time_days():
    assert _relative_time(_iso(timedelta(days=3))) == "3일 전"


def test_relative_time_handles_zulu_suffix():
    # Supabase often returns a trailing 'Z'.
    ts = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _relative_time(ts) == "2분 전"


def test_relative_time_empty_is_blank():
    assert _relative_time(None) == ""
    assert _relative_time("") == ""
