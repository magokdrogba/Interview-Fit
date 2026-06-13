"""Tests for the community feature's pure logic (no Supabase/Streamlit runtime).

UI rendering (render_write_form / render_feed) needs a live Streamlit session,
so we cover only the deterministic, side-effect-free helpers here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.community import db
from src.community.feed import _relative_time
from src.community.gate import has_posted
from src.community.write import _author_hash


# ---------------------------------------------------------------------------
# Fake Supabase query builder for gate.has_posted
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Records the fluent chain and returns a scripted result (or raises)."""

    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.eq_args: tuple | None = None

    def select(self, *_a, **_k):
        return self

    def eq(self, *args):
        self.eq_args = args
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        if self._error is not None:
            raise self._error
        return _FakeResp(self._result)


class _FakeClient:
    def __init__(self, query: _FakeQuery):
        self._query = query

    def table(self, _name):
        return self._query


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
# gate.has_posted — DB-backed membership check
# ---------------------------------------------------------------------------
def test_has_posted_true_when_rows_exist():
    q = _FakeQuery(result=[{"id": 1}])
    assert has_posted(_FakeClient(q), "user-123") is True
    assert q.eq_args == ("user_id", "user-123")  # filters on the right column


def test_has_posted_false_when_no_rows():
    assert has_posted(_FakeClient(_FakeQuery(result=[])), "user-123") is False


def test_has_posted_false_on_db_error():
    q = _FakeQuery(error=RuntimeError("network down"))
    assert has_posted(_FakeClient(q), "user-123") is False


def test_has_posted_handles_none_data():
    assert has_posted(_FakeClient(_FakeQuery(result=None)), "u") is False


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
