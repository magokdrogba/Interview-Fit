"""Tests for ``src.question.generator``.

We never call the real OpenAI API. A ``FakeClient`` records calls and replies
from a programmable script.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.crawler.sources import InterviewContext
from src.question import generator as gen


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClient:
    """Records every call and replies with the next scripted response."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        # Mimic the openai SDK surface: client.chat.completions.create(...)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, response_format, temperature):
        self.calls.append(
            {"model": model, "messages": messages,
             "response_format": response_format, "temperature": temperature}
        )
        if not self._replies:
            raise AssertionError("FakeClient ran out of scripted replies")
        reply = self._replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


def _good_question(category: str = "experience", n_followups: int = 2) -> dict:
    return {
        "category": category,
        "question": f"Tell me about a {category} project.",
        "rationale": "Resume mentions ACME Corp work in 2022.",
        "followups": [f"Probe {i+1}?" for i in range(n_followups)],
    }


def _good_payload(n: int = 10) -> str:
    # Distribute categories so we always have at least 2 of each kind we use.
    cats = ["personality", "domain", "experience", "case"] * 5
    return json.dumps({"questions": [_good_question(cats[i]) for i in range(n)]})


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the cache dir into a tmp path."""
    monkeypatch.setattr(gen, "CACHE_DIR", tmp_path)
    yield


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_happy_path_returns_validated_list():
    client = FakeClient([_good_payload(10)])
    out = gen.generate_questions(
        resume_text="Jane Doe — ACME Corp",
        company="Toss",
        job_posting="Backend",
        interview_context=None,
        client=client,
        n_questions=10,
    )
    assert isinstance(out, list)
    assert len(out) == 10
    assert all(q["category"] in gen.ALLOWED_CATEGORIES for q in out)
    assert all(len(q["followups"]) in (2, 3) for q in out)
    assert len(client.calls) == 1


def test_resume_text_appears_in_user_prompt():
    client = FakeClient([_good_payload(10)])
    gen.generate_questions(
        resume_text="Led migration to event-sourced billing at ACME Corp.",
        company="Toss",
        job_posting="",
        client=client,
    )
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "ACME Corp" in user_msg
    assert "event-sourced billing" in user_msg


def test_interview_context_renders_into_prompt():
    ctx = InterviewContext(
        company="Toss",
        company_summary="Korean fintech app.",
        public_snippets=[{"url": "https://x", "text": "About page."}],
        manual_notes="Asks SQL.",
    )
    client = FakeClient([_good_payload(10)])
    gen.generate_questions(
        resume_text="resume body",
        company="Toss",
        job_posting="",
        interview_context=ctx,
        client=client,
    )
    user_msg = client.calls[0]["messages"][1]["content"]
    for needle in ("Korean fintech app.", "About page.", "Asks SQL."):
        assert needle in user_msg


def test_n_questions_clamped_to_min_and_max():
    client = FakeClient([_good_payload(gen.MAX_QUESTIONS)])
    gen.generate_questions(
        resume_text="x", company="y", job_posting="",
        client=client, n_questions=999,
    )
    user_msg = client.calls[0]["messages"][1]["content"]
    assert f"Number of questions to generate: {gen.MAX_QUESTIONS}" in user_msg

    client2 = FakeClient([_good_payload(gen.MIN_QUESTIONS)])
    gen.generate_questions(
        resume_text="x", company="y2", job_posting="",
        client=client2, n_questions=0,
    )
    user_msg2 = client2.calls[0]["messages"][1]["content"]
    assert f"Number of questions to generate: {gen.MIN_QUESTIONS}" in user_msg2


def test_accepts_bare_array_response():
    # Be liberal in what we accept: a bare list with a valid count is OK.
    bare = json.dumps([_good_question("personality") for _ in range(10)])
    client = FakeClient([bare])
    out = gen.generate_questions(
        resume_text="x", company="y", job_posting="",
        client=client,
    )
    assert len(out) == 10


# ---------------------------------------------------------------------------
# Retry & error paths
# ---------------------------------------------------------------------------
def test_retries_once_on_invalid_json_then_succeeds():
    client = FakeClient(["not json at all", _good_payload(10)])
    out = gen.generate_questions(
        resume_text="x", company="y", job_posting="",
        client=client,
    )
    assert len(out) == 10
    assert len(client.calls) == 2
    # Second attempt must have appended the corrective user message.
    second_msgs = client.calls[1]["messages"]
    assert second_msgs[-1]["role"] == "user"
    assert "schema validator" in second_msgs[-1]["content"]


def test_retries_once_on_schema_violation_then_succeeds():
    # First reply: only 3 questions (below minimum)
    bad = json.dumps({"questions": [_good_question("domain") for _ in range(3)]})
    client = FakeClient([bad, _good_payload(10)])
    out = gen.generate_questions(
        resume_text="x", company="y", job_posting="",
        client=client,
    )
    assert len(out) == 10
    assert len(client.calls) == 2


def test_two_strikes_raises_generation_error():
    bad = json.dumps({"questions": [_good_question("domain") for _ in range(3)]})
    client = FakeClient([bad, bad])
    with pytest.raises(gen.QuestionGenerationError):
        gen.generate_questions(
            resume_text="x", company="y", job_posting="",
            client=client,
        )
    assert len(client.calls) == 2


def test_rejects_unknown_category():
    bad_q = _good_question("experience")
    bad_q["category"] = "philosophy"
    payload = json.dumps({"questions": [bad_q] + [_good_question("domain") for _ in range(9)]})
    client = FakeClient([payload, payload])  # both attempts bad
    with pytest.raises(gen.QuestionGenerationError):
        gen.generate_questions(
            resume_text="x", company="y", job_posting="",
            client=client,
        )


def test_rejects_too_few_followups():
    bad_q = _good_question("experience", n_followups=1)
    payload = json.dumps({"questions": [bad_q] + [_good_question("domain") for _ in range(9)]})
    client = FakeClient([payload, payload])
    with pytest.raises(gen.QuestionGenerationError):
        gen.generate_questions(
            resume_text="x", company="y", job_posting="",
            client=client,
        )


def test_missing_resume_raises_early():
    client = FakeClient([_good_payload(10)])
    with pytest.raises(ValueError):
        gen.generate_questions(
            resume_text="", company="y", job_posting="", client=client,
        )
    assert client.calls == []


def test_missing_company_raises_early():
    client = FakeClient([_good_payload(10)])
    with pytest.raises(ValueError):
        gen.generate_questions(
            resume_text="x", company="   ", job_posting="", client=client,
        )
    assert client.calls == []


def test_default_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(gen, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        gen.generate_questions(
            resume_text="x", company="y", job_posting="",
            # client=None → goes through _default_client
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_hit_skips_second_api_call():
    client1 = FakeClient([_good_payload(10)])
    out1 = gen.generate_questions(
        resume_text="r", company="Toss", job_posting="j", client=client1,
    )
    # Second call with same inputs — second client must NOT be invoked.
    client2 = FakeClient([])  # zero replies; would raise if used
    out2 = gen.generate_questions(
        resume_text="r", company="Toss", job_posting="j", client=client2,
    )
    assert out1 == out2
    assert client2.calls == []


def test_force_refresh_bypasses_cache():
    client1 = FakeClient([_good_payload(10)])
    gen.generate_questions(
        resume_text="r", company="Toss", job_posting="j", client=client1,
    )

    payload2 = json.dumps({"questions": [_good_question("case") for _ in range(10)]})
    client2 = FakeClient([payload2])
    out2 = gen.generate_questions(
        resume_text="r", company="Toss", job_posting="j",
        client=client2, force_refresh=True,
    )
    assert all(q["category"] == "case" for q in out2)
    assert len(client2.calls) == 1


def test_cache_invalidates_on_resume_change():
    client1 = FakeClient([_good_payload(10)])
    gen.generate_questions(
        resume_text="resume v1", company="Toss", job_posting="j", client=client1,
    )
    client2 = FakeClient([_good_payload(10)])  # must be hit because resume changed
    gen.generate_questions(
        resume_text="resume v2", company="Toss", job_posting="j", client=client2,
    )
    assert len(client2.calls) == 1
