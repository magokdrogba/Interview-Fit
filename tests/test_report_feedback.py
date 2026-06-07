"""Tests for ``src.report.feedback``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.report import feedback as F


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, response_format, temperature):
        self.calls.append({"model": model, "messages": messages,
                           "response_format": response_format})
        if not self._replies:
            raise AssertionError("FakeClient ran out of replies")
        reply = self._replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


class FlexibleClient:
    """Fake chat client that tolerates calls with or without response_format."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._replies:
            raise AssertionError("FlexibleClient ran out of replies")
        reply = self._replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


def _sample_analysis() -> dict:
    """Two answered questions with all three axes populated."""
    return {
        "status": "ok",
        "session_id": "20260530-160000",
        "company": "Toss",
        "per_question": [
            {
                "index": 0, "category": "personality",
                "question": "Tell me about yourself.",
                "duration_s": 30.0, "skipped": False, "aborted": False,
                "status": "ok",
                "audio": {
                    "status": "ok", "duration_s": 30.0,
                    "speech": {"speech_ratio": 0.7, "pause_count_3s": 1,
                               "hesitation_before_speech_s": 1.2},
                    "rate": {"syllables_per_minute": 230.0, "rate_label": "normal"},
                    "voice": {"tremor_index": 0.25},
                },
                "vision": {
                    "status": "ok",
                    "gaze": {"looking_ratio": 0.65, "gaze_away_events": 5},
                    "expression": {"positive_ratio": 0.4,
                                   "neutral_ratio": 0.5, "tense_ratio": 0.1},
                    "head": {"yaw_std_deg": 6.0, "direction_changes_per_min": 12.0},
                },
                "language": {
                    "status": "ok",
                    "transcript": {"text": "Um, so I'm a backend engineer.",
                                   "language": "en", "syllable_count": 12,
                                   "word_count": 6, "char_count": 30},
                    "fillers": {"total": 1, "per_minute": 2.0,
                                "counts": {"um": 1}},
                    "repetition": {"top": [], "type_token_ratio": 0.95},
                    "structure": {"status": "ok", "score": 4,
                                  "reason": "두괄식으로 시작했습니다.", "comment": "두괄식으로 시작했습니다."},
                },
            },
            {
                "index": 1, "category": "experience",
                "question": "Walk me through a project.",
                "duration_s": 60.0, "skipped": False, "aborted": False,
                "status": "ok",
                "audio": {
                    "status": "ok", "duration_s": 60.0,
                    "speech": {"speech_ratio": 0.6, "pause_count_3s": 3,
                               "hesitation_before_speech_s": 3.4},
                    "rate": {"syllables_per_minute": 180.0, "rate_label": "slow"},
                    "voice": {"tremor_index": 0.5},
                },
                "vision": {
                    "status": "ok",
                    "gaze": {"looking_ratio": 0.40, "gaze_away_events": 14},
                    "expression": {"positive_ratio": 0.1,
                                   "neutral_ratio": 0.4, "tense_ratio": 0.5},
                    "head": {"yaw_std_deg": 12.0, "direction_changes_per_min": 30.0},
                },
                "language": {
                    "status": "ok",
                    "transcript": {"text": "I worked on the billing migration.",
                                   "language": "en", "syllable_count": 14,
                                   "word_count": 7, "char_count": 35},
                    "fillers": {"total": 6, "per_minute": 8.0,
                                "counts": {"so": 3, "like": 3}},
                    "repetition": {"top": [{"word": "billing", "count": 4}],
                                   "type_token_ratio": 0.7},
                    "structure": {"status": "ok", "score": 3,
                                  "reason": "서술형으로 흘러갑니다.", "comment": "서술형으로 흘러갑니다."},
                },
            },
        ],
        "aggregate": {
            "status": "ok", "answered": 2,
            "vision": {"looking_ratio_mean": 0.525,
                       "gaze_away_events_total": 19,
                       "positive_ratio_mean": 0.25,
                       "tense_ratio_mean": 0.30},
            "audio": {"speech_ratio_mean": 0.65,
                      "pause_count_3s_total": 4,
                      "hesitation_mean_s": 2.3,
                      "syllables_per_minute_mean": 205.0,
                      "tremor_index_mean": 0.375},
            "language": {"fillers_total": 7,
                         "fillers_per_minute_mean": 5.0,
                         "structure_score_mean": 3.5,
                         "conclusion_first_ratio": 0.5},
        },
        "notes": [],
    }


def _good_report() -> str:
    return json.dumps({
        "overall_summary": "You spoke at a normal pace but your eye contact "
                           "dropped on the experience question.",
        "per_question": [
            {"index": 0, "comment": "Strong opener at 230 SPM with 65% camera contact."},
            {"index": 1, "comment": "Looking ratio fell to 40% and 14 gaze-away events."},
        ],
        "priorities": [
            {"area": "vision",
             "observation": "Looking ratio averaged 53% — well below the 80% target.",
             "action": "Practice anchoring your gaze on the camera lens for 5 sec at a time."},
            {"area": "language",
             "observation": "Fillers reached 8/min on the experience answer.",
             "action": "Pause silently instead of filling — record yourself and listen back."},
            {"area": "audio",
             "observation": "Three 3+ second pauses on Q1.",
             "action": "Pre-write a 30-second STAR outline for project answers."},
        ],
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_build_report_happy_path():
    client = FakeClient([_good_report()])
    report = F.build_report(_sample_analysis(), client=client)
    assert report["overall_summary"]
    assert len(report["per_question"]) == 2
    assert [c["index"] for c in report["per_question"]] == [0, 1]
    assert len(report["priorities"]) == 3
    assert {p["area"] for p in report["priorities"]} <= F.ALLOWED_AREAS
    assert report["session_id"] == "20260530-160000"
    assert report["company"] == "Toss"
    assert len(client.calls) == 1


def test_user_prompt_includes_aggregate_numbers():
    client = FakeClient([_good_report()])
    F.build_report(_sample_analysis(), client=client)
    user_msg = client.calls[0]["messages"][1]["content"]
    # Aggregate JSON block must appear verbatim.
    assert "looking_ratio_mean" in user_msg
    # Per-question transcripts get rendered.
    assert "billing migration" in user_msg
    # Structure rubric reason surfaces in the prompt.
    assert "서술형으로 흘러갑니다." in user_msg


# ---------------------------------------------------------------------------
# Retry / schema
# ---------------------------------------------------------------------------
def test_retries_on_wrong_priority_count_then_succeeds():
    bad = json.dumps({
        "overall_summary": "x",
        "per_question": [{"index": 0, "comment": "a"}, {"index": 1, "comment": "b"}],
        "priorities": [{"area": "vision", "observation": "o", "action": "a"}],
    })
    client = FakeClient([bad, _good_report()])
    report = F.build_report(_sample_analysis(), client=client)
    assert len(report["priorities"]) == 3
    assert len(client.calls) == 2
    # Retry user message must reference the schema error.
    retry_msg = client.calls[1]["messages"][-1]["content"]
    assert "3 priorities" in retry_msg


def test_retries_on_mismatched_per_question_indices():
    bad = json.dumps({
        "overall_summary": "x",
        "per_question": [{"index": 0, "comment": "a"}],  # missing Q1
        "priorities": [
            {"area": "vision", "observation": "o", "action": "a"},
            {"area": "audio", "observation": "o", "action": "a"},
            {"area": "language", "observation": "o", "action": "a"},
        ],
    })
    client = FakeClient([bad, _good_report()])
    F.build_report(_sample_analysis(), client=client)
    assert len(client.calls) == 2


def test_rejects_unknown_priority_area():
    bad = json.dumps({
        "overall_summary": "x",
        "per_question": [{"index": 0, "comment": "a"}, {"index": 1, "comment": "b"}],
        "priorities": [
            {"area": "posture", "observation": "o", "action": "a"},
            {"area": "audio", "observation": "o", "action": "a"},
            {"area": "language", "observation": "o", "action": "a"},
        ],
    })
    client = FakeClient([bad, bad])
    with pytest.raises(F.ReportGenerationError):
        F.build_report(_sample_analysis(), client=client)


def test_two_strikes_raises():
    client = FakeClient(["not json", "still not json"])
    with pytest.raises(F.ReportGenerationError):
        F.build_report(_sample_analysis(), client=client)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_round_trip(tmp_path):
    client1 = FakeClient([_good_report()])
    a = F.build_report(_sample_analysis(), session_dir=tmp_path, client=client1)
    assert (tmp_path / "report.json").exists()

    client2 = FakeClient([])  # zero replies — would raise if hit
    b = F.build_report(_sample_analysis(), session_dir=tmp_path, client=client2)
    assert b["overall_summary"] == a["overall_summary"]
    assert b.get("_from_cache") is True
    assert client2.calls == []


def test_force_refresh_bypasses_cache(tmp_path):
    client1 = FakeClient([_good_report()])
    F.build_report(_sample_analysis(), session_dir=tmp_path, client=client1)

    new_payload = json.loads(_good_report())
    new_payload["overall_summary"] = "Refreshed."
    client2 = FakeClient([json.dumps(new_payload)])
    b = F.build_report(_sample_analysis(), session_dir=tmp_path,
                       client=client2, force_refresh=True)
    assert b["overall_summary"] == "Refreshed."
    assert len(client2.calls) == 1


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------
def test_no_data_returns_fallback_without_calling_model():
    analysis = {"status": "ok", "session_id": "s",
                "company": "Acme", "per_question": [],
                "aggregate": {"status": "no-data", "answered": 0}, "notes": []}
    client = FakeClient([])  # would raise if hit
    report = F.build_report(analysis, client=client)
    assert report["priorities"] == []
    assert "분석할 수 있는 답변이 없" in report["overall_summary"]
    assert client.calls == []


def test_bad_analysis_status_returns_fallback():
    client = FakeClient([])
    report = F.build_report({"status": "no-manifest"}, client=client)
    assert report["priorities"] == []
    assert client.calls == []


def test_default_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(F, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        F.build_report(_sample_analysis())


def test_rejects_non_dict_analysis():
    with pytest.raises(TypeError):
        F.build_report("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# generate_question_feedback (Issue 4)
# ---------------------------------------------------------------------------
def _feedback_text() -> str:
    return (
        "[잘한 점] 결제 시스템 마이그레이션 사례를 구체적으로 언급한 점이 좋았습니다.\n"
        "[개선점] 결론을 먼저 말한 뒤 근거를 제시하세요.\n"
        "[바로 적용] 다음 답변은 첫 문장에 핵심 결론을 담아 시작하세요."
    )


def test_question_feedback_happy_path():
    client = FlexibleClient([_feedback_text()])
    q = {"index": 0, "question": "프로젝트 경험을 말씀해 주세요."}
    out = F.generate_question_feedback(
        q, {"gaze": 0.7, "fillers": 1.2}, "저는 결제 마이그레이션을 맡았습니다.",
        client=client,
    )
    assert out["status"] == "ok"
    assert "잘한 점" in out["text"]
    assert "바로 적용" in out["text"]
    assert len(client.calls) == 1
    # Prompt must carry the question, the transcript, and the metrics.
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "프로젝트 경험" in user_msg
    assert "결제 마이그레이션" in user_msg
    assert "gaze" in user_msg


def test_question_feedback_empty_transcript_skips_api():
    client = FlexibleClient([])  # would raise if called
    out = F.generate_question_feedback({"index": 1}, {}, "   ", client=client)
    assert out["status"] == "empty"
    assert client.calls == []


def test_question_feedback_no_api_key(monkeypatch):
    monkeypatch.setattr(F, "OPENAI_API_KEY", None)
    out = F.generate_question_feedback({"index": 0}, {}, "답변 있음", client=None)
    assert out["status"] == "no-api-key"


def test_question_feedback_api_error_is_caught():
    class Boom:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._boom))

        def _boom(self, **kwargs):
            raise RuntimeError("503")

    out = F.generate_question_feedback({"index": 0}, {}, "답변", client=Boom())
    assert out["status"] == "error"


def test_question_feedback_system_prompt_is_korean_format():
    # The system prompt must request the three-section Korean format.
    assert "잘한 점" in F._QUESTION_FEEDBACK_SYSTEM
    assert "개선점" in F._QUESTION_FEEDBACK_SYSTEM
    assert "바로 적용" in F._QUESTION_FEEDBACK_SYSTEM
    assert "반드시 한국어로 작성하세요" in F._QUESTION_FEEDBACK_SYSTEM
