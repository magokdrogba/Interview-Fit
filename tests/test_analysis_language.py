"""Tests for ``src.analysis.language``.

We never call the real Whisper or GPT APIs. Both are reached through a
``client`` object whose surface we mimic with simple namespaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from src.analysis import language as L


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _Resp:
    """Mimic the openai SDK's verbose_json transcription response."""

    def __init__(self, text="", language="en", duration=10.0, segments=None):
        self._text = text
        self._language = language
        self._duration = duration
        self._segments = segments or []
        self.text = text  # SDK exposes both attributes and model_dump()

    def model_dump(self):
        return {
            "text": self._text,
            "language": self._language,
            "duration": self._duration,
            "segments": self._segments,
        }


class _FakeChatResp:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeClient:
    """One fake client covers both STT and chat."""

    def __init__(self, stt_reply: _Resp | Exception | None = None,
                 chat_replies: list[str] | None = None):
        self._stt_reply = stt_reply
        self._chat_replies = list(chat_replies or [])
        self.stt_calls = 0
        self.chat_calls: list[dict] = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._stt))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _stt(self, *, model, file, response_format, language):
        self.stt_calls += 1
        if isinstance(self._stt_reply, Exception):
            raise self._stt_reply
        return self._stt_reply

    def _chat(self, *, model, messages, response_format, temperature):
        self.chat_calls.append({"model": model, "messages": messages})
        if not self._chat_replies:
            raise AssertionError("FakeClient ran out of chat replies")
        return _FakeChatResp(self._chat_replies.pop(0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def wav_path(tmp_path) -> Path:
    p = tmp_path / "answer.wav"
    sr = 16_000
    t = np.arange(sr, dtype="float32") / sr
    samples = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
    sf.write(str(p), samples, sr, subtype="PCM_16")
    return p


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path / "_cache")
    (tmp_path / "_cache").mkdir(exist_ok=True)
    yield


# ---------------------------------------------------------------------------
# Filler + repetition primitives
# ---------------------------------------------------------------------------
def test_filler_metrics_korean():
    text = "음, 그러니까 약간 그냥 이렇게 되게 좋아요."
    m = L._filler_metrics(text, transcript_duration_s=5.0)
    assert m["status"] == "ok"
    assert m["counts"].get("음", 0) >= 1
    assert m["counts"].get("그러니까", 0) >= 1
    assert m["counts"].get("그냥", 0) >= 1
    assert m["total"] >= 4


def test_filler_does_not_count_content_words():
    # 시장/분석/전략 are content words and must never be counted as fillers,
    # even though "그" appears *inside* "그래서" (which is itself a filler).
    text = "어 그래서 저는 그냥 시장 분석을 했고 전략이 중요해요 어"
    m = L._filler_metrics(text, transcript_duration_s=60.0)
    assert "시장" not in m["counts"]
    assert "분석" not in m["counts"]
    assert "전략" not in m["counts"]
    assert m["counts"].get("어") == 2          # standalone tokens only
    assert m["counts"].get("그래서") == 1      # exact token, not "그"
    assert "그" not in m["counts"]             # not double-counted from 그래서


def test_filler_multiword_phrase_detected():
    text = "어떻게 보면 저는 약간 긴장했어요"
    m = L._filler_metrics(text, transcript_duration_s=10.0)
    assert m["counts"].get("어떻게 보면") == 1
    assert m["counts"].get("약간") == 1


def test_filler_counts_sorted_descending():
    text = "어 어 어 음 음 그냥"
    m = L._filler_metrics(text, transcript_duration_s=10.0)
    counts = list(m["counts"].values())
    assert counts == sorted(counts, reverse=True)
    assert next(iter(m["counts"])) == "어"  # most frequent first


def test_filler_none_detected_returns_empty_counts():
    text = "저는 데이터 분석가로서 시장 조사를 수행했습니다"
    m = L._filler_metrics(text, transcript_duration_s=10.0)
    assert m["total"] == 0
    assert m["counts"] == {}


def test_repetition_finds_high_frequency_content_words():
    text = ("payments payments payments are critical. payments at scale must "
            "be reliable and the payments backbone needs SLOs.")
    m = L._repetition_metrics(text)
    assert m["status"] == "ok"
    assert m["top"], m
    top_words = {entry["word"] for entry in m["top"]}
    assert "payments" in top_words
    assert m["type_token_ratio"] < 1.0


def test_repetition_ignores_stopwords():
    text = "the the the the the the the and and and and"
    m = L._repetition_metrics(text)
    assert m["top"] == []  # all stopwords; nothing surfaces


def test_word_and_syllable_counts():
    assert L._word_count("Hello world!") == 2
    # Korean syllables: each Hangul block counts as 1.
    assert L._syllable_count("안녕하세요", "ko") == 5
    # English syllables: rough heuristic.
    assert L._syllable_count("hello world", "en") >= 2


# ---------------------------------------------------------------------------
# transcribe() — caching + error paths
# ---------------------------------------------------------------------------
def test_transcribe_uses_cache_on_second_call(wav_path):
    client = FakeClient(stt_reply=_Resp(text="hi there", duration=3.0))
    a = L.transcribe(wav_path, client=client)
    b = L.transcribe(wav_path, client=client)
    assert a == b
    assert client.stt_calls == 1  # second call hit the cache


def test_transcribe_force_refresh(wav_path):
    client = FakeClient(stt_reply=_Resp(text="v1"))
    L.transcribe(wav_path, client=client)
    client._stt_reply = _Resp(text="v2")
    second = L.transcribe(wav_path, client=client, force_refresh=True)
    assert second["text"] == "v2"
    assert client.stt_calls == 2


def test_transcribe_missing_file_returns_no_file(tmp_path):
    out = L.transcribe(tmp_path / "nope.wav", client=FakeClient(stt_reply=_Resp()))
    assert out["status"] == "no-file"


def test_transcribe_api_error_returns_error(wav_path):
    client = FakeClient(stt_reply=RuntimeError("upstream down"))
    out = L.transcribe(wav_path, client=client)
    assert out["status"] == "error"
    assert "upstream down" in out["error"]


def test_transcribe_no_key_raises_with_helpful_message(monkeypatch, wav_path):
    monkeypatch.setattr(L, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        L.transcribe(wav_path, client=None)


# ---------------------------------------------------------------------------
# analyze_language end-to-end
# ---------------------------------------------------------------------------
def test_analyze_language_happy_path(wav_path):
    transcript_text = "음, 저는 결제 마이그레이션을 이끌었습니다. 그냥 결과는 안정적이었습니다."
    client = FakeClient(
        stt_reply=_Resp(text=transcript_text, duration=12.0, language="ko"),
        chat_replies=[json.dumps({
            "score": 4, "reason": "두괄식으로 시작했고 구체적 사례가 있습니다.",
        })],
    )
    out = L.analyze_language(
        wav_path, question_text="프로젝트 경험을 말씀해 주세요.", client=client,
    )
    assert out["status"] == "ok"
    assert out["transcript"]["word_count"] > 0
    assert out["fillers"]["total"] >= 1  # '음', '그냥' detected
    assert out["repetition"]["status"] == "ok"
    assert out["structure"] == {
        "status": "ok",
        "score": 4,
        "reason": "두괄식으로 시작했고 구체적 사례가 있습니다.",
        "comment": "두괄식으로 시작했고 구체적 사례가 있습니다.",
    }
    # One STT, one chat call.
    assert client.stt_calls == 1
    assert len(client.chat_calls) == 1
    # The chat prompt included the question text + the rubric.
    prompt = client.chat_calls[0]["messages"][1]["content"]
    assert "프로젝트 경험을 말씀해 주세요." in prompt
    assert "루브릭" in prompt


def test_analyze_language_with_cached_transcript_skips_stt(wav_path):
    client = FakeClient(
        stt_reply=RuntimeError("must not be called"),
        chat_replies=[json.dumps({"score": 3, "reason": "구조는 있으나 결론이 늦습니다."})],
    )
    transcript = {"status": "ok", "text": "Some answer here.", "language": "en",
                  "duration_s": 5.0, "segments": []}
    out = L.analyze_language(wav_path, transcript=transcript, client=client)
    assert out["status"] == "ok"
    assert client.stt_calls == 0


def test_analyze_language_skip_structure_skips_gpt(wav_path):
    client = FakeClient(stt_reply=_Resp(text="hello world", duration=2.0))
    out = L.analyze_language(wav_path, client=client, skip_structure=True)
    assert out["status"] == "ok"
    assert "structure" not in out
    assert client.chat_calls == []


def test_analyze_language_propagates_stt_error(wav_path):
    client = FakeClient(stt_reply=RuntimeError("503"))
    out = L.analyze_language(wav_path, client=client)
    assert out["status"] == "error"
    assert out["transcript"]["status"] == "error"


def test_structure_handles_invalid_json(wav_path):
    client = FakeClient(
        stt_reply=_Resp(text="hello", duration=2.0),
        chat_replies=["this is not json"],
    )
    out = L.analyze_language(wav_path, client=client)
    assert out["structure"]["status"] == "error"


def test_structure_rejects_out_of_range_score(wav_path):
    client = FakeClient(
        stt_reply=_Resp(text="hello", duration=2.0),
        chat_replies=[json.dumps({"score": 9, "reason": "x"})],
    )
    out = L.analyze_language(wav_path, client=client)
    assert out["structure"]["status"] == "invalid-response"


def test_structure_rubric_score_and_reason(wav_path):
    client = FakeClient(
        stt_reply=_Resp(text="결론부터 말씀드리면…", duration=8.0, language="ko"),
        chat_replies=[json.dumps({"score": 5, "reason": "결론 우선, 첫째/둘째 구조, 사례 포함."})],
    )
    out = L.analyze_language(wav_path, question_text="갈등 해결 경험은?", client=client)
    assert out["structure"]["status"] == "ok"
    assert out["structure"]["score"] == 5
    assert "구조" in out["structure"]["reason"]
    # Rubric items must be present in the prompt.
    prompt = client.chat_calls[0]["messages"][1]["content"]
    assert "두괄식" in prompt
    assert "첫째/둘째/셋째" in prompt
