"""Tests for ``src.analysis.orchestrator``.

We patch the three per-axis analyzers so the orchestrator can be exercised
end-to-end without any real audio/video work.
"""

from __future__ import annotations

import json

import pytest

from src.analysis import orchestrator as O


def _fake_audio_ok(audio_path=None, *, syllable_count=None, **kw):
    base = {
        "status": "ok",
        "duration_s": 12.0,
        "speech": {"status": "ok", "total_speech_s": 9.0, "total_silence_s": 3.0,
                   "speech_ratio": 0.75, "pause_count_3s": 1,
                   "longest_pause_s": 3.5, "hesitation_before_speech_s": 0.8},
        "voice": {"status": "ok", "f0_mean_hz": 170.0, "f0_cv": 0.08,
                  "jitter": 0.01, "energy_cv": 0.3, "tremor_index": 0.2},
    }
    if syllable_count is not None:
        base["rate"] = {"syllable_count": syllable_count,
                        "syllables_per_minute": 240.0, "rate_label": "normal"}
    return base


def _fake_language_ok(audio_path=None, *, transcript=None, question_text="",
                      client=None, **kw):
    return {
        "status": "ok",
        "transcript": {"text": "Hello", "language": "en", "char_count": 5,
                       "word_count": 1, "syllable_count": 36},
        "fillers": {"status": "ok", "counts": {"um": 2}, "total": 2, "per_minute": 10.0},
        "repetition": {"status": "ok", "top": [], "type_token_ratio": 0.9,
                       "total_words": 30, "content_words": 18},
        "structure": {"status": "ok", "score": 4, "label": "top-down",
                      "conclusion_first": True, "mece": 4, "comment": "good"},
    }


def _fake_vision_ok(video_path=None, *, sample_fps=5, landmarker=None, **kw):
    return {
        "status": "ok", "duration_s": 12.0,
        "frames_sampled": 60, "frames_with_face": 58, "face_coverage": 0.97,
        "gaze": {"status": "ok", "looking_ratio": 0.8, "gaze_away_events": 3},
        "expression": {"status": "ok", "positive_ratio": 0.4,
                       "neutral_ratio": 0.5, "tense_ratio": 0.1},
        "head": {"status": "ok", "yaw_std_deg": 5.0, "pitch_std_deg": 3.0,
                 "roll_std_deg": 2.0, "direction_changes_per_min": 10.0},
    }


@pytest.fixture
def session_dir(tmp_path):
    base = tmp_path / "20260530-160000"
    base.mkdir()
    # Two real (small) media files; they don't need to be valid because the
    # analyzers are patched.
    (base / "q1.mp4").write_bytes(b"\x00" * 16)
    (base / "q1.wav").write_bytes(b"\x00" * 16)
    (base / "q2.mp4").write_bytes(b"\x00" * 16)
    (base / "q2.wav").write_bytes(b"\x00" * 16)

    manifest = {
        "session_id": "20260530-160000",
        "company": "Toss",
        "aborted": False,
        "answers": [
            {"index": 0, "category": "personality",
             "question": "Tell me about yourself.",
             "followups": [],
             "video_path": "q1.mp4", "audio_path": "q1.wav",
             "duration_s": 12.0, "skipped": False, "aborted": False,
             "note": ""},
            {"index": 1, "category": "experience",
             "question": "Walk me through a project.",
             "followups": [],
             "video_path": "q2.mp4", "audio_path": "q2.wav",
             "duration_s": 18.0, "skipped": False, "aborted": False,
             "note": ""},
        ],
        "questions": [],
    }
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base


def test_orchestrator_happy_path(session_dir, monkeypatch):
    monkeypatch.setattr(O, "analyze_audio", _fake_audio_ok)
    monkeypatch.setattr(O, "analyze_language", _fake_language_ok)
    monkeypatch.setattr(O, "analyze_video", _fake_vision_ok)

    out = O.analyze_session(session_dir)
    assert out["status"] == "ok"
    assert out["company"] == "Toss"
    assert len(out["per_question"]) == 2
    pq = out["per_question"][0]
    for key in ("vision", "audio", "language"):
        assert pq[key]["status"] == "ok"
    # Aggregate populates.
    a = out["aggregate"]
    assert a["status"] == "ok"
    assert a["answered"] == 2
    assert a["vision"]["looking_ratio_mean"] == pytest.approx(0.8)
    assert a["audio"]["pause_count_3s_total"] == 2
    assert a["language"]["structure_score_mean"] == pytest.approx(4.0)

    # analysis.json should be written next to manifest.
    assert (session_dir / "analysis.json").exists()


def test_orchestrator_caches_result(session_dir, monkeypatch):
    monkeypatch.setattr(O, "analyze_audio", _fake_audio_ok)
    monkeypatch.setattr(O, "analyze_language", _fake_language_ok)
    monkeypatch.setattr(O, "analyze_video", _fake_vision_ok)
    first = O.analyze_session(session_dir)
    second = O.analyze_session(session_dir)
    assert second.get("_from_cache") is True
    # Cache content matches the first call's per_question payload.
    assert second["per_question"] == first["per_question"]


def test_orchestrator_force_refresh_recomputes(session_dir, monkeypatch):
    monkeypatch.setattr(O, "analyze_audio", _fake_audio_ok)
    monkeypatch.setattr(O, "analyze_language", _fake_language_ok)
    monkeypatch.setattr(O, "analyze_video", _fake_vision_ok)
    O.analyze_session(session_dir)
    out = O.analyze_session(session_dir, force_refresh=True)
    assert "_from_cache" not in out


def test_orchestrator_isolates_per_question_failures(session_dir, monkeypatch):
    calls = {"n": 0}

    def flaky_audio(audio_path=None, *, syllable_count=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("kaboom")
        return _fake_audio_ok(audio_path, syllable_count=syllable_count, **kw)

    monkeypatch.setattr(O, "analyze_audio", flaky_audio)
    monkeypatch.setattr(O, "analyze_language", _fake_language_ok)
    monkeypatch.setattr(O, "analyze_video", _fake_vision_ok)

    out = O.analyze_session(session_dir)
    # First question's audio failed, but the rest of that question (language,
    # vision) plus the second question all ran.
    assert out["per_question"][0]["audio"]["status"] == "error"
    assert out["per_question"][0]["language"]["status"] == "ok"
    assert out["per_question"][1]["audio"]["status"] == "ok"
    assert any("audio analysis error" in n for n in out["notes"])


def test_orchestrator_handles_skipped_question(session_dir, monkeypatch):
    manifest = json.loads((session_dir / "manifest.json").read_text())
    manifest["answers"][0]["skipped"] = True
    manifest["answers"][0]["video_path"] = ""
    manifest["answers"][0]["audio_path"] = ""
    (session_dir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(O, "analyze_audio", _fake_audio_ok)
    monkeypatch.setattr(O, "analyze_language", _fake_language_ok)
    monkeypatch.setattr(O, "analyze_video", _fake_vision_ok)
    out = O.analyze_session(session_dir, force_refresh=True)
    assert out["per_question"][0]["status"] == "skipped"
    # Aggregate only counts the non-skipped one.
    assert out["aggregate"]["answered"] == 1


def test_orchestrator_no_manifest(tmp_path):
    out = O.analyze_session(tmp_path)
    assert out["status"] == "no-manifest"


def test_aggregate_empty_returns_no_data():
    assert O._aggregate([])["status"] == "no-data"
