"""Tests for ``src.interview.session``.

We cannot exercise the camera/microphone in CI, so we test the file-format
contract, the manifest schema, the overlay's no-op path, and the pending-
session JSON round-trip. The OpenCV loop itself is integration-tested by
running the CLI manually.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from src.interview import session
from src.interview.overlay import FaceLandmarkOverlay


# ---------------------------------------------------------------------------
# AudioRecorder
# ---------------------------------------------------------------------------
def _push(recorder: session.AudioRecorder, samples: np.ndarray) -> None:
    """Simulate the sounddevice callback delivering a chunk of audio."""
    recorder._callback(samples, len(samples), None, None)


def test_audio_recorder_save_wav_round_trip(tmp_path):
    rec = session.AudioRecorder(sample_rate=16_000, channels=1)
    # Fake one second of 440 Hz tone, mono, float32 (no stream needed).
    t = np.arange(16_000, dtype="float32") / 16_000
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    _push(rec, tone)
    # Manually drain into save_wav without going through start/stop.
    accumulated = np.concatenate(rec._frames, axis=0)
    out = rec.save_wav(tmp_path / "q1.wav", data=accumulated)

    assert out.exists()
    data, sr = sf.read(str(out))
    assert sr == 16_000
    # 16-bit PCM round-trips to float; length must match.
    assert abs(len(data) - 16_000) <= 1
    # Energy survived the round trip (sanity).
    assert float(np.abs(data).mean()) > 0.05


def test_audio_recorder_save_empty_when_nothing_captured(tmp_path):
    rec = session.AudioRecorder(sample_rate=16_000, channels=1)
    out = rec.save_wav(tmp_path / "empty.wav")
    assert out.exists()
    data, sr = sf.read(str(out))
    assert sr == 16_000
    assert len(data) == 0


def test_audio_recorder_concatenates_multiple_chunks(tmp_path):
    rec = session.AudioRecorder(sample_rate=8_000, channels=1)
    chunk = np.zeros((4_000, 1), dtype="float32")
    _push(rec, chunk)
    _push(rec, chunk + 0.1)
    accumulated = np.concatenate(rec._frames, axis=0)
    out = rec.save_wav(tmp_path / "concat.wav", data=accumulated)
    data, _ = sf.read(str(out))
    assert len(data) == 8_000  # 0.5 + 0.5 sec at 8 kHz


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------
def test_write_manifest_shape(tmp_path):
    result = session.SessionResult(
        session_id="20260530-120000",
        session_dir=tmp_path,
        company="Toss",
        started_at="2026-05-30T12:00:00Z",
        completed_at="2026-05-30T12:05:00Z",
        answers=[
            session.AnswerRecord(
                index=0,
                question="Tell me about yourself.",
                category="personality",
                followups=["What motivates you?"],
                video_path="q1.mp4",
                audio_path="q1.wav",
                answer_started_at="2026-05-30T12:00:10Z",
                answer_ended_at="2026-05-30T12:01:00Z",
                duration_s=50.0,
            ),
        ],
    )
    original_questions = [
        {"category": "personality", "question": "Tell me about yourself.",
         "rationale": "warmup", "followups": ["What motivates you?"]},
    ]
    path = session._write_manifest(result, original_questions)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "20260530-120000"
    assert payload["company"] == "Toss"
    assert len(payload["answers"]) == 1
    a = payload["answers"][0]
    for key in ("index", "category", "question", "video_path", "audio_path",
                "answer_started_at", "answer_ended_at", "duration_s",
                "aborted", "skipped", "note", "followups"):
        assert key in a
    assert payload["questions"] == original_questions
    assert payload["aborted"] is False


# ---------------------------------------------------------------------------
# Overlay graceful no-op
# ---------------------------------------------------------------------------
def test_overlay_no_model_is_noop(monkeypatch):
    # Force the model-fetch path to return None so .draw becomes a no-op
    # without touching the network or loading the heavy MediaPipe task.
    from src.interview import overlay as overlay_mod
    monkeypatch.setattr(overlay_mod, "ensure_face_landmarker_model", lambda *_a, **_k: None)

    overlay = FaceLandmarkOverlay(landmarker=None)
    frame = np.zeros((10, 10, 3), dtype="uint8")
    out = overlay.draw(frame, 0)
    assert out is frame
    assert overlay.active is False


def test_overlay_handles_empty_frame(monkeypatch):
    from src.interview import overlay as overlay_mod
    monkeypatch.setattr(overlay_mod, "ensure_face_landmarker_model", lambda *_a, **_k: None)

    overlay = FaceLandmarkOverlay(landmarker=None)
    out = overlay.draw(np.zeros((0, 0, 3), dtype="uint8"), 0)
    assert out.size == 0


def test_overlay_tolerates_detect_exception():
    """If the landmarker throws on a given frame, .draw must swallow it and
    return the frame unchanged — we never want to crash the live loop."""

    class _Boom:
        def detect_for_video(self, *args, **kwargs):
            raise RuntimeError("boom")

        def close(self):
            pass

    overlay = FaceLandmarkOverlay(landmarker=_Boom())
    frame = (np.random.rand(20, 20, 3) * 255).astype("uint8")
    out = overlay.draw(frame, 0)
    assert out is frame  # returned unchanged


# ---------------------------------------------------------------------------
# Pending-session JSON round-trip
# ---------------------------------------------------------------------------
def test_pending_session_round_trip(tmp_path):
    """Mirror the schema written by app.py and read by run_interview.py."""
    payload = {
        "company": "Toss",
        "questions": [
            {"category": "personality", "question": "Q1",
             "rationale": "r", "followups": ["f1", "f2"]},
            {"category": "domain", "question": "Q2",
             "rationale": "r", "followups": ["f1", "f2"]},
        ],
    }
    f = tmp_path / "_pending_session.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(f.read_text(encoding="utf-8"))
    assert loaded["company"] == "Toss"
    assert len(loaded["questions"]) == 2
    assert loaded["questions"][0]["category"] == "personality"


# ---------------------------------------------------------------------------
# run_session input validation
# ---------------------------------------------------------------------------
def test_run_session_rejects_empty_questions(tmp_path):
    with pytest.raises(ValueError):
        session.run_session([], output_dir=tmp_path)
