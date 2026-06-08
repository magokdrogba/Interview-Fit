"""Tests for ``src.analysis.audio``.

We synthesize WAVs deterministically rather than depending on real recordings,
which keeps the tests hermetic and fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.analysis import audio as A


def _write_wav(path, samples: np.ndarray, sr: int = 16_000) -> None:
    sf.write(str(path), samples.astype("float32"), sr, subtype="PCM_16")


def _tone(seconds: float, freq: float = 220.0, sr: int = 16_000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype="float32") / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _silence(seconds: float, sr: int = 16_000) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype="float32")


def test_graceful_fallback_when_webrtcvad_unavailable(tmp_path, monkeypatch):
    """On platforms without webrtcvad (e.g. Python 3.14 on Streamlit Cloud),
    analyze_audio must not crash: it returns safe-default speech metrics."""
    monkeypatch.setattr(A, "WEBRTCVAD_AVAILABLE", False)
    p = tmp_path / "tone.wav"
    _write_wav(p, _tone(1.5, freq=180.0))

    out = A.analyze_audio(p)
    assert out["status"] == "ok"
    assert out["speech"]["status"] == "unavailable"
    # Safe defaults — never crash downstream consumers.
    assert out["speech"]["pause_count_3s"] == 0
    assert out["speech"]["hesitation_before_speech_s"] == 0.0
    assert out["speech"]["speech_ratio"] is None


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------
def test_no_file_returns_status(tmp_path):
    out = A.analyze_audio(tmp_path / "missing.wav")
    assert out["status"] == "no-file"


def test_empty_file_returns_no_file(tmp_path):
    p = tmp_path / "e.wav"
    p.write_bytes(b"")
    assert A.analyze_audio(p)["status"] == "no-file"


def test_too_short_returns_too_short(tmp_path):
    p = tmp_path / "short.wav"
    _write_wav(p, _tone(0.05))  # 50 ms
    out = A.analyze_audio(p)
    assert out["status"] == "too-short"
    assert out["duration_s"] < 0.1


# ---------------------------------------------------------------------------
# VAD segmentation
# ---------------------------------------------------------------------------
def test_silence_only_yields_zero_speech(tmp_path):
    p = tmp_path / "silent.wav"
    _write_wav(p, _silence(2.0))
    out = A.analyze_audio(p)
    assert out["status"] == "ok"
    assert out["speech"]["status"] == "ok"
    assert out["speech"]["total_speech_s"] == 0.0
    # 2 seconds of silence → leading hesitation captures the whole thing.
    assert out["speech"]["hesitation_before_speech_s"] == pytest.approx(2.0, abs=0.1)


def test_long_silence_counted_as_3s_pause(tmp_path):
    # 0.5s tone, 4s silence, 0.5s tone → 1 pause ≥3s.
    samples = np.concatenate([_tone(0.5), _silence(4.0), _tone(0.5)])
    p = tmp_path / "pause.wav"
    _write_wav(p, samples)
    out = A.analyze_audio(p)
    assert out["status"] == "ok"
    sp = out["speech"]
    assert sp["pause_count_3s"] >= 1
    assert sp["longest_pause_s"] >= 3.0


def test_shape_includes_all_blocks_when_syllables_given(tmp_path):
    samples = np.concatenate([_tone(1.0)])
    p = tmp_path / "tone.wav"
    _write_wav(p, samples)
    out = A.analyze_audio(p, syllable_count=4)
    assert out["status"] == "ok"
    assert "speech" in out
    # With explicit syllable count the rate block should populate.
    if out["speech"]["total_speech_s"] >= 0.5:
        assert "rate" in out


def test_rate_labels_classify_correctly(tmp_path):
    # We don't depend on the real VAD here — just exercise _label_rate directly.
    assert A._label_rate(120.0) == "slow"
    assert A._label_rate(240.0) == "normal"
    assert A._label_rate(400.0) == "fast"
