"""Post-hoc audio analysis: speech-rate context, silence/pauses, voice tremor.

Public API
----------
* :func:`analyze_audio` — takes a path to a per-question WAV (Phase 4 output)
  and returns a dict with three sub-blocks:
    - ``speech``  (webrtcvad-driven): total speech vs silence, leading
      hesitation, count of long pauses, longest pause.
    - ``rate``    (optional): syllables per minute, classification, only if
      the caller passes a ``syllable_count`` (typically derived from the STT
      transcript in :mod:`src.analysis.language`).
    - ``voice``   (librosa.pyin + RMS): mean F0, F0 coefficient of variation,
      a cycle-to-cycle jitter proxy, energy CV, and a single 0-1
      "tremor_index" that fuses the three.

Every public dict always contains a ``status`` key. On any failure we return
``{"status": "error"|"no-file"|"empty"|"too-short", ...}`` rather than raising
so the orchestrator can keep going.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from config import AUDIO_SAMPLE_RATE, LONG_PAUSE_SECONDS, VAD_AGGRESSIVENESS

logger = logging.getLogger(__name__)

# webrtcvad has no wheels for some newer Python versions (e.g. Python 3.14 on
# Streamlit Cloud) and needs a C compiler to build from source. Import it
# defensively so a missing/incompatible build degrades gracefully instead of
# crashing the whole app at import time. See requirements.txt (webrtcvad-wheels).
try:
    import webrtcvad

    WEBRTCVAD_AVAILABLE = True
except Exception:  # noqa: BLE001 - ImportError or any load/ABI failure
    webrtcvad = None  # type: ignore[assignment]
    WEBRTCVAD_AVAILABLE = False
    logger.warning(
        "webrtcvad is unavailable; speech/silence segmentation will be skipped "
        "and the related metrics will report as unmeasured."
    )

# webrtcvad supports 8/16/32/48 kHz only. We resample everything to this.
_VAD_SAMPLE_RATES: tuple[int, ...] = (8_000, 16_000, 32_000, 48_000)
_VAD_FRAME_MS: int = 30  # 30 ms = 480 samples @ 16 kHz = 960 bytes int16 mono
_MIN_DURATION_S: float = 0.25  # below this we refuse to analyze

# Raw cycle-to-cycle jitter for interview-quality mics typically sits in
# 0.0–0.05. We clamp to this and map to a 0–1 tremor display score (Issue 2).
MAX_JITTER: float = 0.05

# Korean syllable blocks (Hangul). One block == one spoken syllable, which is
# the correct unit for speech-rate (SPM). Counting ``len(transcript)`` instead
# inflates the value badly because it includes spaces, punctuation, and Latin
# characters — that is the bug this module exists to avoid.
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_LATIN_WORD_RE = re.compile(r"[A-Za-z']+")


def count_korean_syllables(text: str) -> int:
    """Count spoken syllables in ``text``.

    Korean answers: number of Hangul syllable blocks (가-힣). This is the
    fix for the inflated SPM bug — e.g. a transcript that ``len()`` reports as
    1247 chars resolves to a realistic Hangul-block count here.

    Non-Korean answers (no Hangul present): fall back to a rough English
    vowel-group estimate so SPM is still meaningful instead of zero.
    """
    if not text:
        return 0
    hangul = len(_HANGUL_RE.findall(text))
    if hangul > 0:
        return hangul
    return _latin_syllable_estimate(text)


def _latin_syllable_estimate(text: str) -> int:
    total = 0
    for word in _LATIN_WORD_RE.findall(text):
        groups = _LATIN_VOWEL_GROUP_RE.findall(word)
        n = len(groups)
        if word.lower().endswith("e") and n > 1:
            n -= 1
        total += max(1, n)
    return total


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def analyze_audio(
    audio_path: Path | str,
    *,
    syllable_count: int | None = None,
    transcript: str | None = None,
    vad_aggressiveness: int = VAD_AGGRESSIVENESS,
) -> dict[str, Any]:
    """Compute the audio-axis metrics for one answer clip.

    Parameters
    ----------
    audio_path:
        WAV file written by Phase 4.
    transcript:
        Optional STT transcript. When given (and ``syllable_count`` is not),
        the speech-rate sub-block is computed from the number of Korean
        syllable blocks via :func:`count_korean_syllables` — the correct,
        non-inflated SPM unit.
    syllable_count:
        Optional explicit syllable count. Takes precedence over ``transcript``
        if both are supplied. Mainly kept for tests / callers that already
        have a count.
    vad_aggressiveness:
        webrtcvad mode 0–3. Higher = more aggressive (more silence).
    """
    # Derive syllable count from the transcript when not given explicitly.
    if syllable_count is None and transcript:
        syllable_count = count_korean_syllables(transcript)
    path = Path(audio_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "no-file", "path": str(path)}

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "path": str(path), "error": str(exc)}

    if y.ndim > 1:  # collapse to mono if multi-channel
        y = y.mean(axis=1)
    if y.size == 0:
        return {"status": "empty", "path": str(path)}

    duration_s = float(len(y) / sr)
    if duration_s < _MIN_DURATION_S:
        return {"status": "too-short", "duration_s": duration_s}

    out: dict[str, Any] = {
        "status": "ok",
        "path": str(path),
        "duration_s": round(duration_s, 3),
        "sample_rate": int(sr),
    }
    notes: list[str] = []

    # --- Speech segmentation -------------------------------------------------
    if not WEBRTCVAD_AVAILABLE:
        out["speech"] = _unavailable_speech()
        notes.append("webrtcvad unavailable; speech segmentation skipped")
    else:
        try:
            seg = _vad_segments(y, sr, aggressiveness=vad_aggressiveness)
            out["speech"] = _summarize_segments(seg, duration_s)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vad failed: {exc}")
            out["speech"] = {"status": "error"}

    # --- Speech rate ---------------------------------------------------------
    if syllable_count is not None and out["speech"].get("status") == "ok":
        speech_s = out["speech"]["total_speech_s"]
        if speech_s >= 0.5 and syllable_count > 0:
            spm = (syllable_count / speech_s) * 60.0
            out["rate"] = {
                "syllable_count": int(syllable_count),
                "syllables_per_minute": round(spm, 1),
                "rate_label": _label_rate(spm),
            }
        else:
            out["rate"] = {"status": "no-speech"}

    # --- Voice tremor / stability -------------------------------------------
    try:
        out["voice"] = _voice_metrics(y, sr)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"voice metrics failed: {exc}")
        out["voice"] = {"status": "error"}

    if notes:
        out["notes"] = notes
    return out


# ---------------------------------------------------------------------------
# VAD-driven speech/silence segmentation
# ---------------------------------------------------------------------------
@dataclass
class _Segment:
    start_s: float
    end_s: float
    is_speech: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _resample_for_vad(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Return mono int16 samples at the nearest supported VAD sample rate."""
    if sr not in _VAD_SAMPLE_RATES:
        target = min(_VAD_SAMPLE_RATES, key=lambda r: abs(r - sr))
        y = librosa.resample(y, orig_sr=sr, target_sr=target)
        sr = target
    # float32 [-1,1] → int16
    clipped = np.clip(y, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16), sr


def _vad_segments(y: np.ndarray, sr: int, *, aggressiveness: int) -> list[_Segment]:
    """Run webrtcvad over 30 ms frames and merge into speech/silence segments."""
    samples_i16, vad_sr = _resample_for_vad(y, sr)
    vad = webrtcvad.Vad(int(aggressiveness))
    frame_len = int(vad_sr * _VAD_FRAME_MS / 1000)  # samples per frame
    frame_bytes = frame_len * 2  # int16 → 2 bytes/sample

    raw = samples_i16.tobytes()
    segments: list[_Segment] = []
    cur_is_speech: bool | None = None
    cur_start = 0.0
    t = 0.0
    step_s = _VAD_FRAME_MS / 1000.0

    for off in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        chunk = raw[off : off + frame_bytes]
        try:
            is_sp = vad.is_speech(chunk, vad_sr)
        except Exception:  # noqa: BLE001 - malformed frame → treat as silence
            is_sp = False

        if cur_is_speech is None:
            cur_is_speech = is_sp
            cur_start = t
        elif is_sp != cur_is_speech:
            segments.append(_Segment(cur_start, t, cur_is_speech))
            cur_is_speech = is_sp
            cur_start = t
        t += step_s

    if cur_is_speech is not None and t > cur_start:
        segments.append(_Segment(cur_start, t, cur_is_speech))
    return segments


def _unavailable_speech() -> dict[str, Any]:
    """Safe-default speech block when webrtcvad isn't installed.

    Counts default to 0 / 0.0 so downstream code never crashes. ``speech_ratio``
    is left None (rather than 0.0) so the UI shows "측정 불가" instead of a
    misleading "답변량 부족" when we genuinely could not measure it.
    """
    return {
        "status": "unavailable",
        "total_speech_s": 0.0,
        "total_silence_s": 0.0,
        "speech_ratio": None,
        "pause_count_3s": 0,
        "longest_pause_s": 0.0,
        "hesitation_before_speech_s": 0.0,
    }


def _summarize_segments(segments: list[_Segment], duration_s: float) -> dict[str, Any]:
    speech_s = sum(s.duration_s for s in segments if s.is_speech)
    silence_s = sum(s.duration_s for s in segments if not s.is_speech)
    long_pauses = [s.duration_s for s in segments
                   if not s.is_speech and s.duration_s >= LONG_PAUSE_SECONDS]
    leading = 0.0
    if segments and not segments[0].is_speech:
        leading = segments[0].duration_s

    speech_ratio = speech_s / duration_s if duration_s > 0 else 0.0
    return {
        "status": "ok",
        "total_speech_s": round(speech_s, 3),
        "total_silence_s": round(silence_s, 3),
        "speech_ratio": round(speech_ratio, 4),
        "pause_count_3s": len(long_pauses),
        "longest_pause_s": round(max((s.duration_s for s in segments if not s.is_speech), default=0.0), 3),
        "hesitation_before_speech_s": round(leading, 3),
    }


def _label_rate(spm: float) -> str:
    # Standard conversational Korean/English ranges (approx; broad bins).
    # Sources: 200–250 syll/min is "normal" conversational; ~300+ is fast.
    if spm < 180:
        return "slow"
    if spm > 320:
        return "fast"
    return "normal"


# ---------------------------------------------------------------------------
# F0 / jitter / energy → tremor index
# ---------------------------------------------------------------------------
def _voice_metrics(y: np.ndarray, sr: int) -> dict[str, Any]:
    # librosa.pyin returns NaN where unvoiced. Use a generous F0 range that
    # covers low male to high female voices.
    fmin = librosa.note_to_hz("C2")  # ~65 Hz
    fmax = librosa.note_to_hz("C6")  # ~1046 Hz
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048,
    )
    voiced = f0[~np.isnan(f0)]

    if voiced.size < 20:
        # Too little voiced material to say anything stable.
        return {"status": "insufficient-voiced"}

    f0_mean = float(np.mean(voiced))
    f0_std = float(np.std(voiced))
    f0_cv = f0_std / f0_mean if f0_mean > 0 else 0.0

    # Cycle-to-cycle jitter proxy: relative difference between adjacent voiced F0s.
    periods = 1.0 / voiced
    if periods.size > 1:
        diffs = np.abs(np.diff(periods))
        jitter = float(diffs.mean() / periods.mean())
    else:
        jitter = 0.0

    # Energy stability: RMS over short frames, coefficient of variation.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if rms.size > 1 and rms.mean() > 1e-6:
        energy_cv = float(rms.std() / rms.mean())
    else:
        energy_cv = 0.0

    # Tremor score = normalized jitter (Issue 2). Raw jitter from pyin has a
    # wide range; clamp it to a realistic interview-mic band [0, MAX_JITTER]
    # and map to a 0–1 display score. This is far less over-sensitive than the
    # old blended index.
    raw_jitter = jitter
    normalized = float(min(raw_jitter / MAX_JITTER, 1.0)) if MAX_JITTER > 0 else 0.0
    tremor_index = round(normalized, 3)

    logger.info(
        "voice tremor: raw_jitter=%.5f normalized=%.3f (MAX_JITTER=%.3f)",
        raw_jitter, normalized, MAX_JITTER,
    )

    return {
        "status": "ok",
        "f0_mean_hz": round(f0_mean, 2),
        "f0_cv": round(f0_cv, 4),
        "jitter": round(raw_jitter, 5),
        "raw_jitter": round(raw_jitter, 5),
        "energy_cv": round(energy_cv, 4),
        "tremor_index": tremor_index,
    }
