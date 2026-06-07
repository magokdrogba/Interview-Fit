"""Live interview session: webcam + microphone capture, per-question files.

Public entry points
-------------------
* :class:`AudioRecorder`: thread-safe mic recorder built on ``sounddevice``;
  exposes ``start()`` / ``stop()`` / ``save_wav(path)``.
* :func:`record_answer`: drives one question's recording loop (OpenCV window
  with HUD + landmark overlay + SPACE/ESC handling). Returns timing metadata.
* :func:`run_session`: opens the camera once, iterates the question list,
  writes ``manifest.json`` and per-question ``q{n}.mp4`` / ``q{n}.wav``.

Key bindings during the live session
------------------------------------
* SPACE — toggles between "ready" and "recording" states. The first press on
  a question starts recording its answer; the second press stops it and
  advances to the next question.
* N (or →) — same as a "stop" press once recording, otherwise skip the
  current question with empty files.
* ESC — abort the whole session immediately.

Design notes
------------
* Audio runs in its own ``sounddevice`` callback thread; the video loop reads
  one frame at a time on the main thread. They share nothing but wall-clock.
* The camera and overlay are constructed once per session, not per question,
  to avoid 1–2 s warm-up between questions.
* All exceptions from cv2/sounddevice during one question are surfaced but do
  not stop the session — we move on and record the failure in the manifest.
"""

from __future__ import annotations

import json
import logging
import textwrap
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    RECORDINGS_DIR,
    VIDEO_FPS,
)
from src.interview.overlay import FaceLandmarkOverlay
from src.interview.text_render import render_korean_texts

logger = logging.getLogger(__name__)

# Window + HUD constants
_WINDOW_NAME: str = "AI Mock Interview"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_HEIGHT: int = 132            # px reserved at the top for question text
_HUD_BG_COLOR = (24, 24, 24)
_HUD_TEXT_COLOR = (240, 240, 240)
_REC_COLOR = (0, 0, 220)         # red dot when recording
_READY_COLOR = (200, 200, 0)     # amber when waiting

# HUD font sizes (Pillow pixel sizes for Korean text)
_STATUS_FONT_PX: int = 18
_QUESTION_FONT_PX: int = 22
_QUESTION_LINE_HEIGHT: int = 28
_QUESTION_MAX_LINES: int = 3


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
class AudioRecorder:
    """Background microphone capture into an in-memory buffer."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    # -- public lifecycle --------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("AudioRecorder is already started")
        self._frames = []
        self._started_at = time.monotonic()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros((0, self.channels), dtype="float32")
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self._stopped_at = time.monotonic()
        with self._lock:
            chunks = list(self._frames)
            self._frames = []
        if not chunks:
            return np.zeros((0, self.channels), dtype="float32")
        return np.concatenate(chunks, axis=0)

    def save_wav(self, path: Path, data: np.ndarray | None = None) -> Path:
        """Write the recorded samples to ``path`` as 16-bit PCM WAV.

        If ``data`` is None and the recorder is still running, this calls
        :meth:`stop` first.
        """
        if data is None:
            if self._stream is not None:
                data = self.stop()
            else:
                data = np.zeros((0, self.channels), dtype="float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), data, self.sample_rate, subtype="PCM_16")
        return path

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    # -- internals ---------------------------------------------------------
    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("AudioRecorder status: %s", status)
        with self._lock:
            self._frames.append(indata.copy())


# ---------------------------------------------------------------------------
# Per-question record (data + loop)
# ---------------------------------------------------------------------------
@dataclass
class AnswerRecord:
    index: int
    question: str
    category: str
    followups: list[str]
    video_path: str = ""
    audio_path: str = ""
    answer_started_at: str = ""
    answer_ended_at: str = ""
    duration_s: float = 0.0
    aborted: bool = False
    skipped: bool = False
    note: str = ""


@dataclass
class SessionResult:
    session_id: str
    session_dir: Path
    company: str
    started_at: str
    completed_at: str
    answers: list[AnswerRecord] = field(default_factory=list)
    aborted: bool = False


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def _wrap_text(text: str, max_chars_per_line: int = 60) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=max_chars_per_line) or [""])
    return lines


def _draw_hud(
    frame: np.ndarray,
    *,
    question_idx: int,
    total: int,
    question_text: str,
    recording: bool,
    elapsed_s: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    canvas = np.zeros((h + _HUD_HEIGHT, w, 3), dtype=frame.dtype)
    canvas[:_HUD_HEIGHT] = _HUD_BG_COLOR
    canvas[_HUD_HEIGHT:] = frame

    # --- numeric / ASCII overlays via cv2 (Hershey fonts are fine here) ---
    # Status indicator dot.
    dot_color = _REC_COLOR if recording else _READY_COLOR
    cv2.circle(canvas, (20, 20), 8, dot_color, -1)
    # Question counter is purely numeric/ASCII, so cv2.putText renders it fine.
    cv2.putText(
        canvas, f"Q{question_idx + 1} / {total}",
        (w - 120, 26), _FONT, 0.55, _HUD_TEXT_COLOR, 1, cv2.LINE_AA,
    )

    # --- Korean text via Pillow (cv2.putText cannot render Hangul) ---
    status = (
        f"녹화 중  {elapsed_s:4.1f}초   [SPACE] 정지 · [N] 건너뛰기 · [ESC] 종료"
        if recording
        else "준비됨   [SPACE] 시작 · [N] 건너뛰기 · [ESC] 종료"
    )

    # Estimate how many Korean glyphs fit per line (each ~font-size px wide).
    max_chars = max(12, (w - 32) // _QUESTION_FONT_PX)
    lines = _wrap_text(question_text, max_chars_per_line=max_chars)
    shown = lines[:_QUESTION_MAX_LINES]
    if len(lines) > _QUESTION_MAX_LINES and shown:
        shown[-1] = shown[-1].rstrip() + " …"

    text_items: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = [
        (status, (38, 8), _STATUS_FONT_PX, _HUD_TEXT_COLOR),
    ]
    y = 42
    for line in shown:
        if line:
            text_items.append((line, (16, y), _QUESTION_FONT_PX, _HUD_TEXT_COLOR))
        y += _QUESTION_LINE_HEIGHT

    render_korean_texts(canvas, text_items)
    return canvas


# ---------------------------------------------------------------------------
# Per-question recording loop
# ---------------------------------------------------------------------------
def record_answer(
    cap: cv2.VideoCapture,
    overlay: FaceLandmarkOverlay,
    *,
    question_idx: int,
    total_questions: int,
    question_text: str,
    out_video: Path,
    out_audio: Path,
    fps: int = VIDEO_FPS,
    audio_device: int | str | None = None,
) -> AnswerRecord:
    """Drive the OpenCV loop for a single question. Blocks until the user
    presses SPACE to stop, N to skip, or ESC to abort.

    Returns an :class:`AnswerRecord`; the caller decides what to do with
    ``aborted`` (raise vs. break) and ``skipped`` (mark in manifest).
    """
    rec = AnswerRecord(
        index=question_idx,
        question=question_text,
        category="",  # filled by caller from the question dict
        followups=[],
        video_path=str(out_video.name),
        audio_path=str(out_audio.name),
    )

    # Read one frame to learn the camera resolution.
    ok, first = cap.read()
    if not ok or first is None:
        rec.note = "camera read failed"
        rec.aborted = True
        return rec

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    canvas_size = (w, h + _HUD_HEIGHT)
    writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), canvas_size)
    if not writer.isOpened():
        rec.note = "VideoWriter could not open"
        rec.aborted = True
        return rec

    audio = AudioRecorder(device=audio_device)

    recording = False
    started_mono: float | None = None
    last_ts_ms = 0
    session_clock_start = time.monotonic()

    frame = first
    try:
        while True:
            ts_ms = int((time.monotonic() - session_clock_start) * 1000)
            last_ts_ms = max(last_ts_ms + 1, ts_ms)

            frame = overlay.draw(frame, last_ts_ms)
            elapsed = (time.monotonic() - started_mono) if recording and started_mono else 0.0
            canvas = _draw_hud(
                frame,
                question_idx=question_idx,
                total=total_questions,
                question_text=question_text,
                recording=recording,
                elapsed_s=elapsed,
            )

            if recording:
                writer.write(canvas)

            cv2.imshow(_WINDOW_NAME, canvas)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF

            if key == 27:  # ESC
                rec.aborted = True
                rec.note = "aborted by user"
                break
            if key == ord(" "):
                if not recording:
                    recording = True
                    started_mono = time.monotonic()
                    rec.answer_started_at = datetime.now(timezone.utc).isoformat()
                    audio.start()
                else:
                    break
            elif key in (ord("n"), ord("N"), 83):  # 'n' or right-arrow on macOS
                if recording:
                    break
                rec.skipped = True
                rec.note = "skipped by user"
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                rec.note = "camera read returned no frame"
                rec.aborted = True
                break
    finally:
        writer.release()
        # Finalize audio regardless of how we exited.
        try:
            samples = audio.stop() if audio.is_running else np.zeros((0, 1), dtype="float32")
            audio.save_wav(out_audio, data=samples)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio save failed for q%d: %s", question_idx, exc)
            rec.note = (rec.note + " | " if rec.note else "") + f"audio save failed: {exc}"

    if recording or rec.answer_started_at:
        rec.answer_ended_at = datetime.now(timezone.utc).isoformat()
        if started_mono is not None:
            rec.duration_s = round(time.monotonic() - started_mono, 3)

    if rec.skipped or rec.aborted:
        # Make file paths empty when no real recording happened so Phase 5
        # knows to skip them.
        if rec.skipped:
            try:
                if out_video.exists() and out_video.stat().st_size < 1_000:
                    out_video.unlink()
                if out_audio.exists() and out_audio.stat().st_size < 1_000:
                    out_audio.unlink()
            except OSError:
                pass
            rec.video_path = ""
            rec.audio_path = ""

    return rec


# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------
def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_session(
    questions: Iterable[dict],
    *,
    company: str = "",
    output_dir: Path | None = None,
    camera_index: int = 0,
    fps: int = VIDEO_FPS,
    audio_device: int | str | None = None,
    use_overlay: bool = True,
) -> SessionResult:
    """Run the whole interview. Returns the populated :class:`SessionResult`.

    The session directory always contains:
      * ``manifest.json``         — schema described by :class:`SessionResult`
      * ``q{n}.mp4`` / ``q{n}.wav`` for each non-skipped question (1-indexed)
    """
    qs = list(questions)
    if not qs:
        raise ValueError("questions must contain at least one item")

    session_id = _new_session_id()
    base = Path(output_dir or RECORDINGS_DIR) / session_id
    base.mkdir(parents=True, exist_ok=True)

    result = SessionResult(
        session_id=session_id,
        session_dir=base,
        company=company,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at="",
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check macOS camera permissions for your terminal app."
        )
    try:
        cap.set(cv2.CAP_PROP_FPS, fps)
        overlay = FaceLandmarkOverlay() if use_overlay else FaceLandmarkOverlay(landmarker=None)
        try:
            for i, q in enumerate(qs):
                rec = record_answer(
                    cap,
                    overlay,
                    question_idx=i,
                    total_questions=len(qs),
                    question_text=q.get("question", ""),
                    out_video=base / f"q{i + 1}.mp4",
                    out_audio=base / f"q{i + 1}.wav",
                    fps=fps,
                    audio_device=audio_device,
                )
                rec.category = q.get("category", "")
                rec.followups = list(q.get("followups", []))
                result.answers.append(rec)
                if rec.aborted:
                    result.aborted = True
                    break
        finally:
            overlay.close()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    result.completed_at = datetime.now(timezone.utc).isoformat()
    _write_manifest(result, qs)
    return result


def _write_manifest(result: SessionResult, original_questions: list[dict]) -> Path:
    path = result.session_dir / "manifest.json"
    payload: dict[str, Any] = {
        "session_id": result.session_id,
        "company": result.company,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "aborted": result.aborted,
        "answers": [
            {
                "index": a.index,
                "category": a.category,
                "question": a.question,
                "followups": a.followups,
                "video_path": a.video_path,
                "audio_path": a.audio_path,
                "answer_started_at": a.answer_started_at,
                "answer_ended_at": a.answer_ended_at,
                "duration_s": a.duration_s,
                "aborted": a.aborted,
                "skipped": a.skipped,
                "note": a.note,
            }
            for a in result.answers
        ],
        # Keep a copy of the original question set so Phase 6 can quote it
        # back even if files got moved.
        "questions": original_questions,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
