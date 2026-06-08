"""Post-hoc video analysis: gaze, expression, head movement (MediaPipe Face Landmarker v2).

We sub-sample the answer video at ``ANALYSIS_SAMPLE_FPS`` to keep this fast.
For every sampled frame we extract:

* **landmarks** (478 points incl. iris)  — used for gaze
* **blendshapes** (52 ARKit-style scores) — used for expression
* **facial_transformation_matrix** (4×4) — used for head pose

We aggregate the per-frame numbers into three sub-blocks:

* ``gaze.looking_ratio`` and ``gaze.gaze_away_events`` — share of frames where
  both irises sit within a "looking at the camera" window, and the count of
  saccade-like transitions out of that window.
* ``expression`` — share of frames classified as positive / neutral / tense
  using simple thresholds over blendshape categories.
* ``head`` — std of yaw/pitch/roll (degrees) plus how often the head
  changes direction per minute (a proxy for fidgeting).

Like the audio module, every public dict has a ``status`` field and we never
raise out of the boundary function.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from config import ANALYSIS_SAMPLE_FPS, FACE_LANDMARKER_PATH
from src.interview.overlay import ensure_face_landmarker_model

logger = logging.getLogger(__name__)

# opencv-python needs libGL (a display) which Streamlit Cloud lacks. We install
# opencv-python-headless there (same `cv2` module, no display dep), but still
# import defensively so a missing build degrades gracefully instead of crashing
# the whole app at import time.
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False
    logger.warning(
        "cv2 (OpenCV) is unavailable; video analysis will be skipped and the "
        "video-axis metrics will report as unmeasured."
    )

# mediapipe officially supports Python 3.9–3.12 only and internally imports cv2;
# on unsupported runtimes (e.g. Python 3.14) the import fails. We pin Python via
# runtime.txt for Streamlit Cloud, but still import defensively so a failure
# degrades to safe defaults instead of crashing the app at import time.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = True
except Exception:  # noqa: BLE001 - ImportError or any transitive load failure
    mp = None  # type: ignore[assignment]
    mp_python = None  # type: ignore[assignment]
    mp_vision = None  # type: ignore[assignment]
    MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "mediapipe is unavailable; video analysis will return safe defaults. "
        "Pin Python to 3.11 (see runtime.txt) for full video metrics."
    )

# ---------------------------------------------------------------------------
# Landmark indices (Face Mesh / Face Landmarker v2 share the same scheme)
# ---------------------------------------------------------------------------
# Iris landmarks (only present when ``output_face_blendshapes=True`` *or* the
# default landmark schema, which includes refined irises). Centers are 468/473.
_LEFT_IRIS = 468
_RIGHT_IRIS = 473
# Eye outer/inner corners (looking at user). Used to normalize iris position.
_LEFT_EYE_OUTER, _LEFT_EYE_INNER = 33, 133
_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER = 263, 362

# Threshold (in normalized eye-width units) for "looking roughly at camera".
# Eye width is the distance between outer and inner corner; iris offset from
# the eye-corner midpoint is divided by that. ±0.18 ≈ a relaxed window.
_GAZE_CENTER_THRESHOLD: float = 0.18

# Blendshape categories that map to our coarse buckets.
_POSITIVE_SHAPES: tuple[str, ...] = (
    "mouthSmileLeft", "mouthSmileRight",
    "cheekSquintLeft", "cheekSquintRight",
)
_TENSE_SHAPES: tuple[str, ...] = (
    "browDownLeft", "browDownRight",
    "mouthPressLeft", "mouthPressRight",
    "jawForward",
    "mouthFrownLeft", "mouthFrownRight",
)
# Activation threshold per shape (blendshapes are 0..1).
_BLENDSHAPE_ACTIVE: float = 0.30

# Head-pose direction-change detection
_HEAD_DELTA_DEG: float = 7.0  # angular jump per sample window to count as a change


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _unavailable_video(path: str) -> dict[str, Any]:
    """Safe-default video metrics when OpenCV (cv2) isn't installed.

    Never crashes downstream: sub-blocks report ``no-data`` so the UI shows
    "측정 불가" rather than a fabricated score.
    """
    return {
        "status": "no-cv2",
        "path": path,
        "duration_s": 0.0,
        "frames_sampled": 0,
        "frames_with_face": 0,
        "face_coverage": 0.0,
        "gaze": {"status": "no-data"},
        "expression": {"status": "no-data"},
        "head": {"status": "no-data"},
    }


def _mediapipe_unavailable_video(path: str) -> dict[str, Any]:
    """Safe-default video metrics when mediapipe isn't installed/importable.

    Unlike the cv2 case, we return concrete neutral defaults (attentive gaze,
    calm/neutral expression, no head movement) so the report still renders a
    benign video axis: gaze_ratio=1.0, head_movement=0.0,
    expression positive/neutral/tense = 0/100/0.
    """
    return {
        "status": "no-mediapipe",
        "path": path,
        "duration_s": 0.0,
        "frames_sampled": 0,
        "frames_with_face": 0,
        "face_coverage": 0.0,
        "gaze": {"status": "ok", "looking_ratio": 1.0, "gaze_away_events": 0},
        "expression": {"status": "ok", "positive_ratio": 0.0,
                       "neutral_ratio": 1.0, "tense_ratio": 0.0},
        "head": {"status": "ok", "yaw_std_deg": 0.0, "pitch_std_deg": 0.0,
                 "roll_std_deg": 0.0, "direction_changes_per_min": 0.0},
    }


def analyze_video(
    video_path: Path | str,
    *,
    sample_fps: int = ANALYSIS_SAMPLE_FPS,
    landmarker: Any | None = None,
) -> dict[str, Any]:
    """Compute the video-axis metrics for one answer clip.

    Parameters
    ----------
    video_path:
        Path to ``q{n}.mp4`` from Phase 4.
    sample_fps:
        How many frames per second to actually analyze. Default 5.
    landmarker:
        Optional pre-built ``mediapipe.tasks.python.vision.FaceLandmarker``.
        If ``None`` we build one in IMAGE mode from the cached model file.
        Tests pass a fake.
    """
    if not CV2_AVAILABLE:
        return _unavailable_video(str(video_path))
    if not MEDIAPIPE_AVAILABLE:
        return _mediapipe_unavailable_video(str(video_path))

    path = Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "no-file", "path": str(path)}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "error", "path": str(path), "error": "VideoCapture could not open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (n_frames_total / fps) if fps > 0 else 0.0
    step = max(1, int(round(fps / max(1, sample_fps))))

    owns_landmarker = False
    if landmarker is None:
        landmarker = _build_landmarker()
        owns_landmarker = landmarker is not None
    if landmarker is None:
        cap.release()
        return {"status": "no-model", "path": str(path),
                "duration_s": round(duration_s, 3)}

    looking_flags: list[bool] = []
    pos_flags: list[bool] = []
    tense_flags: list[bool] = []
    yaws: list[float] = []
    pitches: list[float] = []
    rolls: list[float] = []
    sampled = 0
    with_face = 0
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step != 0:
                idx += 1
                continue
            idx += 1
            sampled += 1

            obs = _analyze_frame(landmarker, frame)
            if obs is None:
                continue
            with_face += 1
            looking_flags.append(obs["looking"])
            pos_flags.append(obs["positive"])
            tense_flags.append(obs["tense"])
            yaws.append(obs["yaw"])
            pitches.append(obs["pitch"])
            rolls.append(obs["roll"])
    finally:
        cap.release()
        if owns_landmarker:
            try:
                landmarker.close()
            except Exception:  # pragma: no cover
                pass

    face_coverage = (with_face / sampled) if sampled else 0.0
    out: dict[str, Any] = {
        "status": "ok" if with_face > 0 else "no-face",
        "path": str(path),
        "duration_s": round(duration_s, 3),
        "frames_sampled": int(sampled),
        "frames_with_face": int(with_face),
        "face_coverage": round(face_coverage, 4),
        "gaze": _gaze_summary(looking_flags),
        "expression": _expression_summary(pos_flags, tense_flags),
        "head": _head_summary(yaws, pitches, rolls, sample_fps=sample_fps),
    }
    return out


# ---------------------------------------------------------------------------
# Landmarker construction
# ---------------------------------------------------------------------------
def _build_landmarker() -> Any | None:
    model_path = ensure_face_landmarker_model()
    if model_path is None:
        return None
    try:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        return mp_vision.FaceLandmarker.create_from_options(opts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Face Landmarker build failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------
def _analyze_frame(landmarker: Any, bgr_frame: np.ndarray) -> dict[str, Any] | None:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_image)
    except Exception as exc:  # noqa: BLE001
        logger.debug("landmarker.detect failed: %s", exc)
        return None

    faces = getattr(result, "face_landmarks", None) or []
    if not faces:
        return None

    landmarks = faces[0]
    blends = (result.face_blendshapes or [[]])[0]
    matrices = result.facial_transformation_matrixes or []

    looking = _is_looking_at_camera(landmarks)
    positive, tense = _classify_expression(blends)
    if matrices:
        yaw, pitch, roll = _decompose_rotation(matrices[0])
    else:
        yaw = pitch = roll = math.nan
    return {
        "looking": bool(looking),
        "positive": bool(positive),
        "tense": bool(tense),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
    }


def _is_looking_at_camera(landmarks) -> bool:
    try:
        l_outer, l_inner = landmarks[_LEFT_EYE_OUTER], landmarks[_LEFT_EYE_INNER]
        r_outer, r_inner = landmarks[_RIGHT_EYE_OUTER], landmarks[_RIGHT_EYE_INNER]
        l_iris, r_iris = landmarks[_LEFT_IRIS], landmarks[_RIGHT_IRIS]
    except IndexError:
        return False

    def offset(outer, inner, iris) -> float:
        eye_width = abs(inner.x - outer.x)
        if eye_width < 1e-6:
            return 1.0
        mid_x = (outer.x + inner.x) / 2.0
        return abs(iris.x - mid_x) / eye_width

    return offset(l_outer, l_inner, l_iris) < _GAZE_CENTER_THRESHOLD and \
           offset(r_outer, r_inner, r_iris) < _GAZE_CENTER_THRESHOLD


def _classify_expression(blendshapes) -> tuple[bool, bool]:
    """Return ``(positive_active, tense_active)`` for one frame."""
    by_name: dict[str, float] = {}
    for entry in blendshapes:
        # MediaPipe returns objects with .category_name and .score
        name = getattr(entry, "category_name", None) or getattr(entry, "name", None)
        score = float(getattr(entry, "score", 0.0))
        if name:
            by_name[name] = score
    pos = any(by_name.get(s, 0.0) >= _BLENDSHAPE_ACTIVE for s in _POSITIVE_SHAPES)
    tense = any(by_name.get(s, 0.0) >= _BLENDSHAPE_ACTIVE for s in _TENSE_SHAPES)
    return pos, tense


def _decompose_rotation(matrix4x4) -> tuple[float, float, float]:
    """Extract yaw/pitch/roll in degrees from a 4×4 transformation matrix.

    Convention: Tait-Bryan ZYX (yaw around Y, pitch around X, roll around Z).
    """
    m = np.asarray(matrix4x4, dtype=np.float64)
    if m.shape != (4, 4):
        return math.nan, math.nan, math.nan
    R = m[:3, :3]
    # Tait-Bryan ZYX where the *middle* axis is Y (yaw). For head-pose
    # semantics on a camera-facing user: yaw = side-to-side head turn (Y),
    # pitch = up/down nod (X), roll = head tilt (Z).
    sy = math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
    if sy < 1e-6:
        # Gimbal-lock: pitch is near ±90°; roll/yaw collapse — pick a branch.
        yaw = math.atan2(-R[2, 0], sy)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(-R[2, 0], sy)
        pitch = math.atan2(R[2, 1], R[2, 2])
        roll = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _gaze_summary(looking_flags: list[bool]) -> dict[str, Any]:
    if not looking_flags:
        return {"status": "no-data"}
    ratio = sum(looking_flags) / len(looking_flags)
    # Gaze-away event = transition from looking to not-looking.
    events = sum(
        1 for prev, cur in zip(looking_flags, looking_flags[1:])
        if prev and not cur
    )
    return {
        "status": "ok",
        "looking_ratio": round(ratio, 4),
        "gaze_away_events": int(events),
    }


def _expression_summary(pos_flags: list[bool], tense_flags: list[bool]) -> dict[str, Any]:
    n = len(pos_flags)
    if n == 0:
        return {"status": "no-data"}
    pos = sum(pos_flags) / n
    tense = sum(tense_flags) / n
    # Neutral = neither positive nor tense. Positive overrides if both fire.
    neutral = sum(1 for p, t in zip(pos_flags, tense_flags) if not p and not t) / n
    return {
        "status": "ok",
        "positive_ratio": round(pos, 4),
        "neutral_ratio": round(neutral, 4),
        "tense_ratio": round(tense, 4),
    }


def _head_summary(
    yaws: list[float], pitches: list[float], rolls: list[float], *, sample_fps: int,
) -> dict[str, Any]:
    valid_y = [v for v in yaws if not math.isnan(v)]
    valid_p = [v for v in pitches if not math.isnan(v)]
    valid_r = [v for v in rolls if not math.isnan(v)]
    if not valid_y:
        return {"status": "no-data"}

    def std(xs: list[float]) -> float:
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    # Direction-change rate: count |Δyaw| or |Δpitch| > threshold per sample.
    changes = 0
    for prev, cur in zip(valid_y, valid_y[1:]):
        if abs(cur - prev) > _HEAD_DELTA_DEG:
            changes += 1
    for prev, cur in zip(valid_p, valid_p[1:]):
        if abs(cur - prev) > _HEAD_DELTA_DEG:
            changes += 1
    duration_min = len(valid_y) / sample_fps / 60.0 if sample_fps > 0 else 0.0
    rate = (changes / duration_min) if duration_min > 0 else 0.0
    return {
        "status": "ok",
        "yaw_std_deg": round(std(valid_y), 2),
        "pitch_std_deg": round(std(valid_p), 2),
        "roll_std_deg": round(std(valid_r), 2),
        "direction_changes_per_min": round(rate, 2),
    }
