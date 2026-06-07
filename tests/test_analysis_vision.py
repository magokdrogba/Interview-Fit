"""Tests for ``src.analysis.vision``.

We never download the real Face Landmarker model or open a real camera. The
landmarker is replaced by a scripted fake whose per-frame outputs we control.
A short synthetic MP4 is written with OpenCV to exercise the capture loop.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.analysis import vision as V


def _write_mp4(path, n_frames=10, w=64, h=48, fps=10) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = (np.ones((h, w, 3), dtype="uint8") * (i * 20 % 255)).astype("uint8")
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Fake landmarker
# ---------------------------------------------------------------------------
class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Blendshape:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


def _make_landmarks(iris_offset_x: float = 0.0) -> list[_Landmark]:
    """Build a 478-point landmark list where the eye corners and iris centers
    correspond to a face looking ~straight at the camera (or off-center per
    ``iris_offset_x``)."""
    lm = [_Landmark() for _ in range(478)]
    # Left eye corners (using V's indices)
    lm[V._LEFT_EYE_OUTER] = _Landmark(x=0.30)
    lm[V._LEFT_EYE_INNER] = _Landmark(x=0.40)
    lm[V._RIGHT_EYE_OUTER] = _Landmark(x=0.70)
    lm[V._RIGHT_EYE_INNER] = _Landmark(x=0.60)
    # Iris centers initially at eye midpoints (0.35 / 0.65), shifted by offset
    lm[V._LEFT_IRIS] = _Landmark(x=0.35 + iris_offset_x)
    lm[V._RIGHT_IRIS] = _Landmark(x=0.65 + iris_offset_x)
    return lm


class FakeLandmarker:
    """Scripted landmarker.

    Each call to ``.detect()`` consumes the next entry in ``observations``.
    An entry of ``None`` simulates "no face detected".
    """

    def __init__(self, observations):
        self._obs = list(observations)
        self.closed = False

    def detect(self, mp_image):
        if not self._obs:
            return SimpleNamespace(face_landmarks=[], face_blendshapes=[],
                                   facial_transformation_matrixes=[])
        item = self._obs.pop(0)
        if item is None:
            return SimpleNamespace(face_landmarks=[], face_blendshapes=[],
                                   facial_transformation_matrixes=[])
        return SimpleNamespace(
            face_landmarks=[item["landmarks"]],
            face_blendshapes=[item["blendshapes"]],
            facial_transformation_matrixes=[item["matrix"]],
        )

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------
def test_no_file_returns_no_file(tmp_path):
    out = V.analyze_video(tmp_path / "nope.mp4")
    assert out["status"] == "no-file"


def test_no_model_returns_no_model(tmp_path, monkeypatch):
    p = tmp_path / "tiny.mp4"
    _write_mp4(p)
    monkeypatch.setattr(V, "ensure_face_landmarker_model", lambda *_a, **_k: None)
    out = V.analyze_video(p)
    assert out["status"] == "no-model"
    assert out["duration_s"] > 0


def test_graceful_fallback_when_cv2_unavailable(monkeypatch):
    """On platforms without OpenCV (opencv-python needs libGL, absent on
    Streamlit Cloud) analyze_video must not crash: it returns safe-default
    no-data metrics."""
    monkeypatch.setattr(V, "CV2_AVAILABLE", False)
    out = V.analyze_video("anything.mp4")
    assert out["status"] == "no-cv2"
    assert out["gaze"]["status"] == "no-data"
    assert out["expression"]["status"] == "no-data"
    assert out["head"]["status"] == "no-data"


def test_no_face_detected_marks_no_face(tmp_path):
    p = tmp_path / "tiny.mp4"
    _write_mp4(p, n_frames=20, fps=10)
    fake = FakeLandmarker([None] * 30)
    out = V.analyze_video(p, landmarker=fake)
    assert out["status"] == "no-face"
    assert out["frames_with_face"] == 0
    # Sub-blocks degrade to no-data rather than crashing.
    assert out["gaze"]["status"] == "no-data"
    assert out["expression"]["status"] == "no-data"
    assert out["head"]["status"] == "no-data"


# ---------------------------------------------------------------------------
# Happy path — gaze + expression + head
# ---------------------------------------------------------------------------
def _identity_matrix() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _rotate_y_matrix(deg: float) -> np.ndarray:
    r = math.radians(deg)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = math.cos(r); m[0, 2] = math.sin(r)
    m[2, 0] = -math.sin(r); m[2, 2] = math.cos(r)
    return m


def test_full_pipeline_with_scripted_faces(tmp_path):
    n = 20
    p = tmp_path / "vid.mp4"
    # Match source fps and sample_fps below so every observation gets consumed.
    _write_mp4(p, n_frames=n, fps=5)

    # Build a sequence of observations: first half looking + smiling at yaw 0,
    # second half looking-away with a frown at yaw 25.
    smile = [_Blendshape("mouthSmileLeft", 0.6), _Blendshape("mouthSmileRight", 0.6)]
    frown = [_Blendshape("browDownLeft", 0.7), _Blendshape("browDownRight", 0.7)]

    obs: list[dict] = []
    for i in range(n):
        if i < n // 2:
            obs.append({
                "landmarks": _make_landmarks(iris_offset_x=0.0),  # centered
                "blendshapes": smile,
                "matrix": _identity_matrix(),
            })
        else:
            obs.append({
                "landmarks": _make_landmarks(iris_offset_x=0.05),  # shifted right beyond threshold
                "blendshapes": frown,
                "matrix": _rotate_y_matrix(25.0 if i % 2 else -25.0),
            })

    fake = FakeLandmarker(obs)
    out = V.analyze_video(p, landmarker=fake, sample_fps=5)
    assert out["status"] == "ok"
    assert out["frames_with_face"] > 0

    # Gaze: half the time we were looking at the camera, half we weren't.
    g = out["gaze"]
    assert g["status"] == "ok"
    assert 0.2 < g["looking_ratio"] < 0.8, g
    assert g["gaze_away_events"] >= 1

    # Expression: positive should be visibly nonzero (first half smiled).
    e = out["expression"]
    assert e["status"] == "ok"
    assert e["positive_ratio"] > 0.1
    assert e["tense_ratio"] > 0.1

    # Head pose: yaw alternated between large +/- in the second half.
    h = out["head"]
    assert h["status"] == "ok"
    assert h["yaw_std_deg"] > 0
    assert h["direction_changes_per_min"] > 0

    assert fake.closed is False  # caller-supplied landmarker is not closed


# ---------------------------------------------------------------------------
# Rotation decomposition unit
# ---------------------------------------------------------------------------
def test_rotation_decomposition_identity():
    y, p, r = V._decompose_rotation(np.eye(4))
    assert abs(y) < 1e-6 and abs(p) < 1e-6 and abs(r) < 1e-6


def test_rotation_decomposition_yaw():
    m = _rotate_y_matrix(30.0)
    y, p, r = V._decompose_rotation(m)
    assert abs(y - 30.0) < 1.0, (y, p, r)
    assert abs(p) < 1.0
    assert abs(r) < 1.0
