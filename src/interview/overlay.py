"""Lightweight real-time face-landmark overlay for the live interview window.

Uses the **MediaPipe Tasks Face Landmarker v2** in VIDEO mode. The same model
file is reused in Phase 5 for post-hoc analysis (blendshapes + transformation
matrix), so downloading it now is on-budget.

Design notes
------------
* The model (~3 MB) is downloaded on first use into
  ``data/cache/face_landmarker.task`` and cached forever.
* If the download fails (offline, blocked) the overlay degrades to a **no-op**:
  ``.draw(frame, ts)`` returns the frame unchanged. The interview can still
  proceed; the user just doesn't see the dots.
* We draw **only landmark dots** here. Anything beyond that — gaze direction,
  blendshapes, head pose — is deliberately deferred to Phase 5 to keep this
  loop cheap.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import FACE_LANDMARKER_PATH, FACE_LANDMARKER_URL

logger = logging.getLogger(__name__)

# opencv-python needs libGL (a display) which Streamlit Cloud lacks. We install
# opencv-python-headless there, but import defensively so a missing build
# degrades to a no-op overlay instead of crashing at import time.
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False
    logger.warning("cv2 (OpenCV) is unavailable; the face-landmark overlay is disabled.")

# Dot color (BGR) and radius. Subtle so they don't obscure the user's face.
_DOT_COLOR: tuple[int, int, int] = (0, 255, 0)
_DOT_RADIUS: int = 1


# ---------------------------------------------------------------------------
# Model file
# ---------------------------------------------------------------------------
def ensure_face_landmarker_model(
    path: Path = FACE_LANDMARKER_PATH,
    url: str = FACE_LANDMARKER_URL,
) -> Path | None:
    """Download the Face Landmarker .task file on first use.

    Returns the path on success, ``None`` if the download failed. The caller
    decides how to degrade.
    """
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        logger.info("Downloading Face Landmarker model from %s", url)
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
        return path
    except Exception as exc:  # noqa: BLE001 - any failure -> graceful no-op
        logger.warning("Face Landmarker download failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
class FaceLandmarkOverlay:
    """Thin wrapper around a MediaPipe Face Landmarker in VIDEO mode.

    Construct once per session, call :meth:`draw` per frame, close on exit
    (or use the instance as a context manager).
    """

    def __init__(self, landmarker: Any | None = None):
        """Build a landmarker. If ``landmarker`` is provided (e.g. by a test
        fixture or a caller that wants a different config) it is used as-is;
        the wrapper assumes ownership and will close it.

        Otherwise, attempt to construct one from the cached/downloaded model.
        If the model is unavailable, ``self._landmarker`` stays ``None`` and
        :meth:`draw` becomes a no-op.
        """
        self._landmarker = landmarker
        self._owns = landmarker is not None  # if injected, still close it on exit

        if self._landmarker is None:
            model_path = ensure_face_landmarker_model()
            if model_path is not None:
                try:
                    base = mp_python.BaseOptions(model_asset_path=str(model_path))
                    opts = mp_vision.FaceLandmarkerOptions(
                        base_options=base,
                        running_mode=mp_vision.RunningMode.VIDEO,
                        num_faces=1,
                        # Blendshapes and transformation matrix are heavier and
                        # only needed in Phase 5 post-processing.
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
                    self._owns = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Face Landmarker init failed: %s", exc)
                    self._landmarker = None

    @property
    def active(self) -> bool:
        """True if drawing will actually do anything."""
        return self._landmarker is not None

    def draw(self, bgr_frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Annotate ``bgr_frame`` in place with landmark dots and return it.

        If the model is not loaded, returns the frame unchanged.
        """
        if (
            not CV2_AVAILABLE
            or self._landmarker is None
            or bgr_frame is None
            or bgr_frame.size == 0
        ):
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        except Exception as exc:  # noqa: BLE001
            logger.debug("landmarker.detect_for_video failed: %s", exc)
            return bgr_frame

        for face in result.face_landmarks or []:
            for lm in face:
                x = int(lm.x * w)
                y = int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(bgr_frame, (x, y), _DOT_RADIUS, _DOT_COLOR, -1)
        return bgr_frame

    def close(self) -> None:
        if self._landmarker is not None and self._owns:
            try:
                self._landmarker.close()
            except Exception:  # pragma: no cover
                pass
        self._landmarker = None

    def __enter__(self) -> "FaceLandmarkOverlay":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
