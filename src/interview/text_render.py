"""Korean-capable text rendering for the OpenCV interview HUD.

``cv2.putText`` uses Hershey vector fonts that contain no Hangul glyphs, so any
Korean text renders as a row of "?????". This module draws text with Pillow
instead (full Unicode + TrueType/OpenType support), converting to and from the
BGR ``numpy`` frames that OpenCV uses.

Public API
----------
* :func:`render_korean_text` — draw a single string onto a frame.
* :func:`render_korean_texts` — draw several strings in one pass (one BGR↔RGB
  conversion total, so the HUD stays cheap at 30 fps).
* :func:`korean_font_available` — True if a Korean-capable font was found.

Font resolution order
----------------------
1. Any ``.ttf`` / ``.otf`` / ``.ttc`` bundled under ``<project>/assets/fonts``.
2. Known macOS system fonts (AppleGothic, Apple SD Gothic Neo, Arial Unicode).

If nothing is found we fall back to Pillow's built-in bitmap font (which still
lacks Hangul) and log a warning — callers keep working, text just won't be
pretty.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# A single (text, top_left_xy, font_size_px, bgr_color) draw request.
TextItem = tuple[str, tuple[int, int], int, tuple[int, int, int]]

# Known Korean-capable system fonts, in priority order. AppleGothic.ttf is the
# path called out in the project spec; Apple SD Gothic Neo is the nicer modern
# Korean face; Arial Unicode is a last-resort universal fallback.
_SYSTEM_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Common Linux fallbacks, harmless if absent.
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

_BUNDLED_FONT_DIR = PROJECT_ROOT / "assets" / "fonts"


@lru_cache(maxsize=1)
def _resolve_font_path() -> str | None:
    """Return the path to the first usable Korean-capable font, or None."""
    if _BUNDLED_FONT_DIR.is_dir():
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for path in sorted(_BUNDLED_FONT_DIR.glob(pattern)):
                return str(path)
    for candidate in _SYSTEM_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=32)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load (and cache) a font at the requested pixel size."""
    path = _resolve_font_path()
    if path is None:
        logger.warning(
            "No Korean-capable font found; Hangul will not render correctly. "
            "Drop a .ttf into %s to fix.", _BUNDLED_FONT_DIR
        )
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(path, size)
    except Exception as exc:  # noqa: BLE001 - corrupt/unsupported font file
        logger.warning("Failed to load font %s at size %d: %s", path, size, exc)
        return ImageFont.load_default()


def korean_font_available() -> bool:
    """True if a Korean-capable TrueType/OpenType font was located."""
    return _resolve_font_path() is not None


def render_korean_texts(frame: np.ndarray, items: Sequence[TextItem]) -> np.ndarray:
    """Draw several text items onto a BGR ``frame`` in a single pass.

    Each item is ``(text, (x, y), font_size_px, (b, g, r))`` where ``(x, y)``
    is the **top-left** corner (Pillow convention), not the cv2 baseline.
    Colors are given in **BGR** to match the rest of the OpenCV code. The
    frame is modified in place and also returned.
    """
    if frame is None or frame.size == 0 or not items:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    for text, position, font_size, color in items:
        if not text:
            continue
        font = _load_font(int(font_size))
        # cv2 colors are BGR; Pillow wants RGB.
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))
        draw.text(position, text, font=font, fill=rgb_color)

    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = out
    return frame


def render_korean_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """Draw a single Korean (or any-Unicode) string onto a BGR ``frame``.

    Drop-in replacement for ``cv2.putText`` for non-ASCII text. ``position``
    is the top-left corner; ``color`` is BGR. Returns the modified frame.
    """
    return render_korean_texts(frame, [(text, position, font_size, color)])
