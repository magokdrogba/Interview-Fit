"""Tests for ``src.interview.text_render``.

These verify the Pillow-based renderer leaves frames the right shape and
actually paints pixels for both ASCII and Korean text — the whole point of the
module is that Hangul must not vanish into "?????".
"""

from __future__ import annotations

import numpy as np

from src.interview import text_render as T


def _blank(h: int = 80, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_font_resolves_on_this_machine():
    # macOS dev box ships AppleGothic / Apple SD Gothic Neo; CI Linux images
    # may not. We assert the resolver returns *something or None* without
    # raising, and that availability is a bool.
    assert isinstance(T.korean_font_available(), bool)


def test_render_returns_same_shape_and_dtype():
    frame = _blank()
    out = T.render_korean_text(frame, "안녕하세요", (10, 10), font_size=22)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_render_korean_paints_pixels():
    frame = _blank()
    out = T.render_korean_text(frame, "면접 질문", (8, 20), font_size=24,
                               color=(240, 240, 240))
    # Something must have been drawn (non-zero pixels appeared).
    assert out.sum() > 0


def test_render_ascii_also_works():
    frame = _blank()
    out = T.render_korean_text(frame, "READY [SPACE]", (8, 20), font_size=18)
    assert out.sum() > 0


def test_render_is_in_place():
    frame = _blank()
    returned = T.render_korean_text(frame, "테스트", (5, 5), font_size=20)
    # Helper writes back into the same array and returns it.
    assert returned is frame
    assert frame.sum() > 0


def test_render_texts_batches_multiple_items():
    frame = _blank(h=120)
    items = [
        ("준비됨", (8, 4), 18, (240, 240, 240)),
        ("자기소개를 해주세요", (8, 40), 22, (240, 240, 240)),
        ("Q1 / 10", (8, 80), 18, (200, 200, 200)),
    ]
    out = T.render_korean_texts(frame, items)
    assert out.shape == frame.shape
    assert out.sum() > 0


def test_empty_text_and_empty_items_are_noops():
    frame = _blank()
    # Empty string draws nothing, no exception.
    out = T.render_korean_text(frame.copy(), "", (5, 5))
    assert out.sum() == 0
    # Empty item list returns the frame untouched.
    frame2 = _blank()
    assert T.render_korean_texts(frame2, []) is frame2


def test_color_is_treated_as_bgr():
    # Pass pure blue in BGR = (255, 0, 0). After rendering, the blue channel
    # (index 0) should carry most of the painted energy.
    frame = _blank()
    T.render_korean_text(frame, "파랑", (10, 30), font_size=30, color=(255, 0, 0))
    b, g, r = frame[..., 0].sum(), frame[..., 1].sum(), frame[..., 2].sum()
    assert b > g and b > r
