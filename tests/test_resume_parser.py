"""Tests for ``src.resume.parser``.

Covers:
  * raw-text input → normalized output, no sections lost
  * heading detection (English + Korean)
  * path-style input (``.txt`` round-trip via ``tmp_path``)
  * PDF round-trip if a PDF generator is available; otherwise skipped
"""

from __future__ import annotations

import importlib.util

import pytest

from src.resume.parser import ParsedResume, parse_resume


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_EN = """\
Jane Doe
jane@example.com   |   +1-555-0100

Summary
Backend engineer with 5 years of experience shipping payment systems.

Experience
ACME Corp — Senior Engineer (2022–present)
  * Led migration of billing service to event-sourced architecture.
  * Reduced p99 latency by 38% via async batching.

Projects
- Open-source contributor to FastAPI ecosystem.

Skills
Python, Go, PostgreSQL, Kafka, Kubernetes
"""

SAMPLE_KR = """\
홍길동
이메일: hong@example.com

자기 소개
사용자 경험에 집착하는 5년차 백엔드 엔지니어.

경력
토스 — 시니어 엔지니어 (2022–현재)
  • 결제 게이트웨이 신규 설계 및 배포

프로젝트
- 사내 모니터링 대시보드 오픈소스화

기술 스택
Python, Kotlin, Kafka, gRPC
"""


# ---------------------------------------------------------------------------
# Plain-text inputs
# ---------------------------------------------------------------------------
def test_parse_text_preserves_full_content() -> None:
    parsed = parse_resume(SAMPLE_EN)
    assert isinstance(parsed, ParsedResume)
    assert parsed.source_type == "text"
    # Every distinctive substring survives normalization.
    for needle in ("Jane Doe", "ACME Corp", "FastAPI", "Kubernetes"):
        assert needle in parsed.raw_text


def test_english_section_detection() -> None:
    parsed = parse_resume(SAMPLE_EN)
    assert {"summary", "experience", "projects", "skills"} <= set(parsed.sections)
    assert "ACME Corp" in parsed.sections["experience"]
    assert "FastAPI" in parsed.sections["projects"]
    assert "Kubernetes" in parsed.sections["skills"]


def test_korean_section_detection() -> None:
    parsed = parse_resume(SAMPLE_KR)
    keys = set(parsed.sections)
    assert {"summary", "experience", "projects", "skills"} <= keys
    assert "토스" in parsed.sections["experience"]
    assert "Kafka" in parsed.sections["skills"]


def test_preamble_captured_when_no_leading_heading() -> None:
    parsed = parse_resume(SAMPLE_EN)
    assert "_preamble" in parsed.sections
    assert "jane@example.com" in parsed.sections["_preamble"]


def test_no_headings_yields_single_preamble_section() -> None:
    parsed = parse_resume("Just a freeform blurb with no structured headings at all.")
    assert list(parsed.sections.keys()) == ["_preamble"]


def test_whitespace_is_normalized() -> None:
    messy = "Line  one   has    runs.\r\n\r\n\r\n\r\nLine two."
    parsed = parse_resume(messy)
    # Horizontal runs collapsed to a single space.
    assert "Line one has runs." in parsed.raw_text
    # Triple+ blank lines collapsed to a single blank line.
    assert "\n\n\n" not in parsed.raw_text


def test_long_line_with_heading_word_is_not_a_heading() -> None:
    text = (
        "Summary\n"
        "I have extensive experience leading distributed teams across timezones, "
        "and my experience scales beyond just engineering management.\n"
    )
    parsed = parse_resume(text)
    # The long second line contains 'experience' but must stay inside summary.
    assert "experience" not in parsed.sections  # no new section opened
    assert "extensive experience" in parsed.sections["summary"]


# ---------------------------------------------------------------------------
# Path-based inputs
# ---------------------------------------------------------------------------
def test_parse_from_txt_path(tmp_path) -> None:
    f = tmp_path / "resume.txt"
    f.write_text(SAMPLE_EN, encoding="utf-8")
    parsed = parse_resume(f)
    assert parsed.source_type == "text"
    assert "ACME Corp" in parsed.raw_text


def test_parse_from_string_path(tmp_path) -> None:
    f = tmp_path / "resume.txt"
    f.write_text("Just a one-line resume.", encoding="utf-8")
    parsed = parse_resume(str(f))
    assert parsed.source_type == "text"
    assert "one-line resume" in parsed.raw_text


# ---------------------------------------------------------------------------
# PDF round-trip (skipped if no PDF generator is available)
# ---------------------------------------------------------------------------
def _has_reportlab() -> bool:
    return importlib.util.find_spec("reportlab") is not None


@pytest.mark.skipif(not _has_reportlab(), reason="reportlab not installed")
def test_parse_pdf_bytes_round_trip() -> None:  # pragma: no cover - optional
    from io import BytesIO

    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Experience")
    c.drawString(72, 700, "ACME Corp - Senior Engineer")
    c.save()
    parsed = parse_resume(buf.getvalue())
    assert parsed.source_type == "bytes"
    assert "ACME Corp" in parsed.raw_text
