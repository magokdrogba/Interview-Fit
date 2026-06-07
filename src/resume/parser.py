"""Resume parsing: PDF or plain text -> normalized text + best-effort sections.

Public API
----------
- ``parse_resume(source)`` accepts:
    * ``pathlib.Path`` or ``str`` path to a ``.pdf`` or ``.txt`` file
    * ``bytes`` (raw PDF bytes, e.g. from a Streamlit ``UploadedFile``)
    * raw text string (treated as plain text; no path resolution attempted
      if it contains newlines, whitespace, or no path-like characters)
- Returns a :class:`ParsedResume` dataclass with ``raw_text`` and ``sections``.

Section detection is a heuristic. We scan line-by-line for short lines whose
trimmed content matches a known heading (English + Korean), and split the
body between consecutive headings. Anything before the first heading lands in
``sections["_preamble"]``.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pdfplumber

# ---------------------------------------------------------------------------
# Section heading dictionary
# ---------------------------------------------------------------------------
# Maps a *canonical key* -> list of regex alternations (case-insensitive,
# anchored later when matched). Korean and English headings are both included
# because most users on this project will work with Korean job boards.
_SECTION_HEADINGS: Final[dict[str, list[str]]] = {
    "summary": [
        r"summary", r"profile", r"about\s*me", r"introduction",
        r"자기\s*소개", r"소개", r"프로필",
    ],
    "experience": [
        r"experience", r"work\s*experience", r"employment",
        r"professional\s*experience", r"career",
        r"경력", r"경력\s*사항", r"업무\s*경험", r"근무\s*경력",
    ],
    "projects": [
        r"projects?", r"selected\s*projects",
        r"프로젝트", r"주요\s*프로젝트",
    ],
    "skills": [
        r"skills?", r"technical\s*skills", r"tech\s*stack", r"core\s*skills",
        r"기술", r"기술\s*스택", r"보유\s*기술", r"스킬",
    ],
    "education": [
        r"education", r"academic\s*background",
        r"학력", r"학력\s*사항", r"교육",
    ],
    "certifications": [
        r"certifications?", r"licenses?",
        r"자격증", r"자격\s*사항",
    ],
    "awards": [
        r"awards?", r"honou?rs?", r"achievements?",
        r"수상", r"수상\s*경력",
    ],
    "publications": [
        r"publications?", r"papers?",
        r"논문", r"출판",
    ],
    "languages": [
        r"languages?",
        r"어학", r"외국어",
    ],
}

# Pre-compile a single regex that matches *any* heading and captures it.
# Headings are typically short standalone lines, optionally followed by a
# colon. We allow leading/trailing whitespace and optional surrounding
# punctuation (e.g. "■ Experience", "## Skills").
_HEADING_ALTERNATIONS: Final[str] = "|".join(
    f"(?P<{key}>{'|'.join(alts)})" for key, alts in _SECTION_HEADINGS.items()
)
_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    rf"^[\s\W_]{{0,4}}(?:{_HEADING_ALTERNATIONS})\s*[:：]?\s*$",
    flags=re.IGNORECASE,
)


@dataclass
class ParsedResume:
    """Structured resume content.

    Attributes
    ----------
    raw_text:
        Full normalized plain text of the resume (whitespace collapsed,
        trailing spaces stripped, blank lines collapsed to at most one).
    sections:
        Mapping from canonical section key (see ``_SECTION_HEADINGS`` keys)
        to that section's body text. Always contains ``_preamble`` for any
        content that appears before the first detected heading; this is the
        only key that may be present without a corresponding detected
        heading.
    source_type:
        ``"pdf"``, ``"text"``, or ``"bytes"`` — recorded for debugging.
    """

    raw_text: str
    sections: dict[str, str] = field(default_factory=dict)
    source_type: str = "text"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_resume(source: str | Path | bytes) -> ParsedResume:
    """Parse a resume from a path, raw bytes, or raw text.

    Path resolution is conservative: if ``source`` is a string that looks
    like multi-line content (contains newlines) it is treated as raw text
    regardless of whether an identically-named file might exist.
    """
    raw_text, source_type = _extract_text(source)
    normalized = _normalize_whitespace(raw_text)
    sections = _split_sections(normalized)
    return ParsedResume(raw_text=normalized, sections=sections, source_type=source_type)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _extract_text(source: str | Path | bytes) -> tuple[str, str]:
    """Dispatch on the source type and return (text, source_type_label)."""
    if isinstance(source, bytes):
        return _extract_pdf_bytes(source), "bytes"

    if isinstance(source, Path):
        return _extract_from_path(source)

    if isinstance(source, str):
        # Multi-line strings are clearly raw text, never a path.
        if "\n" in source or len(source) > 1000:
            return source, "text"
        # Otherwise treat as a path *if it exists*; else as raw text.
        candidate = Path(source)
        if candidate.exists() and candidate.is_file():
            return _extract_from_path(candidate)
        return source, "text"

    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _extract_from_path(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_bytes(path.read_bytes()), "pdf"
    # Treat anything else as plain text (".txt", ".md", or no extension).
    return path.read_text(encoding="utf-8", errors="replace"), "text"


def _extract_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes via pdfplumber. Joins pages with blank lines."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_WHITESPACE_RUN = re.compile(r"[ \t ]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and limit blank lines to at most one."""
    # Normalize line endings first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of horizontal whitespace inside each line.
    lines = [_WHITESPACE_RUN.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines to 2 (i.e. one blank line maximum).
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------
def _split_sections(text: str) -> dict[str, str]:
    """Walk the text line-by-line, splitting at recognised headings.

    A line is treated as a heading only if it is short (<= 40 chars after
    stripping) — long lines containing the word "experience" inside a
    sentence won't be mistaken for a heading.
    """
    sections: dict[str, list[str]] = {"_preamble": []}
    order: list[str] = ["_preamble"]
    current = "_preamble"

    for line in text.split("\n"):
        stripped = line.strip()
        if 0 < len(stripped) <= 40:
            match = _HEADING_RE.match(stripped)
            if match and match.lastgroup is not None:
                current = match.lastgroup
                if current not in sections:
                    sections[current] = []
                    order.append(current)
                continue
        sections[current].append(line)

    # Materialize, stripping leading/trailing blank lines per section.
    out: dict[str, str] = {}
    for key in order:
        body = "\n".join(sections[key]).strip()
        if body:
            out[key] = body
    return out
