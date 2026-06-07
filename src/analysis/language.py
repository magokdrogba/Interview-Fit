"""Post-hoc language analysis: STT (Whisper) + filler words + repetition + structure.

Public API
----------
* :func:`transcribe` — calls the Whisper API on a WAV file and returns
  ``{"text", "language", "segments"}``. Results are cached by audio-file hash
  + model name so re-runs cost nothing.
* :func:`analyze_language` — orchestrates: STT (unless caller passes a cached
  transcript), filler-word counting (English + Korean), content-word
  repetition + type-token ratio, and a GPT-scored answer-structure judgment
  (top-down / STAR / narrative / unstructured, 1-5 score, one-line comment).

Like :mod:`src.analysis.audio`, every public dict has a ``status`` field and
exceptions are never raised at the boundary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from config import CACHE_DIR, GPT_MODEL, OPENAI_API_KEY, WHISPER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler / stopword vocab
# ---------------------------------------------------------------------------
# Korean filler ("말버릇") words. We count only exact token matches against
# this list — content words like "전략", "시장", "분석" are never treated as
# fillers. Multi-word fillers (with a space) are matched as phrases first.
FILLER_WORDS: tuple[str, ...] = (
    "어", "음", "그", "근데", "그래서", "뭐", "약간", "그냥",
    "아무튼", "어쨌든", "이제", "뭔가", "사실", "진짜", "일단",
    "그니까", "그러니까", "좀", "막", "되게", "엄청", "너무",
    "이런", "저런", "그런", "뭐랄까", "어떻게 보면",
)
_FILLER_SINGLE: frozenset[str] = frozenset(w for w in FILLER_WORDS if " " not in w)
_FILLER_MULTI: tuple[str, ...] = tuple(w for w in FILLER_WORDS if " " in w)
# Korean syllable runs become candidate tokens for exact-match filler counting.
_KO_TOKEN_RE = re.compile(r"[가-힣]+")

# Minimal English + Korean stopword set for repetition scoring. Not linguistic
# — just enough to keep "the", "is", "and" from dominating the top words.
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "the", "a", "an", "and", "or", "but", "so", "if", "of", "to", "in", "on",
    "at", "for", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "it", "they",
    "my", "your", "our", "their", "his", "her", "its",
    "have", "has", "had", "do", "did", "does", "not", "no",
    "as", "from", "into", "than", "then", "there", "here", "very", "just", "also",
    # Korean (very rough — particles get attached so this is conservative)
    "그리고", "그래서", "하지만", "그러나", "또는", "그",
})


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def analyze_language(
    audio_path: Path | str,
    *,
    transcript: dict | None = None,
    question_text: str = "",
    client: Any | None = None,
    structure_model: str = GPT_MODEL,
    skip_structure: bool = False,
) -> dict[str, Any]:
    """Run language-axis analysis on one answer clip.

    The caller may pass a pre-computed ``transcript`` dict (the shape
    returned by :func:`transcribe`) to skip a Whisper round-trip. If
    ``client`` is None we build one from ``OPENAI_API_KEY``.
    """
    path = Path(audio_path)
    if transcript is None:
        transcript = transcribe(path, client=client)
    if transcript.get("status") != "ok":
        return {"status": transcript.get("status", "no-transcript"),
                "transcript": transcript}

    text = (transcript.get("text") or "").strip()
    language_code = transcript.get("language", "")

    out: dict[str, Any] = {
        "status": "ok",
        "transcript": {
            "text": text,
            "language": language_code,
            "char_count": len(text),
            "word_count": _word_count(text),
            "syllable_count": _syllable_count(text, language_code),
        },
        "fillers": _filler_metrics(text, transcript_duration_s=transcript.get("duration_s")),
        "repetition": _repetition_metrics(text),
    }

    if not skip_structure:
        out["structure"] = _structure_score(
            text=text,
            question_text=question_text,
            client=client,
            model=structure_model,
        )
    return out


# ---------------------------------------------------------------------------
# Whisper transcription + cache
# ---------------------------------------------------------------------------
def transcribe(
    audio_path: Path | str,
    *,
    client: Any | None = None,
    model: str = WHISPER_MODEL,
    language: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return ``{"status", "text", "language", "duration_s", "segments"}``.

    Cached at ``data/cache/stt_<sha256(file)+model>.json``. Misses go to the
    OpenAI API and the result is written before returning.
    """
    path = Path(audio_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "no-file", "path": str(path)}

    digest = _hash_file(path)
    cache_path = CACHE_DIR / f"stt_{digest}_{_safe(model)}.json"
    if not force_refresh and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("STT cache at %s unreadable; re-fetching", cache_path)

    if client is None:
        client = _default_client()

    try:
        with path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                language=language,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("STT failed for %s: %s", path, exc)
        return {"status": "error", "error": str(exc), "path": str(path)}

    payload = _normalize_stt_response(resp)
    payload["model"] = model
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        logger.warning("Failed to write STT cache: %s", exc)
    return payload


def _normalize_stt_response(resp: Any) -> dict[str, Any]:
    """Adapt the SDK response object into our cache schema. The SDK returns a
    pydantic model; ``.model_dump()`` is the documented dict shape."""
    # The SDK exposes either a dict-like object or a pydantic model.
    if hasattr(resp, "model_dump"):
        raw = resp.model_dump()
    elif isinstance(resp, dict):
        raw = resp
    else:
        raw = {"text": str(resp)}

    text = raw.get("text", "") or ""
    return {
        "status": "ok",
        "text": text,
        "language": raw.get("language", ""),
        "duration_s": float(raw.get("duration") or 0.0),
        "segments": raw.get("segments") or [],
    }


# ---------------------------------------------------------------------------
# Filler / repetition / structure
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[\w']+", flags=re.UNICODE)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _syllable_count(text: str, language_code: str) -> int:
    """Crude syllable estimator: Korean Hangul syllable blocks (U+AC00–U+D7A3)
    each count as 1; for everything else, fall back to a simple English
    syllable heuristic (vowel-group counting). Good enough for SPM."""
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    if hangul > 0 and (language_code.startswith("ko") or hangul > len(text) * 0.3):
        return hangul + _english_syllables(re.sub(r"[가-힣]+", " ", text))
    return _english_syllables(text)


def _english_syllables(text: str) -> int:
    total = 0
    for word in _WORD_RE.findall(text):
        w = word.lower()
        groups = re.findall(r"[aeiouy]+", w)
        s = len(groups)
        if w.endswith("e") and s > 1:
            s -= 1
        total += max(1, s)
    return total


def _filler_metrics(text: str, *, transcript_duration_s: float | None) -> dict[str, Any]:
    """Count Korean filler words by exact token / phrase match.

    Returns ``total``, ``per_minute``, and ``counts`` ({word: count} sorted by
    frequency descending). Content words are never counted because we only
    match tokens that are exactly in :data:`FILLER_WORDS`.
    """
    if not text.strip():
        return {"status": "empty", "total": 0, "counts": {}, "per_minute": 0.0}

    counts: Counter[str] = Counter()

    # Phrase fillers first (e.g. "어떻게 보면"), removing matches so their
    # constituent tokens aren't double-counted as single-word fillers.
    working = text
    for phrase in _FILLER_MULTI:
        n = working.count(phrase)
        if n:
            counts[phrase] += n
            working = working.replace(phrase, " ")

    # Single-token fillers: exact match against Korean syllable runs.
    for token in _KO_TOKEN_RE.findall(working):
        if token in _FILLER_SINGLE:
            counts[token] += 1

    total = int(sum(counts.values()))
    if transcript_duration_s and transcript_duration_s > 0:
        per_minute = round(total / transcript_duration_s * 60.0, 2)
    else:
        # Approximate from word count assuming 150 wpm — better than zero.
        wc = _word_count(text) or 1
        per_minute = round(total / wc * 150.0, 2)

    counts_sorted = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    return {"status": "ok", "counts": counts_sorted, "total": total, "per_minute": per_minute}


def _repetition_metrics(text: str) -> dict[str, Any]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return {"status": "empty", "top": [], "type_token_ratio": 0.0}

    content = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    counter = Counter(content)
    top = [{"word": w, "count": c} for w, c in counter.most_common(5) if c >= 3]
    ttr = round(len(set(words)) / len(words), 4)
    return {"status": "ok", "top": top, "type_token_ratio": ttr,
            "total_words": len(words), "content_words": len(content)}


# ---------------------------------------------------------------------------
# GPT-based answer-structure scorer
# ---------------------------------------------------------------------------
_STRUCTURE_SYSTEM = (
    "당신은 컨설팅 면접 채점관입니다. 아래 루브릭에 따라 답변을 평가하고, "
    '반드시 JSON으로만 응답하세요: {"score": N, "reason": "..."}'
)

# Rubric template. ``reason`` should explain, in Korean, which rubric items
# were met. We fill {question}/{transcript} at call time.
_STRUCTURE_RUBRIC = """\
다음 루브릭으로 1–5점을 채점하세요. 각 항목을 개별 확인 후 합산하세요.

[+1점] 결론을 먼저 말했는가? (두괄식 시작)
[+1점] 답변을 명시적으로 구조화했는가?
       (예: "첫째/둘째/셋째", "세 가지로 말씀드리면", "크게 두 측면에서")
[+1점] 각 항목에 구체적인 사례나 수치가 포함되어 있는가?
[+1점] 논리적 흐름이 자연스러운가? (원인→결과, 문제→해결 등)
[+1점] 마무리 문장으로 답변이 명확하게 종결되는가?

채점 기준:
5점: 컨설팅 면접 수준의 완성도 높은 구조
4점: 체계적, 일부 보완 가능
3점: 구조가 있으나 불완전
2점: 구조 시도가 있으나 일관성 부족
1점: 나열식, 결론 없음

reason 필드에는 어떤 항목을 충족/미충족했는지 한국어로 간단히 설명하세요.

평가할 답변: {transcript}
면접 질문: {question}
"""


def _structure_score(
    *,
    text: str,
    question_text: str,
    client: Any | None,
    model: str,
) -> dict[str, Any]:
    if not text.strip():
        return {"status": "empty"}
    if client is None and not OPENAI_API_KEY:
        return {"status": "no-api-key"}
    if client is None:
        client = _default_client()

    user_msg = _STRUCTURE_RUBRIC.format(
        transcript=text.strip(),
        question=question_text.strip() or "(질문 정보 없음)",
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _STRUCTURE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Structure scoring failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    # Accept int or numeric-string scores; clamp to 1..5.
    raw_score = parsed.get("score")
    try:
        score = int(round(float(raw_score)))
    except (TypeError, ValueError):
        return {"status": "invalid-response", "raw": parsed}
    if not 1 <= score <= 5:
        return {"status": "invalid-response", "raw": parsed}

    reason = str(parsed.get("reason") or "").strip()
    return {
        "status": "ok",
        "score": score,
        "reason": reason,
        # Keep ``comment`` as an alias so existing readers keep working.
        "comment": reason,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hash_file(path: Path, *, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-")


def _default_client() -> Any:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env or pass "
            "a `client=` argument to analyze_language()."
        )
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)
