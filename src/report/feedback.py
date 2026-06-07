"""Feedback report builder.

Consumes the dict returned by :func:`src.analysis.orchestrator.analyze_session`
and asks GPT to translate the quantitative axis metrics into:

* a 3–5 sentence ``overall_summary``
* per-question one-line comments
* **exactly three** ``priorities`` — the highest-impact habits to work on,
  each with an ``observation`` (what the data shows) and a concrete ``action``
  (what to practice).

Design choices
--------------
* JSON-only output via ``response_format={"type": "json_object"}``; we
  validate against a strict schema and retry once on violation.
* Cache the final report next to the input analysis as ``report.json`` so
  re-opening Streamlit doesn't trigger another model call.
* Pure-Python fallback when the analysis itself contains no usable data
  (e.g. every question was skipped) — we never call the model in that case.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Final

from config import GPT_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

ALLOWED_AREAS: Final[frozenset[str]] = frozenset({"vision", "audio", "language"})
PRIORITY_COUNT: Final[int] = 3


class ReportGenerationError(RuntimeError):
    """Raised when we cannot get a schema-valid report after a retry."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_report(
    analysis: dict[str, Any],
    *,
    session_dir: Path | None = None,
    client: Any | None = None,
    model: str = GPT_MODEL,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return a structured feedback report for one session's analysis dict.

    Parameters
    ----------
    analysis:
        The dict produced by ``analyze_session``. Pass the dict directly so
        this function is usable in tests without touching disk.
    session_dir:
        Optional. If provided we cache the result at
        ``<session_dir>/report.json`` and read it back on subsequent calls
        unless ``force_refresh`` is set.
    client:
        Optional OpenAI client (test-injectable). Falls back to a real one
        built from ``OPENAI_API_KEY``.
    """
    if not isinstance(analysis, dict):
        raise TypeError("analysis must be a dict")
    if analysis.get("status") != "ok":
        return _empty_report(analysis)
    answered = (analysis.get("aggregate") or {}).get("answered", 0)
    if not answered:
        return _empty_report(analysis)

    cache_path = (Path(session_dir) / "report.json") if session_dir else None
    if cache_path and not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["_from_cache"] = True
            return cached
        except (OSError, json.JSONDecodeError):
            logger.warning("report.json cache unreadable; recomputing")

    if client is None:
        client = _default_client()

    report = _call_with_retry(client=client, model=model, analysis=analysis)
    report["session_id"] = analysis.get("session_id", "")
    report["company"] = analysis.get("company", "")
    report["model"] = model

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover
            logger.warning("Failed to write report cache %s: %s", cache_path, exc)
    return report


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT: Final[str] = """\
당신은 한 지원자의 모의면접 세션을 검토하는 면접 코치입니다. (a) 전체 답변에 대한 \
종합 행동 지표와 (b) 질문별 지표 및 답변 전문을 받게 됩니다. 간결한 피드백 \
리포트를 작성하세요.

모든 피드백은 반드시 한국어로 작성하세요.

다음과 정확히 같은 형태의 JSON 객체를 반환하세요:

{
  "overall_summary": "<다음 세 가지를 순서대로 담은 한국어 문단: \
(a) 전체 총평 1-2문장, (b) 가장 잘한 점 1가지, (c) 가장 시급한 개선점 1가지. \
지표 수치를 근거로 구체적으로 쓰고, '~일 수도 있다' 같은 모호한 표현은 피하세요.>",
  "per_question": [
    {"index": <입력과 동일한 정수>,
     "comment": "<정확히 다음 두 줄 형식의 한국어:\\n\
핵심: [이 답변의 가장 두드러진 점 1가지]\\n\
개선: [다음 번에 어떻게 하면 좋은지 행동 중심으로 1-2문장]\\n\
지표 수치(숫자나 판정)를 최소 1개 인용하세요. 답변 데이터가 없으면 그 사실을 \
간단히 적으세요.>"}
  ],
  "priorities": [
    {"area": "<vision | audio | language 중 하나>",
     "observation": "<데이터가 보여주는 바를 숫자와 함께 한국어로>",
     "action": "<다음 번에 연습할 구체적인 행동 1가지를 한국어로>"}
  ]
}

참고 — area 값은 내부 키이며, 사용자에게는 다음 한국어 축 이름으로 표시됩니다:
- vision   → 답변 태도 (시선·표정·자세)
- audio    → 전달력 (속도·침묵·떨림)
- language → 사고력 (논리·구조·내용)

규칙:
1. ``priorities``는 영향력이 큰 순서로 정확히 3개여야 합니다.
2. ``per_question``은 입력의 각 질문마다 정확히 하나씩, 같은 순서로, 동일한 \
``index`` 값으로 포함해야 합니다. 건너뛰거나 중단된 질문은 짧게 언급하세요.
3. ``area`` 값(vision/audio/language)을 제외한 모든 텍스트는 한국어로 작성하세요.
4. JSON 외의 텍스트는 출력하지 마세요. 마크다운 코드펜스도 쓰지 마세요.
"""


def _build_user_prompt(analysis: dict[str, Any]) -> str:
    aggregate = analysis.get("aggregate") or {}
    parts: list[str] = [
        f"Session: {analysis.get('session_id', '')}",
        f"Company: {analysis.get('company', '')}",
        "",
        "# Aggregate metrics",
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        "",
        "# Per-question metrics + transcripts",
    ]
    for q in analysis.get("per_question") or []:
        parts.append("")
        parts.append(_render_question_block(q))

    parts.append("")
    parts.append(
        "Produce the JSON object now. Remember: exactly 3 priorities, one "
        "per_question entry per input question, no prose outside the JSON."
    )
    return "\n".join(parts)


def _render_question_block(q: dict[str, Any]) -> str:
    idx = q.get("index", "?")
    cat = q.get("category", "")
    head = f"## Q{idx} [{cat}] {q.get('question', '')}"
    if q.get("skipped") or q.get("aborted"):
        return head + f"\n  (status: {q.get('status', '?')})"

    audio = q.get("audio") or {}
    language = q.get("language") or {}
    vision = q.get("vision") or {}
    transcript = (language.get("transcript") or {}).get("text", "")[:600]

    return "\n".join([
        head,
        "audio: " + json.dumps({
            "speech_ratio": (audio.get("speech") or {}).get("speech_ratio"),
            "pause_count_3s": (audio.get("speech") or {}).get("pause_count_3s"),
            "hesitation_before_speech_s": (audio.get("speech") or {}).get("hesitation_before_speech_s"),
            "syllables_per_minute": (audio.get("rate") or {}).get("syllables_per_minute"),
            "rate_label": (audio.get("rate") or {}).get("rate_label"),
            "tremor_index": (audio.get("voice") or {}).get("tremor_index"),
        }, ensure_ascii=False),
        "vision: " + json.dumps({
            "looking_ratio": (vision.get("gaze") or {}).get("looking_ratio"),
            "gaze_away_events": (vision.get("gaze") or {}).get("gaze_away_events"),
            "positive_ratio": (vision.get("expression") or {}).get("positive_ratio"),
            "tense_ratio": (vision.get("expression") or {}).get("tense_ratio"),
            "yaw_std_deg": (vision.get("head") or {}).get("yaw_std_deg"),
            "head_changes_per_min": (vision.get("head") or {}).get("direction_changes_per_min"),
        }, ensure_ascii=False),
        "language: " + json.dumps({
            "fillers_total": (language.get("fillers") or {}).get("total"),
            "fillers_per_minute": (language.get("fillers") or {}).get("per_minute"),
            "filler_words": (language.get("fillers") or {}).get("counts"),
            "structure_score": (language.get("structure") or {}).get("score"),
            "structure_reason": (language.get("structure") or {}).get("reason"),
            "type_token_ratio": (language.get("repetition") or {}).get("type_token_ratio"),
            "top_repeats": (language.get("repetition") or {}).get("top"),
        }, ensure_ascii=False),
        f"transcript: {transcript}" if transcript else "transcript: (empty)",
    ])


# ---------------------------------------------------------------------------
# Call + retry + schema check
# ---------------------------------------------------------------------------
def _call_with_retry(*, client: Any, model: str, analysis: dict) -> dict:
    user_prompt = _build_user_prompt(analysis)
    expected_indices = [
        q.get("index") for q in (analysis.get("per_question") or [])
    ]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    last_error: str = ""
    for attempt in (1, 2):
        raw = _chat_complete_json(client, model, messages)
        try:
            parsed = json.loads(raw)
            _validate(parsed, expected_indices)
            return parsed
        except (json.JSONDecodeError, ReportGenerationError, ValueError, TypeError) as exc:
            last_error = str(exc)
            logger.warning("Report generation attempt %d failed: %s", attempt, exc)
            if attempt == 2:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was rejected with error: "
                        f"{last_error}. Return ONLY a JSON object matching the "
                        "schema in the system message. Exactly 3 priorities, "
                        "exactly one per_question entry per input question."
                    ),
                }
            )

    raise ReportGenerationError(
        f"Failed to obtain a valid report after 2 attempts: {last_error}"
    )


def _chat_complete_json(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


def _validate(parsed: Any, expected_indices: list[int]) -> None:
    if not isinstance(parsed, dict):
        raise ReportGenerationError("top-level must be an object")
    for key in ("overall_summary", "per_question", "priorities"):
        if key not in parsed:
            raise ReportGenerationError(f"missing field '{key}'")
    if not isinstance(parsed["overall_summary"], str) or not parsed["overall_summary"].strip():
        raise ReportGenerationError("overall_summary must be a non-empty string")

    pq = parsed["per_question"]
    if not isinstance(pq, list):
        raise ReportGenerationError("per_question must be an array")
    if [item.get("index") if isinstance(item, dict) else None for item in pq] != expected_indices:
        raise ReportGenerationError(
            "per_question indices must match input in count and order: "
            f"expected {expected_indices}"
        )
    for i, item in enumerate(pq):
        if not isinstance(item.get("comment"), str) or not item["comment"].strip():
            raise ReportGenerationError(f"per_question[{i}].comment must be a non-empty string")

    pri = parsed["priorities"]
    if not isinstance(pri, list) or len(pri) != PRIORITY_COUNT:
        raise ReportGenerationError(f"priorities must have exactly {PRIORITY_COUNT} items")
    for i, p in enumerate(pri):
        if not isinstance(p, dict):
            raise ReportGenerationError(f"priorities[{i}] must be an object")
        if p.get("area") not in ALLOWED_AREAS:
            raise ReportGenerationError(
                f"priorities[{i}].area must be one of {sorted(ALLOWED_AREAS)}"
            )
        for key in ("observation", "action"):
            if not isinstance(p.get(key), str) or not p[key].strip():
                raise ReportGenerationError(
                    f"priorities[{i}].{key} must be a non-empty string"
                )


# ---------------------------------------------------------------------------
# Fallback for empty / no-data analyses
# ---------------------------------------------------------------------------
def _empty_report(analysis: dict[str, Any]) -> dict[str, Any]:
    status = analysis.get("status", "no-data")
    return {
        "status": status,
        "session_id": analysis.get("session_id", ""),
        "company": analysis.get("company", ""),
        "overall_summary": (
            "이번 세션에는 분석할 수 있는 답변이 없어 피드백을 생성할 수 없어요. "
            "리포트를 보기 전에 면접을 끝까지 진행해 주세요."
        ),
        "per_question": [
            {"index": q.get("index", -1),
             "comment": "(녹화된 답변 없음)"}
            for q in analysis.get("per_question") or []
        ],
        "priorities": [],
    }


# ---------------------------------------------------------------------------
# Per-question personalized feedback (Issue 4)
# ---------------------------------------------------------------------------
_QUESTION_FEEDBACK_SYSTEM: Final[str] = """\
당신은 면접 강사입니다. 지원자의 답변 전문과 분석 지표를 바탕으로 아래 형식으로 \
피드백을 작성하세요. 반드시 한국어로 작성하세요.

[잘한 점] 이 답변에서 긍정적인 요소 1가지 — 구체적인 내용을 인용해서 칭찬할 것
[개선점] 가장 중요한 개선 포인트 1가지 — 행동 중심으로 작성할 것
[바로 적용] 다음 답변에서 즉시 실천할 수 있는 팁 1가지 — 한 문장으로

위 세 항목만, 각 항목을 줄바꿈으로 구분해 출력하세요. 다른 머리말이나 설명은 \
붙이지 마세요."""


def generate_question_feedback(
    q: dict[str, Any],
    metrics: dict[str, Any],
    transcript: str,
    *,
    client: Any | None = None,
    model: str = GPT_MODEL,
) -> dict[str, Any]:
    """Generate a personalized [잘한 점]/[개선점]/[바로 적용] block for one answer.

    Returns ``{"status": ..., "text": str}``. Never raises at the boundary:
      * ``empty``      — no transcript to comment on (no API call made)
      * ``no-api-key`` — no client and no key (no API call made)
      * ``error``      — the API call failed
      * ``ok``         — ``text`` holds the Korean feedback block
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {"status": "empty", "text": "답변이 녹화되지 않아 피드백을 생성할 수 없어요."}
    if client is None and not OPENAI_API_KEY:
        return {"status": "no-api-key", "text": ""}
    if client is None:
        client = _default_client()

    question_text = q.get("question", "") if isinstance(q, dict) else str(q)
    user_msg = "\n".join([
        f"[질문] {question_text}",
        "",
        "[분석 지표]",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "",
        "[답변 전문]",
        transcript,
        "",
        "위 형식에 맞춰 한국어로 피드백을 작성하세요.",
    ])

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _QUESTION_FEEDBACK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - surfaced as status, never raised
        logger.warning("generate_question_feedback failed: %s", exc)
        return {"status": "error", "text": "", "error": str(exc)}

    if not text:
        return {"status": "error", "text": "", "error": "empty response"}
    return {"status": "ok", "text": text}


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------
def _default_client() -> Any:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and paste "
            "your key, or pass a `client=` argument."
        )
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


# Used by tests to recompute the cache key for a given analysis.
def _content_hash(analysis: dict[str, Any]) -> str:  # pragma: no cover - utility
    return hashlib.sha256(
        json.dumps(analysis, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
