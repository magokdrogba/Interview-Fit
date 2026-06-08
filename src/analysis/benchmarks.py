"""Benchmark table: turn a raw metric value into a Korean verdict.

Every metric shown to the user must carry three things:

    [측정값]  ·  [기준 범위]  ·  [🟢/🟡/🔴 판정]

plus a one-line piece of advice that depends on which band the value fell into.
This module centralizes those thresholds so the UI (and any future export) all
read from one source of truth.

Public API
----------
* :func:`evaluate` — ``evaluate(metric_id, value) -> MetricVerdict | None``.
  Returns ``None`` for missing/None values so callers can render a placeholder.
* :data:`AXES` — the three behavioral axes (Issue 1 labels), each with the
  ordered metric ids that belong to it. Used to lay out the report.
* :data:`METRIC_IDS` — the set of valid metric ids.

Axis labels (MECE, from the interviewer's perspective)
------------------------------------------------------
* 답변 태도 (시선·표정·자세)  — what the interviewer *sees*
* 전달력 (속도·침묵·떨림)      — *how* you speak
* 사고력 (논리·구조·내용)      — *what* you say
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Axis labels (Issue 1 canonical)
# ---------------------------------------------------------------------------
AXIS_ATTITUDE: str = "답변 태도 (시선·표정·자세)"
AXIS_DELIVERY: str = "전달력 (속도·침묵·떨림)"
AXIS_THINKING: str = "사고력 (논리·구조·내용)"

# (axis_label, emoji, [metric_id, ...]) in display order.
AXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (AXIS_ATTITUDE, "👁", ("gaze", "expression", "head_movement")),
    (AXIS_DELIVERY, "🎙", ("answer_volume", "long_pauses", "hesitation",
                           "speech_rate")),
    (AXIS_THINKING, "💬", ("fillers", "structure")),
)


@dataclass(frozen=True)
class MetricVerdict:
    """A fully resolved, display-ready judgment for one metric."""

    metric_id: str
    name: str          # display name, e.g. "카메라 응시 비율"
    value_text: str    # formatted measured value, e.g. "72%"
    range_text: str    # benchmark range, e.g. "권장 ≥ 70%"
    icon: str          # 🟢 / 🟡 / 🔴
    verdict: str       # short label, e.g. "안정적"
    advice: str        # one-line, band-dependent guidance

    def headline(self) -> str:
        """`항목명: 측정값 · 기준 범위 · 🟢 판정` — the one-line summary."""
        return f"{self.name}: {self.value_text} · {self.range_text} · {self.icon} {self.verdict}"


# Icons
_G, _Y, _R = "🟢", "🟡", "🔴"


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-metric classifiers. Each returns (icon, verdict, advice).
# ---------------------------------------------------------------------------
def _gaze(v: float):
    if v >= 0.70:
        return _G, "안정적", "면접에서 권장하는 시선 유지 수준입니다."
    if v >= 0.50:
        return _Y, "보통", "시선이 가끔 카메라를 벗어납니다."
    return _R, "개선 필요", "카메라를 바라보는 연습이 필요합니다."


def _expression_verdict(positive: float, tense: float):
    """Context-aware expression judgment (Issue 5).

    In a formal interview a neutral expression is professional and is NOT
    penalized. We only flag *genuinely tense* expressions; low smiling on its
    own is fine.
    """
    if positive >= 0.15:
        return _G, "밝은 표정", "밝고 자연스러운 표정입니다."
    if positive >= 0.05:
        return _G, "안정적", "안정적이고 차분한 인상입니다. 면접에서 중립적 표정은 자연스럽습니다."
    # positive < 5% — only a problem if the face also reads as tense.
    if tense > 0.20:
        return _Y, "다소 경직", "다소 경직된 표정이 보입니다. 자연스럽게 힘을 빼보세요."
    return _G, "차분함", "차분하고 집중된 표정입니다."


def _head_movement(v: float):
    if v <= 4:
        return _G, "안정적", "고개 움직임이 안정적입니다."
    if v <= 8:
        return _Y, "약간 과다", "답변 중 고개 움직임을 의식적으로 줄여보세요."
    return _R, "과다", "긴장감이 고개 움직임으로 드러나고 있습니다."


def _answer_volume(v: float):
    if v >= 0.60:
        return _G, "충분한 답변량", "답변량이 충분합니다."
    if v >= 0.40:
        return _Y, "답변이 다소 짧습니다", "구체적인 사례나 근거를 추가해보세요."
    return _R, "답변량 부족", "답변이 너무 짧습니다. 경험을 구체적으로 풀어내세요."


def _long_pauses(v: float):
    n = round(v)
    if n == 0:
        return _G, "자연스러운 흐름", "긴 침묵 없이 답변을 이어갔습니다."
    if n <= 2:
        return _Y, "가끔 막힘", "가끔 막히는 부분이 있습니다."
    return _R, "긴 공백이 잦음", "답변 전 핵심 키워드를 미리 떠올리는 연습을 하세요."


def _hesitation(v: float):
    if v <= 1.0:
        return _G, "빠른 답변", "바로 답변을 시작했습니다."
    if v <= 2.5:
        return _Y, "약간의 준비 시간", "두괄식으로 결론부터 말하는 연습을 권장합니다."
    return _R, "긴 침묵 후 시작", "면접관은 첫 3초에 집중합니다. 결론 한 줄을 먼저 말하세요."


def _speech_rate(v: float):
    if v < 200:
        return _R, "너무 느림", "조금 더 또렷하고 적극적으로 말해보세요."
    if v < 250:
        return _Y, "약간 느림", "조금 더 활기차게 말하면 전달력이 높아집니다."
    if v <= 330:
        return _G, "적정 속도", "면접에 적절한 말 속도입니다."
    if v <= 400:
        return _Y, "약간 빠름", "조금 더 천천히 말하면 전달력이 높아집니다."
    return _R, "너무 빠름", "면접관이 내용을 소화하기 어렵습니다. 의식적으로 속도를 줄이세요."


def _fillers(v: float):
    if v <= 1.0:
        return _G, "매우 적음", "군더더기 표현이 거의 없습니다."
    if v <= 3.0:
        return _Y, "적정 수준", "습관어가 적정 수준입니다."
    if v <= 5.0:
        return _Y, "다소 많음", '"음", "어", "그" 등을 의식적으로 줄여보세요.'
    return _R, "과다", "습관어가 많으면 답변의 신뢰도가 낮아집니다. 말 중간에 짧게 멈추는 연습을 하세요."


def _structure(v: float):
    if v >= 4:
        return _G, "체계적", "결론이 먼저 나오고 근거가 명확합니다."
    if v >= 3:
        return _Y, "보통", "구조는 있으나 결론이 다소 늦게 나옵니다."
    return _R, "비구조적", "결론 없이 나열식으로 답변하고 있습니다. 두괄식 구조를 연습하세요."


# ---------------------------------------------------------------------------
# Value formatters
# ---------------------------------------------------------------------------
def _fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _fmt_per_min(v: float) -> str:
    return f"{v:.1f}회/분"


def _fmt_count(v: float) -> str:
    return f"{round(v)}회"


def _fmt_seconds(v: float) -> str:
    return f"{v:.1f}초"


def _fmt_spm(v: float) -> str:
    return f"분당 {round(v)}음절"


def _fmt_fillers(v: float) -> str:
    return f"{v:.1f}개/분"


def _fmt_score(v: float) -> str:
    return f"{int(v)}점" if float(v).is_integer() else f"{v:.1f}점"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# metric_id -> (display_name, range_text, value_formatter, classifier)
_REGISTRY: dict[str, tuple] = {
    "gaze": ("카메라 응시 비율", "권장 ≥ 70%", _fmt_pct, _gaze),
    # expression is special-cased in evaluate(): it needs BOTH positive and
    # tense ratios, so its classifier is None and handled separately.
    "expression": ("표정", "중립 표정도 정상 · 긴장 ≤ 20% 권장", None, None),
    "head_movement": ("고개 움직임", "권장 0-4회/분", _fmt_per_min, _head_movement),
    "answer_volume": ("답변량 (실제 발화 시간 비율)", "권장 ≥ 60%", _fmt_pct, _answer_volume),
    "long_pauses": ("3초 이상 침묵 횟수", "권장 0회", _fmt_count, _long_pauses),
    "hesitation": ("답변 시작 전 머뭇거림", "권장 ≤ 1.0초", _fmt_seconds, _hesitation),
    "speech_rate": ("말 속도 (분당 음절 수)", "권장 250-330 SPM", _fmt_spm, _speech_rate),
    "fillers": ("습관어 (군더더기 표현)", "권장 ≤ 1.0개/분", _fmt_fillers, _fillers),
    "structure": ("답변 구조 점수", "권장 4점 이상", _fmt_score, _structure),
}

METRIC_IDS: frozenset[str] = frozenset(_REGISTRY)


def name_of(metric_id: str) -> str | None:
    """Return the display name for a metric id, or None if unknown."""
    entry = _REGISTRY.get(metric_id)
    return entry[0] if entry else None


def _evaluate_expression(value, name: str, range_text: str) -> MetricVerdict | None:
    """Expression verdict from a {'positive': x, 'tense': y} dict (or a bare
    positive ratio, in which case tense is treated as 0)."""
    if isinstance(value, dict):
        positive = _num(value.get("positive"))
        tense = _num(value.get("tense")) or 0.0
    else:
        positive = _num(value)
        tense = 0.0
    if positive is None:
        return None
    icon, verdict, advice = _expression_verdict(positive, tense)
    return MetricVerdict(
        metric_id="expression",
        name=name,
        value_text=f"긍정 {positive * 100:.0f}% / 긴장 {tense * 100:.0f}%",
        range_text=range_text,
        icon=icon,
        verdict=verdict,
        advice=advice,
    )


def evaluate(metric_id: str, value) -> MetricVerdict | None:
    """Resolve ``value`` for ``metric_id`` into a display-ready verdict.

    Returns ``None`` if the metric id is unknown or the value is missing /
    non-numeric, so the caller can render a "측정 불가" placeholder.
    """
    entry = _REGISTRY.get(metric_id)
    if entry is None:
        return None
    name, range_text, fmt, classify = entry
    if metric_id == "expression":
        return _evaluate_expression(value, name, range_text)
    v = _num(value)
    if v is None:
        return None
    icon, verdict, advice = classify(v)
    return MetricVerdict(
        metric_id=metric_id,
        name=name,
        value_text=fmt(v),
        range_text=range_text,
        icon=icon,
        verdict=verdict,
        advice=advice,
    )
