"""Community write form: share one interview experience.

``render_write_form(client)`` renders the form, validates input, inserts a row
(including the authenticated ``user_id`` and account nickname) into the
``interview_posts`` table, then unlocks the feed and reruns. All Supabase
errors are surfaced to the user without crashing.

Requires a logged-in user: ``st.session_state["user"]`` (set by auth_ui) and
``st.session_state["nickname"]`` (the account's display nickname).
"""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

# Fixed salt so the same nickname maps to the same anonymous id across sessions,
# without storing the raw nickname as the identifier. Not a security secret.
_AUTHOR_SALT = "ai-mock-interview::community::v1"

_ROUND_OPTIONS = ["1차", "2차", "임원면접", "최종면접", "인턴"]
_RESULT_OPTIONS = ["합격", "불합격", "결과대기중"]
_ATMOSPHERE_OPTIONS = ["편안함", "보통", "압박면접"]

_MIN_REVIEW_CHARS = 50
_MAX_NICKNAME_CHARS = 20


def _author_hash(nickname: str) -> str:
    digest = hashlib.sha256((nickname + _AUTHOR_SALT).encode("utf-8")).hexdigest()
    return digest[:12]


def render_write_form(client: Any) -> None:
    """Render the interview-experience write form for the logged-in user.

    On success, sets ``st.session_state["has_posted_cache"] = True`` to unlock
    the feed, then reruns. ``client`` is a configured Supabase client and the
    caller guarantees the user is logged in.
    """
    user = st.session_state.get("user")
    user_id = getattr(getattr(user, "user", None), "id", None)
    if user_id is None:
        st.error("로그인이 필요합니다. 다시 로그인해주세요.")
        return

    nickname = st.session_state.get("nickname", "익명")
    st.caption(f"작성자: @{nickname}")

    with st.form("community_write", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("지원 회사 *", placeholder="예: BCG, McKinsey")
        with col2:
            role = st.text_input("직무 *", placeholder="예: Research Analyst")

        col3, col4, col5 = st.columns(3)
        with col3:
            round_ = st.selectbox("면접 단계 *", _ROUND_OPTIONS)
        with col4:
            result = st.selectbox("결과 *", _RESULT_OPTIONS)
        with col5:
            atmosphere = st.selectbox("면접 분위기 *", _ATMOSPHERE_OPTIONS)

        questions = st.text_area(
            "받은 질문들 *", height=120,
            placeholder="한 줄에 하나씩 적어주세요.\n예) 본인의 가장 임팩트 있던 프로젝트는?\n왜 컨설팅인가요?",
        )
        review = st.text_area(
            "전체 후기 *", height=200,
            placeholder=f"면접 과정을 자세히 적어주세요. (최소 {_MIN_REVIEW_CHARS}자)",
        )
        tips = st.text_area("준비 팁 (선택)", height=100,
                            placeholder="후배들에게 도움이 될 준비 팁이 있다면 적어주세요.")

        submitted = st.form_submit_button("후기 등록하기")

    if not submitted:
        return

    # --- Validation --------------------------------------------------------
    errors: list[str] = []
    if not company.strip():
        errors.append("지원 회사를 입력해주세요.")
    if not role.strip():
        errors.append("직무를 입력해주세요.")
    question_lines = [q.strip() for q in questions.splitlines() if q.strip()]
    if not question_lines:
        errors.append("받은 질문을 최소 1개 이상 입력해주세요.")
    if len(review.strip()) < _MIN_REVIEW_CHARS:
        errors.append(f"전체 후기는 최소 {_MIN_REVIEW_CHARS}자 이상 작성해주세요. "
                      f"(현재 {len(review.strip())}자)")

    if errors:
        for msg in errors:
            st.error(msg)
        return

    # --- Insert ------------------------------------------------------------
    # Auto title, e.g. "BCG RA 인턴 면접 후기".
    title = f"{company.strip()} {role.strip()} {round_} 면접 후기"
    payload = {
        "title": title,
        "company": company.strip(),
        "role": role.strip(),
        "round": round_,
        "result": result,
        "atmosphere": atmosphere,
        "questions": "\n".join(question_lines),
        "review": review.strip(),
        "tips": tips.strip() or None,
        "user_id": user_id,
        "author_hash": _author_hash(nickname),
        "nickname": nickname,
    }

    try:
        client.table("interview_posts").insert(payload).execute()
    except Exception as exc:  # noqa: BLE001 - never crash the app on DB error
        st.error(f"후기 등록에 실패했어요. 잠시 후 다시 시도해주세요.\n\n({exc})")
        return

    st.success("후기가 등록됐어요! 이제 다른 분들의 후기를 볼 수 있습니다 🎉")
    st.session_state["has_posted_cache"] = True
    st.rerun()
