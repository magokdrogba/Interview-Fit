"""Community feed: search, filter, sort, post cards, likes, pagination.

``render_feed(client)`` is shown only after the user has shared a post
(see :mod:`src.community.gate`). All Supabase access is wrapped so a backend
error degrades to a friendly message instead of crashing the app.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

_PAGE_SIZE = 10
_RESULT_FILTERS = ["전체", "합격", "불합격", "결과대기중"]
_SORT_OPTIONS = ["최신순", "좋아요순"]
_RESULT_EMOJI = {"합격": "✅", "불합격": "❌", "결과대기중": "⏳"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _relative_time(created_at: str | None) -> str:
    """Render an ISO timestamp as a Korean relative time (예: '3일 전')."""
    if not created_at:
        return ""
    raw = created_at.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        # Trim fractional seconds if fromisoformat can't parse them.
        try:
            dt = datetime.fromisoformat(raw.split(".")[0] + "+00:00")
        except ValueError:
            return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = datetime.now(timezone.utc) - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "방금 전"
    if secs < 3600:
        return f"{int(secs // 60)}분 전"
    if secs < 86_400:
        return f"{int(secs // 3600)}시간 전"
    if secs < 86_400 * 30:
        return f"{int(secs // 86_400)}일 전"
    return dt.strftime("%Y-%m-%d")


def _fetch_posts(client: Any, *, company: str, result: str, sort: str, limit: int) -> list[dict]:
    """Query interview_posts with the active filters. Returns [] on error."""
    try:
        q = client.table("interview_posts").select("*")
        if company.strip():
            q = q.ilike("company", f"%{company.strip()}%")
        if result != "전체":
            q = q.eq("result", result)
        if sort == "좋아요순":
            q = q.order("likes", desc=True).order("created_at", desc=True)
        else:
            q = q.order("created_at", desc=True)
        q = q.limit(limit)
        resp = q.execute()
        return list(resp.data or [])
    except Exception as exc:  # noqa: BLE001
        st.error(f"후기를 불러오지 못했어요. 잠시 후 다시 시도해주세요.\n\n({exc})")
        return []


def _toggle_like(client: Any, post: dict) -> None:
    """Increment a post's like count once per session."""
    liked: set = st.session_state.setdefault("liked_posts", set())
    post_id = post.get("id")
    if post_id in liked:
        return
    try:
        new_count = int(post.get("likes") or 0) + 1
        client.table("interview_posts").update({"likes": new_count}).eq("id", post_id).execute()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"좋아요를 반영하지 못했어요. ({exc})")
        return
    liked.add(post_id)
    st.rerun()


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------
def _render_card(client: Any, post: dict) -> None:
    liked: set = st.session_state.setdefault("liked_posts", set())
    post_id = post.get("id")
    emoji = _RESULT_EMOJI.get(post.get("result", ""), "•")

    with st.container(border=True):
        st.markdown(
            f"🏢 **{post.get('company', '')}**  ·  {post.get('role', '')}  ·  "
            f"{post.get('round', '')}"
        )
        st.caption(
            f"{emoji} {post.get('result', '')}  ·  분위기: {post.get('atmosphere', '')}  ·  "
            f"{_relative_time(post.get('created_at'))}"
        )

        # Question preview (each line as a Q.)
        q_lines = [q for q in (post.get("questions") or "").splitlines() if q.strip()]
        if q_lines:
            preview = "\n".join(f"- Q. {q.strip()}" for q in q_lines[:3])
            st.markdown(preview)
            if len(q_lines) > 3:
                st.caption(f"… 외 {len(q_lines) - 3}개 질문")

        with st.expander("후기 전체 보기 ▼"):
            all_qs = "\n".join(f"- Q. {q.strip()}" for q in q_lines)
            if all_qs:
                st.markdown("**받은 질문**")
                st.markdown(all_qs)
            st.markdown("**전체 후기**")
            st.write(post.get("review", ""))
            if post.get("tips"):
                st.markdown("**준비 팁**")
                st.write(post["tips"])
            st.caption(f"작성자: @{post.get('nickname', '익명')}")

        likes = int(post.get("likes") or 0)
        already = post_id in liked
        label = f"👍 {likes}" + ("  (좋아요 완료)" if already else "")
        st.button(label, key=f"like_{post_id}", disabled=already,
                  on_click=_toggle_like, args=(client, post))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_feed(client: Any) -> None:
    st.subheader("💬 면접 후기 커뮤니티")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        company = st.text_input("회사 검색", key="feed_company", placeholder="회사명으로 검색")
    with col2:
        result = st.selectbox("결과 필터", _RESULT_FILTERS, key="feed_result")
    with col3:
        sort = st.selectbox("정렬", _SORT_OPTIONS, key="feed_sort")

    # Pagination state — reset whenever a filter/sort changes.
    filter_sig = (company.strip(), result, sort)
    if st.session_state.get("_feed_sig") != filter_sig:
        st.session_state["_feed_sig"] = filter_sig
        st.session_state["_feed_limit"] = _PAGE_SIZE
    limit = st.session_state.get("_feed_limit", _PAGE_SIZE)

    # Fetch one extra row to detect whether a "더 보기" page exists.
    posts = _fetch_posts(client, company=company, result=result, sort=sort, limit=limit + 1)
    has_more = len(posts) > limit
    posts = posts[:limit]

    if not posts:
        st.info("아직 등록된 후기가 없어요. 검색 조건을 바꾸거나 첫 후기를 남겨보세요!")
        return

    for post in posts:
        _render_card(client, post)

    if has_more:
        if st.button("더 보기", key="feed_more"):
            st.session_state["_feed_limit"] = limit + _PAGE_SIZE
            st.rerun()
    else:
        st.caption(f"총 {len(posts)}개의 후기를 보고 있어요.")
