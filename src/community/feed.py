"""Community feed — Naver-cafe-style list + post-detail views.

``render_feed(client, current_user=None)`` switches between a list view and a
detail view using ``st.session_state["selected_post_id"]`` (no page reload).
All Supabase access is wrapped so a backend error degrades to a friendly
message instead of crashing the app.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import streamlit as st

_PAGE_SIZE = 15
_CANDIDATE_CAP = 300  # max rows pulled before client-side sort/paginate
_RESULT_FILTERS = ["전체", "합격", "불합격", "결과대기중"]
_SORT_OPTIONS = ["최신순", "좋아요순", "댓글많은순"]
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


def _post_title(post: dict) -> str:
    """Post title, with a fallback for rows created before the title column."""
    title = (post.get("title") or "").strip()
    if title:
        return title
    return (f"{post.get('company', '')} {post.get('role', '')} "
            f"{post.get('round', '')} 면접 후기").strip()


def _fetch_candidates(client: Any, *, company: str, result: str) -> list[dict]:
    """Fetch filtered posts (newest first), capped. Returns [] on error."""
    try:
        q = client.table("interview_posts").select("*")
        if company.strip():
            q = q.ilike("company", f"%{company.strip()}%")
        if result != "전체":
            q = q.eq("result", result)
        q = q.order("created_at", desc=True).limit(_CANDIDATE_CAP)
        return list(q.execute().data or [])
    except Exception as exc:  # noqa: BLE001
        st.error(f"후기를 불러오지 못했어요. 잠시 후 다시 시도해주세요.\n\n({exc})")
        return []


def _comment_counts(client: Any, post_ids: list[int]) -> dict[int, int]:
    """Map post_id -> comment count for the given ids. {} on error."""
    if not post_ids:
        return {}
    try:
        resp = (
            client.table("interview_comments")
            .select("post_id")
            .in_("post_id", post_ids)
            .execute()
        )
        return dict(Counter(row["post_id"] for row in (resp.data or [])))
    except Exception:  # noqa: BLE001 - comments are optional; degrade silently
        return {}


def _fetch_post(client: Any, post_id: int) -> dict | None:
    try:
        resp = client.table("interview_posts").select("*").eq("id", post_id).limit(1).execute()
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception as exc:  # noqa: BLE001
        st.error(f"후기를 불러오지 못했어요. ({exc})")
        return None


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


def _bump_views(client: Any, post: dict) -> int:
    """Increment a post's view count once per session; return the new count."""
    viewed: set = st.session_state.setdefault("viewed_posts", set())
    post_id = post.get("id")
    current = int(post.get("views") or 0)
    if post_id in viewed:
        return current
    viewed.add(post_id)
    new_count = current + 1
    try:
        client.table("interview_posts").update({"views": new_count}).eq("id", post_id).execute()
    except Exception:  # noqa: BLE001 - view count is best-effort
        return current
    return new_count


# ---------------------------------------------------------------------------
# List view
# ---------------------------------------------------------------------------
def _render_row(post: dict, *, comments: int) -> None:
    post_id = post.get("id")
    emoji = _RESULT_EMOJI.get(post.get("result", ""), "•")
    likes = int(post.get("likes") or 0)

    with st.container(border=True):
        if st.button(_post_title(post), key=f"open_{post_id}", use_container_width=True):
            st.session_state["selected_post_id"] = post_id
            st.rerun()
        st.caption(
            f"{emoji} {post.get('result', '')}  ·  {post.get('atmosphere', '')}  ·  "
            f"@{post.get('nickname', '익명')}  ·  {_relative_time(post.get('created_at'))}  ·  "
            f"💬 {comments}  ·  👍 {likes}"
        )


def _render_list(client: Any) -> None:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        company = st.text_input("회사 검색", key="feed_company", placeholder="회사명으로 검색")
    with col2:
        result = st.selectbox("결과 필터", _RESULT_FILTERS, key="feed_result")
    with col3:
        sort = st.selectbox("정렬", _SORT_OPTIONS, key="feed_sort")

    # Reset pagination whenever a filter/sort changes.
    filter_sig = (company.strip(), result, sort)
    if st.session_state.get("_feed_sig") != filter_sig:
        st.session_state["_feed_sig"] = filter_sig
        st.session_state["_feed_limit"] = _PAGE_SIZE
    limit = st.session_state.get("_feed_limit", _PAGE_SIZE)

    candidates = _fetch_candidates(client, company=company, result=result)
    if not candidates:
        st.info("아직 등록된 후기가 없어요. 검색 조건을 바꾸거나 첫 후기를 남겨보세요!")
        return

    counts = _comment_counts(client, [p["id"] for p in candidates if p.get("id") is not None])

    if sort == "좋아요순":
        candidates.sort(key=lambda p: (int(p.get("likes") or 0), p.get("created_at") or ""),
                        reverse=True)
    elif sort == "댓글많은순":
        candidates.sort(key=lambda p: (counts.get(p.get("id"), 0), p.get("created_at") or ""),
                        reverse=True)
    # 최신순 is already created_at desc from the DB.

    visible = candidates[:limit]
    has_more = len(candidates) > limit

    for post in visible:
        _render_row(post, comments=counts.get(post.get("id"), 0))

    st.caption(f"총 {len(candidates)}개의 후기가 있어요.")
    if has_more and st.button("더 보기", key="feed_more"):
        st.session_state["_feed_limit"] = limit + _PAGE_SIZE
        st.rerun()


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------
def _render_detail(client: Any, post_id: int, current_user: Any) -> None:
    if st.button("← 목록으로", key="back_to_list"):
        st.session_state["selected_post_id"] = None
        st.rerun()

    post = _fetch_post(client, post_id)
    if post is None:
        st.warning("후기를 찾을 수 없어요. 목록으로 돌아가 주세요.")
        return

    views = _bump_views(client, post)

    # Header
    st.header(_post_title(post))
    st.caption(
        f"@{post.get('nickname', '익명')}  ·  {_relative_time(post.get('created_at'))}  ·  "
        f"조회 {views}"
    )

    # Meta line
    emoji = _RESULT_EMOJI.get(post.get("result", ""), "•")
    st.markdown(
        f"🏢 **{post.get('company', '')}**  |  직무: {post.get('role', '')}  |  "
        f"면접단계: {post.get('round', '')}  |  결과: {emoji} {post.get('result', '')}  |  "
        f"분위기: {post.get('atmosphere', '')}"
    )
    st.divider()

    # Questions
    q_lines = [q.strip() for q in (post.get("questions") or "").splitlines() if q.strip()]
    if q_lines:
        st.markdown("#### 📋 받은 질문")
        for i, q in enumerate(q_lines, start=1):
            st.markdown(f"**Q{i}.** {q}")

    # Review
    st.markdown("#### 📝 면접 후기")
    st.write(post.get("review", ""))

    # Tips
    if post.get("tips"):
        st.markdown("#### 💡 준비 팁")
        st.write(post["tips"])

    # Like
    liked: set = st.session_state.setdefault("liked_posts", set())
    likes = int(post.get("likes") or 0)
    already = post_id in liked
    st.button(f"👍 {likes}" + ("  (좋아요 완료)" if already else ""),
              key=f"like_{post_id}", disabled=already,
              on_click=_toggle_like, args=(client, post))

    # Comments
    from src.community.comments import render_comments

    render_comments(client, post_id, current_user)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_feed(client: Any, current_user: Any | None = None) -> None:
    """Render the community feed: list view, or detail view for a selected post."""
    selected = st.session_state.get("selected_post_id")
    if selected:
        _render_detail(client, selected, current_user)
    else:
        _render_list(client)
