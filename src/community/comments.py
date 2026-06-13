"""Comments for a community post (list + write form).

``render_comments(client, post_id, current_user)`` shows the comment list and,
for a logged-in user, a write form. All Supabase access is wrapped so a backend
error degrades to a friendly message instead of crashing the app.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.community.feed import _relative_time


def load_comments(client: Any, post_id: int) -> list[dict]:
    """Return comments for ``post_id`` oldest-first. Returns [] on error."""
    try:
        resp = (
            client.table("interview_comments")
            .select("*")
            .eq("post_id", post_id)
            .order("created_at", desc=False)
            .execute()
        )
        return list(resp.data or [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"댓글을 불러오지 못했어요. ({exc})")
        return []


def comment_count(client: Any, post_id: int) -> int:
    """Number of comments on a post (best-effort; 0 on error)."""
    return len(load_comments(client, post_id))


def _extract_user_id(current_user: Any) -> str | None:
    try:
        return current_user.user.id
    except AttributeError:
        try:
            return current_user.id
        except AttributeError:
            return None


def render_comments(client: Any, post_id: int, current_user: Any) -> None:
    """Render the comment list and (if logged in) the write form for a post."""
    comments = load_comments(client, post_id)

    st.divider()
    st.markdown(f"💬 **댓글 {len(comments)}개**")

    for c in comments:
        st.markdown(f"**@{c.get('nickname', '익명')}**  ·  {_relative_time(c.get('created_at'))}")
        st.write(c.get("content", ""))
        st.markdown("")  # small spacer

    user_id = _extract_user_id(current_user) if current_user is not None else None
    if user_id is None:
        st.caption("댓글을 작성하려면 로그인이 필요합니다.")
        return

    nickname = st.session_state.get("nickname", "익명")
    with st.form(f"comment_form_{post_id}", clear_on_submit=True):
        content = st.text_area("댓글을 입력하세요", height=80, key=f"comment_input_{post_id}")
        submitted = st.form_submit_button("댓글 등록")

    if not submitted:
        return
    if not content.strip():
        st.error("댓글 내용을 입력해주세요.")
        return

    try:
        client.table("interview_comments").insert({
            "post_id": post_id,
            "user_id": user_id,
            "nickname": nickname,
            "content": content.strip(),
        }).execute()
    except Exception as exc:  # noqa: BLE001
        st.error(f"댓글 등록에 실패했어요. 잠시 후 다시 시도해주세요.\n\n({exc})")
        return

    st.rerun()
