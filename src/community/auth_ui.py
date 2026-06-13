"""Supabase Auth UI for the community feature (email + password).

Public API
----------
* :func:`restore_session` — rehydrate a logged-in session on app start.
* :func:`render_auth_page` — login / signup tabs; returns True if logged in.
* :func:`render_logout_button` — sign out + clear session.
* :func:`current_user_id` — safely read the user id from the stored session.

All Supabase calls are wrapped so auth errors surface as friendly messages and
never crash the app.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def current_user_id(user: Any) -> str | None:
    """Best-effort extraction of the auth user id from a stored session/response."""
    try:
        return user.user.id
    except Exception:  # noqa: BLE001
        return None


def _nickname_from_user(user: Any) -> str:
    """Resolve a display nickname: user_metadata → pending signup value → email."""
    try:
        meta = getattr(user.user, "user_metadata", None) or {}
        nick = meta.get("nickname")
    except Exception:  # noqa: BLE001
        nick = None
    if not nick:
        nick = st.session_state.get("pending_nickname")
    if not nick:
        try:
            nick = (user.user.email or "").split("@")[0] or "익명"
        except Exception:  # noqa: BLE001
            nick = "익명"
    return nick


def _store_login(user: Any) -> None:
    st.session_state["user"] = user
    st.session_state["nickname"] = _nickname_from_user(user)
    # Force a fresh has-posted check for the newly logged-in account.
    st.session_state.pop("has_posted_cache", None)


def restore_session(client: Any) -> None:
    """Rehydrate an existing Supabase session into st.session_state on app start."""
    if "user" in st.session_state:
        return
    try:
        session = client.auth.get_session()
    except Exception:  # noqa: BLE001
        session = None
    if session is not None and current_user_id(session) is not None:
        _store_login(session)


def render_logout_button(client: Any) -> None:
    """Render a logout button; signs out and clears the local session."""
    if st.button("로그아웃", key="community_logout"):
        try:
            client.auth.sign_out()
        except Exception:  # noqa: BLE001
            pass
        for key in ("user", "nickname", "has_posted_cache", "pending_nickname",
                    "_sb_client"):
            st.session_state.pop(key, None)
        st.rerun()


def render_auth_page(client: Any) -> bool:
    """Render login / signup tabs. Returns True if the user is logged in."""
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    # --- Login -------------------------------------------------------------
    with tab_login:
        with st.form("community_login"):
            email = st.text_input("이메일", key="login_email")
            password = st.text_input("비밀번호", type="password", key="login_pw")
            submitted = st.form_submit_button("로그인")
        if submitted:
            try:
                res = client.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password}
                )
            except Exception:  # noqa: BLE001
                res = None
            if res is not None and current_user_id(res) is not None:
                _store_login(res)
                st.rerun()
            else:
                st.error("이메일 또는 비밀번호가 올바르지 않습니다")

    # --- Sign up -----------------------------------------------------------
    with tab_signup:
        with st.form("community_signup"):
            email = st.text_input("이메일", key="signup_email")
            password = st.text_input("비밀번호", type="password", key="signup_pw")
            password2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2")
            nickname = st.text_input("닉네임 (커뮤니티 표시용)", max_chars=20,
                                     key="signup_nickname")
            submitted = st.form_submit_button("회원가입")
        if submitted:
            if not email.strip() or not password:
                st.error("이메일과 비밀번호를 입력해주세요.")
            elif password != password2:
                st.error("비밀번호가 일치하지 않습니다.")
            elif not nickname.strip():
                st.error("닉네임을 입력해주세요.")
            else:
                try:
                    client.auth.sign_up({
                        "email": email.strip(),
                        "password": password,
                        "options": {"data": {"nickname": nickname.strip()}},
                    })
                except Exception as exc:  # noqa: BLE001
                    st.error(f"회원가입에 실패했어요. 다시 시도해주세요.\n\n({exc})")
                else:
                    st.session_state["pending_nickname"] = nickname.strip()
                    st.success("가입 완료! 이메일 인증 후 로그인해주세요")

    return "user" in st.session_state
