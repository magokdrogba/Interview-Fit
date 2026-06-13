"""Write-gate for the community feed.

Users must share one interview experience before they can read others'
(잡플래닛 / 블라인드 style). The gate is per browser session, tracked in
``st.session_state``.
"""

from __future__ import annotations

import streamlit as st

_POSTED_KEY = "community_posted"


def has_posted() -> bool:
    """True if this browser session has already submitted a post."""
    return bool(st.session_state.get(_POSTED_KEY))


def mark_posted() -> None:
    """Record that the user has shared a post, unlocking the feed."""
    st.session_state[_POSTED_KEY] = True
