"""Supabase client for the community feature.

``get_client()`` returns a configured Supabase client, or ``None`` when the
backend isn't set up (no URL/key, or the ``supabase`` package isn't installed).
Callers MUST handle ``None`` and degrade gracefully — the community UI shows a
setup guide instead of crashing the app.

Credentials are read from Streamlit secrets (cloud) first, then the environment
/ local ``.env`` (mirrors how the OpenAI key is resolved in ``config.py``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _secret(name: str) -> str | None:
    """Read a secret from st.secrets (if available) then the environment."""
    try:
        import streamlit as st

        # st.secrets behaves like a mapping; .get avoids raising when absent.
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:  # noqa: BLE001 - no secrets file / not in a Streamlit run
        pass
    return os.getenv(name)


_CLIENT_CACHE_KEY = "_sb_client"


def get_client() -> Any | None:
    """Return a Supabase client, or ``None`` if it can't be configured.

    The client is cached in ``st.session_state`` and reused across reruns.
    This is important under auth-based RLS: ``sign_in_with_password`` attaches
    the auth token to *this* client instance, so subsequent table queries must
    reuse the same client to be treated as authenticated.

    Never raises: missing package or missing credentials both yield ``None``.
    """
    # Reuse a previously-built (and possibly authenticated) client.
    try:
        import streamlit as st

        cached = st.session_state.get(_CLIENT_CACHE_KEY)
        if cached is not None:
            return cached
    except Exception:  # noqa: BLE001 - not in a Streamlit run
        st = None  # type: ignore[assignment]

    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None

    try:
        from supabase import create_client
    except Exception as exc:  # noqa: BLE001 - package not installed
        logger.warning("supabase package unavailable: %s", exc)
        return None

    try:
        client = create_client(url, key)
    except Exception as exc:  # noqa: BLE001 - bad URL/key, network, etc.
        logger.warning("Failed to create Supabase client: %s", exc)
        return None

    if st is not None:
        try:
            st.session_state[_CLIENT_CACHE_KEY] = client
        except Exception:  # noqa: BLE001
            pass
    return client
