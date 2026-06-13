"""Write-gate for the community feed (account-based).

A logged-in user must have shared at least one interview post before they can
read others'. Membership is determined by querying the database for posts
authored by the user; the result is cached in ``st.session_state`` by the
caller (see app.py) to avoid a DB round-trip on every rerun.
"""

from __future__ import annotations

from typing import Any


def has_posted(client: Any, user_id: str) -> bool:
    """Return True if ``user_id`` has at least one post. False on any error.

    Defensive: only a non-empty string id is a valid query argument, so a User
    object or None (a caller bug) safely yields False instead of a TypeError.
    """
    if not user_id or not isinstance(user_id, str):
        return False
    try:
        res = (
            client.table("interview_posts")
            .select("id")
            .eq("user_id", str(user_id))
            .limit(1)
            .execute()
        )
        return len(res.data or []) > 0
    except Exception as e:  # noqa: BLE001 - never crash on a DB hiccup
        print(f"[DEBUG] has_posted error: {e}")
        return False
