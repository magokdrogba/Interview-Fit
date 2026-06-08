"""Session-level analysis orchestrator.

Reads ``manifest.json`` from a Phase 4 session directory, runs the three
per-axis analyzers on every answered question, aggregates the results, and
writes a single ``analysis.json`` next to the manifest so Phase 6 can read
metrics without re-doing any work.

Caching strategy
----------------
* Whisper transcripts are cached inside :mod:`src.analysis.language` by file
  hash, so re-running the orchestrator never re-transcribes unchanged files.
* The orchestrator itself caches ``analysis.json``. Pass ``force_refresh=True``
  to ignore it.
* Per-question failures are isolated: a broken video doesn't stop the audio
  or language analysis from running, and a broken question doesn't stop the
  next one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from src.analysis.audio import analyze_audio
from src.analysis.language import analyze_language
from src.analysis.vision import analyze_video

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_session(
    session_dir: Path | str,
    *,
    openai_client: Any | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run all three analyses on every answered question and return a dict.

    Output schema::

        {
          "session_id": str, "company": str,
          "per_question": [
            {"index": int, "category": str, "question": str,
             "duration_s": float,
             "vision": {...}, "audio": {...}, "language": {...}}
          ],
          "aggregate": {
            "vision": {...}, "audio": {...}, "language": {...}
          },
          "notes": [str, ...]
        }
    """
    base = Path(session_dir)
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return {"status": "no-manifest", "session_dir": str(base)}

    out_path = base / "analysis.json"
    if not force_refresh and out_path.exists():
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            cached["_from_cache"] = True
            return cached
        except (OSError, json.JSONDecodeError):
            logger.warning("analysis.json cache unreadable; recomputing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    answers = manifest.get("answers") or []

    per_question: list[dict[str, Any]] = []
    notes: list[str] = []
    for ans in answers:
        per_question.append(
            _analyze_one(base, ans, openai_client=openai_client, notes=notes)
        )

    result: dict[str, Any] = {
        "status": "ok",
        "session_id": manifest.get("session_id", base.name),
        "company": manifest.get("company", ""),
        "per_question": per_question,
        "aggregate": _aggregate(per_question),
        "notes": notes,
    }
    try:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        logger.warning("Failed to write %s: %s", out_path, exc)
    return result


# ---------------------------------------------------------------------------
# Per-question
# ---------------------------------------------------------------------------
def _analyze_one(
    base: Path,
    answer: dict[str, Any],
    *,
    openai_client: Any | None,
    notes: list[str],
) -> dict[str, Any]:
    idx = answer.get("index", -1)
    skipped = bool(answer.get("skipped"))
    aborted = bool(answer.get("aborted"))
    question_text = answer.get("question", "")
    out: dict[str, Any] = {
        "index": idx,
        "category": answer.get("category", ""),
        "question": question_text,
        "duration_s": float(answer.get("duration_s") or 0.0),
        "skipped": skipped,
        "aborted": aborted,
    }
    if skipped or aborted:
        out["status"] = "skipped" if skipped else "aborted"
        return out

    video_rel = answer.get("video_path") or ""
    audio_rel = answer.get("audio_path") or ""
    video_path = (base / video_rel) if video_rel else None
    audio_path = (base / audio_rel) if audio_rel else None

    # --- language (run first so the transcript is available for audio SPM) ---
    if audio_path is not None:
        try:
            out["language"] = analyze_language(
                audio_path,
                question_text=question_text,
                client=openai_client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_language q%s failed: %s", idx, exc)
            out["language"] = {"status": "error", "error": str(exc)}
            notes.append(f"q{idx}: language analysis error: {exc}")
    else:
        out["language"] = {"status": "no-file"}

    transcript_text = ((out.get("language") or {}).get("transcript") or {}).get("text") or ""

    # --- audio (speech rate derived from the Korean syllable count) ---
    if audio_path is not None:
        try:
            out["audio"] = analyze_audio(audio_path, transcript=transcript_text)
        except Exception as exc:  # noqa: BLE001 - never bubble up
            logger.warning("analyze_audio q%s failed: %s", idx, exc)
            out["audio"] = {"status": "error", "error": str(exc)}
            notes.append(f"q{idx}: audio analysis error: {exc}")
    else:
        out["audio"] = {"status": "no-file"}

    # --- vision ---
    if video_path is not None:
        try:
            out["vision"] = analyze_video(video_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_video q%s failed: %s", idx, exc)
            out["vision"] = {"status": "error", "error": str(exc)}
            notes.append(f"q{idx}: vision analysis error: {exc}")
    else:
        out["vision"] = {"status": "no-file"}

    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _avg(values: list[float]) -> float | None:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return round(mean(cleaned), 4) if cleaned else None


def _sum_int(values: list[int]) -> int:
    return int(sum(v for v in values if isinstance(v, (int, float))))


def _aggregate(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [q for q in per_question if q.get("status") == "ok"]
    if not ok:
        return {"status": "no-data", "answered": 0}

    def collect(axis: str, *path: str) -> list[Any]:
        out: list[Any] = []
        for q in ok:
            cur: Any = q.get(axis) or {}
            for p in path:
                if not isinstance(cur, dict):
                    break
                cur = cur.get(p)
            if cur is not None:
                out.append(cur)
        return out

    return {
        "status": "ok",
        "answered": len(ok),
        "vision": {
            "looking_ratio_mean": _avg(collect("vision", "gaze", "looking_ratio")),
            "gaze_away_events_total": _sum_int(collect("vision", "gaze", "gaze_away_events")),
            "positive_ratio_mean": _avg(collect("vision", "expression", "positive_ratio")),
            "neutral_ratio_mean": _avg(collect("vision", "expression", "neutral_ratio")),
            "tense_ratio_mean": _avg(collect("vision", "expression", "tense_ratio")),
            "yaw_std_mean": _avg(collect("vision", "head", "yaw_std_deg")),
            "pitch_std_mean": _avg(collect("vision", "head", "pitch_std_deg")),
            "head_changes_per_min_mean": _avg(collect("vision", "head", "direction_changes_per_min")),
        },
        "audio": {
            "speech_ratio_mean": _avg(collect("audio", "speech", "speech_ratio")),
            "pause_count_3s_total": _sum_int(collect("audio", "speech", "pause_count_3s")),
            "hesitation_mean_s": _avg(collect("audio", "speech", "hesitation_before_speech_s")),
            "syllables_per_minute_mean": _avg(collect("audio", "rate", "syllables_per_minute")),
        },
        "language": {
            "fillers_total": _sum_int(collect("language", "fillers", "total")),
            "fillers_per_minute_mean": _avg(collect("language", "fillers", "per_minute")),
            "structure_score_mean": _avg(collect("language", "structure", "score")),
            "conclusion_first_ratio": _avg(
                [1.0 if bool(v) else 0.0 for v in collect("language", "structure", "conclusion_first")]
            ),
            "type_token_ratio_mean": _avg(collect("language", "repetition", "type_token_ratio")),
        },
    }
