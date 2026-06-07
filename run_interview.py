"""Live interview session runner.

Streamlit cannot host an OpenCV window cleanly, so the live capture phase
lives in this standalone CLI. Pair it with ``streamlit run app.py``:

    1. In Streamlit, build your question set and click
       "Save and start live interview". That writes
       ``data/cache/_pending_session.json``.
    2. In a terminal, run::

           python run_interview.py --pending

       The OpenCV window opens. SPACE starts/stops each answer, N skips,
       ESC aborts. Per-question ``q{n}.mp4`` / ``q{n}.wav`` plus
       ``manifest.json`` land under ``data/recordings/<session-id>/``.
    3. Return to Streamlit (Phase 5 + 6) to analyze the session.

Alternative entry points: ``--questions path/to/questions.json`` or pipe a
JSON list of questions to stdin via ``--questions -``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import PENDING_SESSION_PATH, RECORDINGS_DIR, VIDEO_FPS
from src.interview.session import run_session

logger = logging.getLogger(__name__)


def _load_questions(args: argparse.Namespace) -> tuple[str, list[dict]]:
    """Return (company, questions) from the chosen source."""
    if args.pending:
        if not PENDING_SESSION_PATH.exists():
            sys.exit(
                f"No pending session file at {PENDING_SESSION_PATH}. "
                "Click 'Save and start live interview' in Streamlit first."
            )
        payload = json.loads(PENDING_SESSION_PATH.read_text(encoding="utf-8"))
        return payload.get("company", ""), list(payload.get("questions", []))

    if args.questions:
        if args.questions == "-":
            payload = json.load(sys.stdin)
        else:
            payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "questions" in payload:
            return payload.get("company", ""), list(payload["questions"])
        if isinstance(payload, list):
            return args.company or "", payload
        sys.exit("--questions file must be a list or an object with a 'questions' key")

    sys.exit("Provide either --pending or --questions PATH.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live AI mock interview session (camera + mic)."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--pending",
        action="store_true",
        help="Read the question set Streamlit dropped at "
             f"{PENDING_SESSION_PATH.relative_to(Path.cwd()) if PENDING_SESSION_PATH.is_relative_to(Path.cwd()) else PENDING_SESSION_PATH}.",
    )
    src.add_argument(
        "--questions",
        metavar="PATH",
        help="JSON file with a 'questions' list, or '-' to read JSON from stdin.",
    )

    parser.add_argument("--company", default="", help="Company name (only used with --questions).")
    parser.add_argument("--camera", type=int, default=0, help="cv2 VideoCapture index.")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS, help="Target capture FPS.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RECORDINGS_DIR,
        help="Where to place the session directory.",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="sounddevice input device (index or substring). Defaults to system default.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip the live face-landmark overlay (saves a bit of CPU).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    company, questions = _load_questions(args)
    if not questions:
        sys.exit("Question list is empty; nothing to do.")

    audio_device: int | str | None = args.audio_device
    if audio_device is not None and audio_device.isdigit():
        audio_device = int(audio_device)

    print(f"Starting session for {company or '(no company)'}, {len(questions)} questions.")
    print("Controls:  SPACE = start/stop answer  ·  N = skip  ·  ESC = abort")
    result = run_session(
        questions,
        company=company,
        output_dir=args.output_dir,
        camera_index=args.camera,
        fps=args.fps,
        audio_device=audio_device,
        use_overlay=not args.no_overlay,
    )
    print(f"\nSession saved to: {result.session_dir}")
    print(f"manifest:        {result.session_dir / 'manifest.json'}")
    if result.aborted:
        print("(session was aborted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
