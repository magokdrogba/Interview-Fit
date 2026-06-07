"""Central configuration: paths, thresholds, model name constants.

Loaded by every module that needs a path or a tunable. Do NOT put secrets here -
those live in `.env` and are read via `os.getenv`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
CACHE_DIR: Path = DATA_DIR / "cache"

for _d in (RECORDINGS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Secrets / API
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")

# ---------------------------------------------------------------------------
# Capture / analysis tunables
# ---------------------------------------------------------------------------
VIDEO_FPS: int = 30
ANALYSIS_SAMPLE_FPS: int = 5  # sub-sample rate for post-hoc vision analysis
AUDIO_SAMPLE_RATE: int = 16_000  # Hz; matches Whisper expected input
AUDIO_CHANNELS: int = 1

# VAD / silence thresholds
VAD_AGGRESSIVENESS: int = 2  # 0..3, higher = more aggressive silence trimming
LONG_PAUSE_SECONDS: float = 3.0

# Crawling
HTTP_USER_AGENT: str = "ai-mock-interview/0.1 (+research; contact: local-user)"
HTTP_REQUEST_DELAY_SECONDS: float = 1.5

# Question generation
QUESTION_COUNT_RANGE: tuple[int, int] = (8, 12)
FOLLOWUPS_PER_QUESTION: int = 3

# MediaPipe Face Landmarker v2 model (used both for live overlay and Phase 5 analysis)
FACE_LANDMARKER_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH: Path = CACHE_DIR / "face_landmarker.task"

# Live interview pending-session file (Streamlit -> CLI handoff)
PENDING_SESSION_PATH: Path = CACHE_DIR / "_pending_session.json"
