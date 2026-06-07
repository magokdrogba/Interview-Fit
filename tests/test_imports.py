"""Phase 0 smoke test: every module must import cleanly with zero side effects
beyond creating the data directories declared in config.py.
"""

from __future__ import annotations

import importlib

MODULES = [
    "config",
    "src.resume.parser",
    "src.crawler.base",
    "src.crawler.sources",
    "src.question.generator",
    "src.interview.session",
    "src.interview.overlay",
    "src.analysis.vision",
    "src.analysis.audio",
    "src.analysis.language",
    "src.report.feedback",
]


def test_all_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)


def test_config_paths_created() -> None:
    import config

    assert config.RECORDINGS_DIR.is_dir()
    assert config.CACHE_DIR.is_dir()
