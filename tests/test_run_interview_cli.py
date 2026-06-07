"""Smoke tests for the ``run_interview`` CLI surface (no camera/mic touched)."""

from __future__ import annotations

import json

import pytest

import run_interview


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run_interview.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--pending" in out
    assert "--questions" in out


def test_cli_no_source_exits_with_message():
    with pytest.raises(SystemExit) as excinfo:
        run_interview.main([])
    # argparse + our explicit sys.exit both use a non-zero / message form.
    assert excinfo.value.code not in (0, None)


def test_cli_loads_questions_file_shape(tmp_path, monkeypatch):
    """Confirm _load_questions accepts both shapes (list and {questions:[...]})."""
    qs = [{"category": "domain", "question": "Q1", "rationale": "r",
           "followups": ["a", "b"]}]
    f = tmp_path / "qs.json"
    f.write_text(json.dumps(qs), encoding="utf-8")
    args = run_interview.argparse.Namespace(
        pending=False, questions=str(f), company="Acme",
    )
    company, loaded = run_interview._load_questions(args)
    assert company == "Acme"
    assert loaded == qs

    f.write_text(json.dumps({"company": "Toss", "questions": qs}), encoding="utf-8")
    args.company = ""
    company, loaded = run_interview._load_questions(args)
    assert company == "Toss"
    assert loaded == qs


def test_cli_pending_missing_file_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(run_interview, "PENDING_SESSION_PATH", tmp_path / "nope.json")
    args = run_interview.argparse.Namespace(
        pending=True, questions=None, company="",
    )
    with pytest.raises(SystemExit):
        run_interview._load_questions(args)
