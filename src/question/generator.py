"""GPT-driven interview-question generator.

Public API
----------
- :func:`generate_questions` returns a list of structured question dicts:
    ``[{"category", "question", "rationale", "followups": [str, ...]}]``

Design choices
--------------
* **JSON-only output.** We pass ``response_format={"type": "json_object"}`` so
  the model returns a single JSON object; we then read the ``"questions"``
  array out of it. We do not rely on Markdown or free-form text.
* **One retry on schema mismatch.** If the first response is unparseable or
  doesn't match the schema, we re-send the failure as a corrective user
  message and try once more. Two strikes and we raise.
* **Caching.** The full input (resume hash + company + job posting + context
  block + model name) is hashed; identical inputs reuse the cached JSON.
* **Dependency injection.** ``client`` is a constructor arg so tests can pass
  a fake. Real callers leave it as ``None`` and we lazy-construct one.

The model is configurable via the ``GPT_MODEL`` env var (default
``gpt-4o-mini`` from :mod:`config`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Final, Iterable

from config import CACHE_DIR, GPT_MODEL, OPENAI_API_KEY
from src.crawler.sources import InterviewContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
ALLOWED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"personality", "domain", "experience", "case"}
)
MIN_QUESTIONS: Final[int] = 8
MAX_QUESTIONS: Final[int] = 12
MIN_FOLLOWUPS: Final[int] = 2
MAX_FOLLOWUPS: Final[int] = 3


class QuestionGenerationError(RuntimeError):
    """Raised when we cannot get a schema-valid response after a retry."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_questions(
    resume_text: str,
    company: str,
    job_posting: str,
    interview_context: InterviewContext | dict | str | None = None,
    *,
    n_questions: int = 10,
    model: str = GPT_MODEL,
    client: Any | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Generate a tailored interview-question set.

    Parameters
    ----------
    resume_text:
        Plain-text resume (typically ``ParsedResume.raw_text``).
    company, job_posting:
        Free text.
    interview_context:
        Optional. Either an :class:`InterviewContext`, a plain dict (with the
        same fields), or a pre-rendered prompt block string. Anything falsy
        is skipped.
    n_questions:
        Target count, clamped to ``[MIN_QUESTIONS, MAX_QUESTIONS]``.
    model:
        Override the default model name.
    client:
        Optional OpenAI client. If ``None``, one is built from
        ``OPENAI_API_KEY`` lazily.
    force_refresh:
        Skip the on-disk cache and call the model again.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text must be non-empty")
    if not company or not company.strip():
        raise ValueError("company must be non-empty")

    n = max(MIN_QUESTIONS, min(MAX_QUESTIONS, n_questions))
    context_block = _render_context(interview_context)

    cache_key = _cache_key(resume_text, company, job_posting, context_block, model, n)
    cache_path = _cache_path(company)
    if not force_refresh and cache_path.exists():
        cached = _read_cache(cache_path, cache_key)
        if cached is not None:
            return cached

    if client is None:
        client = _default_client()

    questions = _call_with_retry(
        client=client,
        model=model,
        resume_text=resume_text,
        company=company,
        job_posting=job_posting,
        context_block=context_block,
        n_questions=n,
    )

    _write_cache(cache_path, cache_key, questions)
    return questions


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT: Final[str] = """\
You are a seasoned interviewer at a top-tier company. Your job is to design \
a tailored interview-question set for ONE specific candidate at ONE specific \
company. You will receive the candidate's resume, the target company, the job \
posting, and supplementary public context.

You must return a JSON object with this exact top-level shape:

{
  "questions": [
    {
      "category": "<one of: personality | domain | experience | case>",
      "question": "<a single concrete interview question>",
      "rationale": "<one or two sentences explaining why THIS question for \
THIS candidate — reference specific resume content (project name, employer, \
skill) when the category is 'experience'>",
      "followups": ["<probe 1>", "<probe 2>", "<probe 3 (optional)>"]
    }
  ]
}

Rules:
1. Provide exactly the number of questions requested by the user (between \
8 and 12).
2. Distribute across all four categories. Always include at least 2 \
'experience' questions that quote specific resume content.
3. Each question must have 2 or 3 followups. Followups are deepening probes \
that assume an initial answer was given (e.g. 'how did you measure that?', \
'what would you do differently next time?').
4. 'case' is for case/situational/role-play prompts (especially relevant for \
consulting, product, strategy roles). If the role is clearly engineering-only, \
you may use 'case' for system-design or scenario questions.
5. Write in the same language as the resume when possible. If the resume is \
Korean, write in Korean; if mixed, prefer Korean.
6. Output ONLY the JSON object. No markdown fences, no commentary.
"""


def _build_user_prompt(
    *,
    resume_text: str,
    company: str,
    job_posting: str,
    context_block: str,
    n_questions: int,
) -> str:
    parts: list[str] = [
        f"Target company: {company}",
        f"Number of questions to generate: {n_questions}",
        "",
        "# Resume (plain text)",
        resume_text.strip(),
    ]
    if job_posting.strip():
        parts += ["", "# Job posting", job_posting.strip()]
    if context_block.strip():
        parts += ["", "# Supplementary context", context_block.strip()]
    parts += [
        "",
        "Now produce the JSON object as specified. Do not include any text "
        "outside the JSON.",
    ]
    return "\n".join(parts)


def _render_context(ctx: InterviewContext | dict | str | None) -> str:
    if not ctx:
        return ""
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, InterviewContext):
        return ctx.to_prompt_block()
    if is_dataclass(ctx):
        ctx = asdict(ctx)  # type: ignore[arg-type]
    if isinstance(ctx, dict):
        # Best-effort: reconstruct a prompt block from a plain dict.
        return InterviewContext(**{k: ctx.get(k) for k in (
            "company", "job_posting", "company_summary", "public_snippets",
            "manual_notes", "sources", "generated_at",
        ) if k in ctx} | {"company": ctx.get("company", "")}).to_prompt_block()
    raise TypeError(f"Unsupported context type: {type(ctx).__name__}")


# ---------------------------------------------------------------------------
# Call + retry
# ---------------------------------------------------------------------------
def _call_with_retry(
    *,
    client: Any,
    model: str,
    resume_text: str,
    company: str,
    job_posting: str,
    context_block: str,
    n_questions: int,
) -> list[dict]:
    user_prompt = _build_user_prompt(
        resume_text=resume_text,
        company=company,
        job_posting=job_posting,
        context_block=context_block,
        n_questions=n_questions,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: str = ""
    for attempt in (1, 2):
        raw = _chat_complete_json(client, model, messages)
        try:
            parsed = json.loads(raw)
            questions = _coerce_questions(parsed, n_questions)
            _validate_questions(questions, n_questions)
            return questions
        except (json.JSONDecodeError, QuestionGenerationError, ValueError, TypeError) as exc:
            last_error = str(exc)
            logger.warning("Question generation attempt %d failed: %s", attempt, exc)
            if attempt == 2:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was rejected by the schema "
                        f"validator with error: {last_error}. "
                        "Return ONLY a valid JSON object matching the schema "
                        "in the system message. No prose, no markdown."
                    ),
                }
            )

    raise QuestionGenerationError(
        f"Failed to obtain a schema-valid response after 2 attempts: {last_error}"
    )


def _chat_complete_json(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    """Issue the chat completion and return the assistant message content."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def _coerce_questions(parsed: Any, n_questions: int) -> list[dict]:
    """Extract the questions array from a parsed JSON object.

    The model is instructed to return ``{"questions": [...]}`` but we also
    accept a bare list to be liberal in what we accept.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("questions", "items", "data"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
    raise QuestionGenerationError(
        f"Expected a JSON object with a 'questions' array (or a bare array); "
        f"got {type(parsed).__name__}"
    )


def _validate_questions(questions: list[dict], n_questions: int) -> None:
    if not isinstance(questions, list):
        raise QuestionGenerationError("'questions' must be an array")
    if not (MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS):
        raise QuestionGenerationError(
            f"Expected {MIN_QUESTIONS}..{MAX_QUESTIONS} questions, got {len(questions)}"
        )

    for i, q in enumerate(questions):
        prefix = f"questions[{i}]"
        if not isinstance(q, dict):
            raise QuestionGenerationError(f"{prefix} must be an object")
        for key in ("category", "question", "rationale", "followups"):
            if key not in q:
                raise QuestionGenerationError(f"{prefix} missing field '{key}'")
        if q["category"] not in ALLOWED_CATEGORIES:
            raise QuestionGenerationError(
                f"{prefix}.category={q['category']!r} not in {sorted(ALLOWED_CATEGORIES)}"
            )
        for txt_key in ("question", "rationale"):
            if not isinstance(q[txt_key], str) or not q[txt_key].strip():
                raise QuestionGenerationError(f"{prefix}.{txt_key} must be a non-empty string")
        fups = q["followups"]
        if not isinstance(fups, list):
            raise QuestionGenerationError(f"{prefix}.followups must be an array")
        if not (MIN_FOLLOWUPS <= len(fups) <= MAX_FOLLOWUPS):
            raise QuestionGenerationError(
                f"{prefix}.followups must have {MIN_FOLLOWUPS}..{MAX_FOLLOWUPS} items "
                f"(got {len(fups)})"
            )
        for j, f in enumerate(fups):
            if not isinstance(f, str) or not f.strip():
                raise QuestionGenerationError(
                    f"{prefix}.followups[{j}] must be a non-empty string"
                )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _slug(company: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", company.strip().lower()).strip("-")
    return s or "company"


def _cache_path(company: str) -> Path:
    return Path(CACHE_DIR) / f"questions_{_slug(company)}.json"


def _cache_key(
    resume_text: str,
    company: str,
    job_posting: str,
    context_block: str,
    model: str,
    n_questions: int,
) -> str:
    payload = json.dumps(
        {
            "r": resume_text,
            "c": company,
            "j": job_posting,
            "x": context_block,
            "m": model,
            "n": n_questions,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(path: Path, key: str) -> list[dict] | None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict) or body.get("_key") != key:
        return None
    qs = body.get("questions")
    if not isinstance(qs, list):
        return None
    return qs


def _write_cache(path: Path, key: str, questions: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"_key": key, "questions": questions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover
        logger.warning("Failed to write question cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------
def _default_client() -> Any:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and paste "
            "your key, or pass a `client=` argument."
        )
    from openai import OpenAI  # imported lazily so tests don't need the SDK loaded

    return OpenAI(api_key=OPENAI_API_KEY)
