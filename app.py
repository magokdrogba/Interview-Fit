"""Streamlit entry point: resume + company input (Phase 1) and final report (Phase 6).

The live interview itself runs in a separate OpenCV window (``run_interview.py``)
because Streamlit cannot host a real-time camera feed cleanly.

Run with::

    streamlit run app.py

NOTE: All user-facing text is Korean (target users are Korean university
students). Code identifiers, keys, and log messages stay in English.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Detect Streamlit Community Cloud so we can swap the local OpenCV window for a
# browser-camera flow (Issue 6, Part C).
IS_CLOUD: bool = (
    os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit-sharing"
    or ("HOSTNAME" in os.environ and "streamlit" in os.environ.get("HOSTNAME", ""))
)

# streamlit-webrtc enables true in-browser video+audio recording. It's optional:
# if it (or its native deps) aren't installed we fall back to st.camera_input
# snapshots. We only *detect* availability here via find_spec — the actual
# import (which pulls in PyAV) is deferred into the cloud branch so local
# desktop runs never load PyAV's ffmpeg alongside OpenCV's (avoids a macOS
# duplicate-dylib clash).
import importlib.util

_WEBRTC_AVAILABLE = importlib.util.find_spec("streamlit_webrtc") is not None

# STUN server so WebRTC can traverse NAT when deployed on the cloud.
_RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

from config import OPENAI_API_KEY, PENDING_SESSION_PATH, RECORDINGS_DIR
from src.analysis import benchmarks as B
from src.analysis.orchestrator import analyze_session
from src.crawler.sources import InterviewContext, collect_interview_context
from src.question.generator import (
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    QuestionGenerationError,
    generate_questions,
)
from src.report.feedback import (
    ReportGenerationError,
    build_report,
    generate_question_feedback,
)
from src.resume.parser import ParsedResume, parse_resume

# ---------------------------------------------------------------------------
# Session-state contract used by later phases.
# ---------------------------------------------------------------------------
# Keys set here:
#   - "resume": ParsedResume
#   - "company": str
#   - "job_posting": str
# Later phases will read these and add their own keys
# (e.g. "interview_context", "questions", "session_dir", "metrics", "report").
SESSION_KEYS_PHASE_1 = ("resume", "company", "job_posting")
SESSION_KEYS_PHASE_2 = ("interview_context",)
SESSION_KEYS_PHASE_3 = ("questions",)
SESSION_KEYS_PHASE_4 = ("selected_session_dir",)
SESSION_KEYS_PHASE_6 = ("analysis", "report")


# Question-category badges (Korean labels, English keys come from the model).
_CATEGORY_BADGES: dict[str, str] = {
    "personality": "🧭 인성",
    "domain": "🛠️ 직무 지식",
    "experience": "📂 경험",
    "case": "🧩 케이스",
}

# Korean labels for the three behavioral axes (used in the priority callouts).
# These are the MECE, interviewer-perspective labels (see benchmarks.AXES):
# what the interviewer sees / how you speak / what you say.
_AREA_LABELS: dict[str, str] = {
    "vision": B.AXIS_ATTITUDE,    # 답변 태도 (시선·표정·자세)
    "audio": B.AXIS_DELIVERY,     # 전달력 (속도·침묵·떨림)
    "language": B.AXIS_THINKING,  # 사고력 (논리·구조·내용)
}

# Korean labels for speech-rate and answer-structure classifications.
_RATE_LABELS: dict[str, str] = {
    "fast": "빠름",
    "normal": "보통",
    "slow": "느림",
}
_STRUCTURE_LABELS: dict[str, str] = {
    "top-down": "두괄식",
    "STAR": "STAR",
    "narrative": "서술형",
    "unstructured": "비구조적",
}


# Friendly Korean labels for the canonical resume section keys.
_SECTION_LABELS: dict[str, str] = {
    "_preamble": "헤더 / 연락처",
    "summary": "자기소개",
    "experience": "경력",
    "projects": "프로젝트",
    "skills": "기술",
    "education": "학력",
    "certifications": "자격증",
    "awards": "수상",
    "publications": "출판 / 논문",
    "languages": "어학",
}


def _render_input_form() -> None:
    """Sidebar form that gathers resume + company + job posting."""
    st.sidebar.header("1. 입력")

    with st.sidebar.form("inputs"):
        uploaded = st.file_uploader(
            "이력서 (PDF 또는 .txt)",
            type=["pdf", "txt"],
            help="이력서는 로컬에서만 분석돼요. 어떤 서버에도 업로드되지 않아요.",
        )
        company = st.text_input(
            "지원 회사",
            value=st.session_state.get("company", ""),
            placeholder="예: 토스, 쿠팡, 맥킨지",
        )
        job_posting = st.text_area(
            "채용 공고 (전문 붙여넣기)",
            value=st.session_state.get("job_posting", ""),
            height=180,
            placeholder="채용 공고 내용을 여기에 붙여넣으세요. 질문 생성에 사용돼요.",
        )
        submitted = st.form_submit_button("이력서 분석하고 저장하기")

    if not submitted:
        return

    if uploaded is None:
        st.sidebar.error("먼저 이력서 파일을 첨부해 주세요.")
        return
    if not company.strip():
        st.sidebar.error("지원 회사를 입력해 주세요.")
        return

    # pdfplumber / our parser want bytes for PDFs; for .txt we can also pass
    # bytes (parser handles bytes only for PDFs, so decode txt ourselves).
    data = uploaded.read()
    if uploaded.name.lower().endswith(".pdf"):
        source: bytes | str = data
    else:
        source = data.decode("utf-8", errors="replace")

    try:
        parsed = parse_resume(source)
    except Exception as exc:  # pragma: no cover - surfaced to user
        st.sidebar.error(f"이력서 분석에 실패했어요: {exc}")
        return

    st.session_state["resume"] = parsed
    st.session_state["company"] = company.strip()
    st.session_state["job_posting"] = job_posting.strip()
    st.sidebar.success("저장했어요. 오른쪽에서 분석된 이력서를 확인하세요.")


def _render_parsed_resume(parsed: ParsedResume) -> None:
    st.subheader("분석된 이력서")
    st.caption(
        f"원본 형식: `{parsed.source_type}`  ·  "
        f"{len(parsed.raw_text):,}자  ·  "
        f"섹션 {len(parsed.sections)}개 감지됨"
    )

    if parsed.sections:
        for key, body in parsed.sections.items():
            label = _SECTION_LABELS.get(key, key.title())
            with st.expander(f"📄 {label}", expanded=(key == "experience")):
                st.text(body)
    else:
        st.info("섹션을 감지하지 못했어요. 원본 텍스트를 대신 보여드릴게요.")

    with st.expander("전체 정규화된 텍스트 보기", expanded=False):
        st.text(parsed.raw_text)


def _render_inputs_summary() -> None:
    st.subheader("저장된 입력")
    company = st.session_state.get("company", "")
    job_posting = st.session_state.get("job_posting", "")
    st.write(f"**지원 회사:** {company or '_미설정_'}")
    if job_posting:
        with st.expander("채용 공고 내용", expanded=False):
            st.text(job_posting)
    else:
        st.write("**채용 공고:** _미설정_")


def main() -> None:
    st.set_page_config(page_title="AI 모의면접", layout="wide")

    tab_interview, tab_community = st.tabs(["🏠 AI 모의면접", "💬 면접 후기"])
    with tab_interview:
        _render_interview_tab()
    with tab_community:
        _render_community_tab()


def _render_interview_tab() -> None:
    st.title("AI 모의면접")
    st.caption(
        "이력서와 지원 회사를 입력하면 맞춤형 면접 질문을 만들고, 실시간 모의면접을 "
        "녹화한 뒤, 답변 태도·전달력·사고력을 분석한 피드백 리포트를 받아볼 수 있어요."
    )

    _render_input_form()

    if "resume" not in st.session_state:
        st.info(
            "시작하려면 왼쪽 사이드바에서 이력서를 업로드하고 지원 회사를 입력하세요. "
            "분석된 내용이 여기에 표시돼요."
        )
        return

    parsed: ParsedResume = st.session_state["resume"]
    _render_inputs_summary()
    st.divider()
    _render_parsed_resume(parsed)
    st.divider()
    _render_context_panel()
    st.divider()
    _render_questions_panel()
    st.divider()
    _render_session_panel()
    st.divider()
    _render_report_panel()


def _render_community_tab() -> None:
    """면접 후기 커뮤니티 — Supabase Auth 로그인 + 게시물 기반 write-gate."""
    from src.community.auth_ui import (
        render_auth_page,
        render_logout_button,
        restore_session,
    )
    from src.community.db import get_client
    from src.community.feed import render_feed
    from src.community.gate import has_posted
    from src.community.write import render_write_form

    st.title("💬 면접 후기 커뮤니티")

    client = get_client()
    if client is None:
        st.warning("커뮤니티 기능을 사용하려면 Supabase 설정이 필요합니다.")
        st.info("`DEPLOY.md`의 'Supabase 설정' 섹션을 참고하세요.")
        return

    # Rehydrate an existing login (survives reruns / re-login).
    restore_session(client)
    user = st.session_state.get("user")

    if not user:
        render_auth_page(client)
        return

    # Extract the actual UUID string from the stored Supabase session/response.
    # The shape differs between sign_in responses and restored sessions, so we
    # try the common nestings before falling back.
    try:
        user_id = user.user.id
    except AttributeError:
        try:
            user_id = user.id
        except AttributeError:
            user_id = str(user)
    print(f"[DEBUG] user type: {type(user)}, user_id: {user_id}")

    if not user_id:
        # Corrupt/expired session object — force re-login.
        st.session_state.pop("user", None)
        render_auth_page(client)
        return

    # Logout button, top-right.
    _, col_logout = st.columns([5, 1])
    with col_logout:
        render_logout_button(client)
    st.caption(f"@{st.session_state.get('nickname', '익명')} 님으로 로그인됨")

    # Cache the has-posted check to avoid a DB hit on every rerun.
    if "has_posted_cache" not in st.session_state:
        st.session_state["has_posted_cache"] = has_posted(client, user_id)

    if not st.session_state["has_posted_cache"]:
        st.info(
            "다른 분들의 후기를 보려면 먼저 본인의 경험을 공유해주세요.\n\n"
            "작성 완료 후 전체 후기를 열람할 수 있습니다."
        )
        render_write_form(client)
    else:
        render_feed(client)
        with st.expander("✏️ 후기 추가 작성"):
            render_write_form(client)


# ---------------------------------------------------------------------------
# Phase 2 — Interview-context collection
# ---------------------------------------------------------------------------
def _render_context_panel() -> None:
    st.subheader("2. 면접 정보 수집")
    st.caption(
        "지원 회사 이름만으로 면접 후기·질문·기업문화·인재상 관련 공개 정보를 자동으로 "
        "수집해요(위키백과 + 웹 검색). 로그인이 필요한 사이트(잡플래닛, 캐치, 블라인드, "
        "링커리어)는 직접 찾은 후기를 아래 메모 칸에 붙여넣어 주세요. 이 앱은 해당 "
        "사이트를 크롤링하지 않아요."
    )

    company: str = st.session_state["company"]
    job_posting: str = st.session_state.get("job_posting", "")

    with st.form("context"):
        manual_notes = st.text_area(
            "직접 입력한 메모 (직접 찾은 면접 후기를 붙여넣으세요)",
            value=st.session_state.get("_context_manual", ""),
            height=160,
            placeholder=(
                "예: 잡플래닛·캐치·블라인드 후기에서 정리한 요점이나 커피챗 메모. "
                "질문에 반영하고 싶지만 앱이 직접 가져오기 어려운 내용을 적어주세요."
            ),
        )
        force = st.checkbox("강제 새로고침 (캐시 무시)", value=False)
        submitted = st.form_submit_button("면접 정보 수집하기")

    # Persist raw form input so re-renders keep the user's drafts.
    st.session_state["_context_manual"] = manual_notes

    if submitted:
        with st.spinner(f"{company} 면접 정보를 수집하는 중..."):
            ctx = collect_interview_context(
                company,
                job_posting=job_posting,
                manual_input=manual_notes,
                force_refresh=force,
            )
        st.session_state["interview_context"] = ctx

        # Count distinct sources: web/wiki snippets + the wikipedia summary.
        n_sources = len(ctx.public_snippets) + (1 if ctx.company_summary else 0)
        if n_sources == 0 and not ctx.manual_notes:
            st.warning(
                "공개 출처에서 수집된 내용이 없고 직접 입력한 메모도 없어요. 이 경우 "
                "질문 생성은 모델이 이미 알고 있는 회사 정보에만 의존하게 돼요."
            )
        else:
            st.success(f"총 {n_sources}개 출처에서 면접 관련 정보를 수집했습니다.")

    ctx_obj = st.session_state.get("interview_context")
    if isinstance(ctx_obj, InterviewContext):
        _render_context_preview(ctx_obj)


def _render_context_preview(ctx: InterviewContext) -> None:
    with st.expander("수집 출처", expanded=False):
        if ctx.sources:
            for s in ctx.sources:
                st.text(f"• {s}")
        else:
            st.text("(없음)")

    if ctx.company_summary:
        with st.expander("회사 요약 (위키백과)", expanded=True):
            st.write(ctx.company_summary)

    if ctx.public_snippets:
        for snip in ctx.public_snippets:
            with st.expander(f"공개 자료 — {snip['url']}", expanded=False):
                st.text(snip["text"])

    if ctx.manual_notes:
        with st.expander("직접 입력한 메모", expanded=False):
            st.text(ctx.manual_notes)


# ---------------------------------------------------------------------------
# Phase 3 — Question generation
# ---------------------------------------------------------------------------
def _render_questions_panel() -> None:
    st.subheader("3. 면접 질문")

    ctx = st.session_state.get("interview_context")
    if not isinstance(ctx, InterviewContext):
        st.info("먼저 위에서 면접 정보를 수집한 뒤, 여기에서 질문을 생성하세요.")
        return

    if not OPENAI_API_KEY:
        st.error(
            "`OPENAI_API_KEY`가 설정되지 않았어요.\n\n"
            "- 로컬 실행: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY`를 입력하세요.\n"
            "- 웹 배포: Streamlit Cloud → Manage app → Secrets에 키를 추가하세요."
        )
        return

    parsed: ParsedResume = st.session_state["resume"]

    with st.form("questions_widget"):
        n = st.slider(
            "질문 개수",
            min_value=MIN_QUESTIONS,
            max_value=MAX_QUESTIONS,
            value=st.session_state.get("_q_n", 10),
            step=1,
        )
        force = st.checkbox("강제 새로고침 (모델 재호출)", value=False)
        submitted = st.form_submit_button("질문 생성하기")

    st.session_state["_q_n"] = n

    if submitted:
        with st.spinner("모델을 호출하고 있어요…"):
            try:
                qs = generate_questions(
                    resume_text=parsed.raw_text,
                    company=st.session_state["company"],
                    job_posting=st.session_state.get("job_posting", ""),
                    interview_context=ctx,
                    n_questions=n,
                    force_refresh=force,
                )
            except QuestionGenerationError as exc:
                st.error(f"형식에 맞는 질문 세트를 생성하지 못했어요: {exc}")
                return
            except RuntimeError as exc:
                st.error(str(exc))
                return
        st.session_state["questions"] = qs
        st.success(f"질문 {len(qs)}개를 생성했어요.")

    questions = st.session_state.get("questions")
    if questions:
        _render_questions(questions)


def _render_questions(questions: list[dict]) -> None:
    # Category-distribution snapshot
    counts: dict[str, int] = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    badge_line = "  ·  ".join(
        f"{_CATEGORY_BADGES.get(cat, cat)}: {n}" for cat, n in counts.items()
    )
    st.caption(badge_line)

    for i, q in enumerate(questions, start=1):
        badge = _CATEGORY_BADGES.get(q["category"], q["category"])
        with st.expander(f"**Q{i}.** {badge} — {q['question']}", expanded=(i == 1)):
            st.markdown(f"*이 질문을 한 이유:* {q['rationale']}")
            st.markdown("**꼬리 질문:**")
            for f in q["followups"]:
                st.markdown(f"- {f}")


# ---------------------------------------------------------------------------
# Phase 4 — Live interview hand-off + session picker
# ---------------------------------------------------------------------------
def _render_local_session(questions: list[dict] | None) -> None:
    """Local desktop flow: hand off to the OpenCV window via run_interview.py."""
    st.caption(
        "실시간 녹화는 웹캠과 마이크를 쓰는 별도의 OpenCV 창에서 진행돼요 "
        "(데스크톱에서는 Streamlit이 실시간 카메라를 띄울 수 없어요). 여기서 질문을 "
        "저장한 뒤, 아래 명령어를 터미널에서 실행하세요. 면접이 끝나면 이 페이지로 "
        "돌아오세요."
    )
    if not questions:
        st.info("먼저 위에서 질문을 생성하세요.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("💾 저장하고 모의면접 준비하기"):
            payload = {
                "company": st.session_state.get("company", ""),
                "saved_at_session_keys": {
                    "company": st.session_state.get("company", ""),
                    "n_questions": len(questions),
                },
                "questions": questions,
            }
            PENDING_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
            PENDING_SESSION_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success(f"저장했어요: {PENDING_SESSION_PATH}")
    with col2:
        st.caption(f"대기 파일: `{PENDING_SESSION_PATH}`")

    st.markdown("**터미널에서 이 명령어를 실행하세요:**")
    st.code(
        "cd ~/ai-mock-interview\n"
        "source .venv/bin/activate\n"
        "python run_interview.py --pending",
        language="bash",
    )
    st.caption("OpenCV 창 조작법 — SPACE: 답변 시작/정지  ·  N: 건너뛰기  ·  ESC: 종료.")


def _render_cloud_session(questions: list[dict] | None) -> None:
    """Cloud flow dispatcher: WebRTC (video+audio) if available, else snapshots."""
    if _WEBRTC_AVAILABLE:
        _render_webrtc_session(questions)
    else:
        _render_camera_input_session(questions)


# ---------------------------------------------------------------------------
# Cloud flow A — true video+audio recording via streamlit-webrtc
# ---------------------------------------------------------------------------
def _render_webrtc_session(questions: list[dict] | None) -> None:
    """Record each answer (video + audio) in the browser using streamlit-webrtc.

    Each question gets its own ``webrtc_streamer``: press START to begin, answer,
    then press STOP — aiortc's MediaRecorder finalizes ``q{n}.mp4`` (with audio).
    On save we extract ``q{n}.wav`` from each clip so the full 3-axis analysis
    (vision + audio + language/STT) can run, exactly like the local flow.
    """
    # Deferred imports (pull in PyAV) — only on the cloud WebRTC path.
    from aiortc.contrib.media import MediaRecorder
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    st.info("웹 버전에서는 브라우저 카메라와 마이크로 답변을 녹화합니다. "
            "각 질문에서 START를 눌러 답변하고, 끝나면 STOP을 누르세요.")
    if not questions:
        st.info("먼저 위에서 질문을 생성하세요.")
        return

    # One session directory per recording attempt, persisted across reruns.
    session_dir = st.session_state.get("_webrtc_session_dir")
    if session_dir is None:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-web"
        session_dir = str(Path(RECORDINGS_DIR) / session_id)
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        st.session_state["_webrtc_session_dir"] = session_dir
    base = Path(session_dir)

    recorded = 0
    for i, q in enumerate(questions):
        out_video = base / f"q{i + 1}.mp4"
        with st.expander(f"Q{i + 1}. {q.get('question', '')}", expanded=(i == 0)):
            webrtc_streamer(
                key=f"webrtc_{base.name}_{i}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=_RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": True},
                in_recorder_factory=_make_recorder_factory(MediaRecorder, out_video),
            )
            if out_video.exists() and out_video.stat().st_size > 0:
                st.success("녹화 파일 저장됨 ✓")
                recorded += 1
            else:
                st.caption("START를 눌러 녹화를 시작하세요.")

    st.caption(f"녹화 완료: {recorded} / {len(questions)} 문항")
    if recorded and st.button("📼 녹화 종료 및 분석용으로 저장"):
        saved = _save_webrtc_session(base, questions)
        st.session_state["selected_session_dir"] = str(saved)
        # Allow a fresh directory on the next attempt.
        st.session_state.pop("_webrtc_session_dir", None)
        st.success(f"저장했어요: {saved.name} — 아래 5단계에서 분석할 수 있어요.")


def _make_recorder_factory(media_recorder_cls, out_path: Path):
    """Return a zero-arg factory that records the incoming stream to ``out_path``."""
    def factory():
        return media_recorder_cls(str(out_path))
    return factory


def _extract_wav_from_mp4(mp4_path: Path, wav_path: Path, sample_rate: int = 16_000) -> bool:
    """Extract mono 16-bit PCM audio from an MP4 into a WAV. Returns success."""
    import av  # PyAV (installed with streamlit-webrtc)

    try:
        inp = av.open(str(mp4_path))
    except Exception:  # noqa: BLE001
        return False
    if not any(s.type == "audio" for s in inp.streams):
        inp.close()
        return False

    try:
        out = av.open(str(wav_path), mode="w")
        ostream = out.add_stream("pcm_s16le", rate=sample_rate)
        try:
            ostream.layout = "mono"
        except Exception:  # noqa: BLE001 - some PyAV builds set this differently
            pass
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in inp.decode(audio=0):
            frame.pts = None
            for rframe in resampler.resample(frame):
                for packet in ostream.encode(rframe):
                    out.mux(packet)
        for packet in ostream.encode(None):
            out.mux(packet)
        out.close()
    except Exception:  # noqa: BLE001
        return False
    finally:
        inp.close()
    return wav_path.exists() and wav_path.stat().st_size > 0


def _save_webrtc_session(base: Path, questions: list[dict]) -> Path:
    """Build manifest from recorded q{n}.mp4 clips, extracting q{n}.wav per clip."""
    answers: list[dict] = []
    for i, q in enumerate(questions):
        mp4 = base / f"q{i + 1}.mp4"
        rec = {
            "index": i,
            "category": q.get("category", ""),
            "question": q.get("question", ""),
            "followups": list(q.get("followups", [])),
            "video_path": "",
            "audio_path": "",
            "answer_started_at": "",
            "answer_ended_at": "",
            "duration_s": 0.0,
            "aborted": False,
            "skipped": True,
            "note": "no recording",
        }
        if mp4.exists() and mp4.stat().st_size > 0:
            rec["video_path"] = mp4.name
            rec["skipped"] = False
            rec["note"] = ""
            wav = base / f"q{i + 1}.wav"
            if _extract_wav_from_mp4(mp4, wav):
                rec["audio_path"] = wav.name
            else:
                rec["note"] = "audio extraction failed"
        answers.append(rec)

    manifest = {
        "session_id": base.name,
        "company": st.session_state.get("company", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "aborted": False,
        "answers": answers,
        "questions": questions,
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return base


# ---------------------------------------------------------------------------
# Cloud flow B — st.camera_input snapshot fallback (no streamlit-webrtc)
# ---------------------------------------------------------------------------
def _render_camera_input_session(questions: list[dict] | None) -> None:
    """Fallback when streamlit-webrtc is unavailable: one snapshot per question.

    Snapshots are assembled into short MP4 clips so vision analysis can run.
    Audio/STT is unavailable here, so those axes are skipped downstream.
    """
    st.info("웹 버전에서는 브라우저 카메라를 사용합니다. (streamlit-webrtc 미설치 — "
            "스냅샷 모드이며 마이크 녹음/음성·언어 분석은 생략됩니다.)")
    if not questions:
        st.info("먼저 위에서 질문을 생성하세요.")
        return

    shots: dict = st.session_state.setdefault("_cloud_shots", {})
    for i, q in enumerate(questions):
        with st.expander(f"Q{i + 1}. {q.get('question', '')}", expanded=(i == 0)):
            img = st.camera_input("답변하는 모습을 촬영하세요", key=f"cam_{i}")
            if img is not None:
                shots[i] = img.getvalue()
                st.success("촬영 완료 ✓")
            elif i in shots:
                st.caption("이전에 촬영한 사진이 저장되어 있어요.")

    captured = sorted(shots)
    st.caption(f"촬영 완료: {len(captured)} / {len(questions)} 문항")

    if captured and st.button("📼 촬영 종료 및 분석용으로 저장"):
        session_dir = _save_cloud_session(questions, shots)
        if session_dir is not None:
            st.session_state["selected_session_dir"] = str(session_dir)
            st.success(f"저장했어요: {session_dir.name} — 아래 5단계에서 분석할 수 있어요.")


def _save_cloud_session(questions: list[dict], shots: dict) -> Path | None:
    """Write per-question MP4 clips + manifest from captured snapshots."""
    import cv2
    import numpy as np

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-web"
    base = Path(RECORDINGS_DIR) / session_id
    base.mkdir(parents=True, exist_ok=True)

    answers: list[dict] = []
    for i, q in enumerate(questions):
        rec = {
            "index": i,
            "category": q.get("category", ""),
            "question": q.get("question", ""),
            "followups": list(q.get("followups", [])),
            "video_path": "",
            "audio_path": "",      # no audio in the browser fallback
            "answer_started_at": "",
            "answer_ended_at": "",
            "duration_s": 0.0,
            "aborted": False,
            "skipped": i not in shots,
            "note": "" if i in shots else "no capture",
        }
        if i in shots:
            arr = np.frombuffer(shots[i], dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                h, w = frame.shape[:2]
                out_video = base / f"q{i + 1}.mp4"
                writer = cv2.VideoWriter(
                    str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (w, h)
                )
                # Repeat the snapshot for ~1s so vision sampling has frames.
                for _ in range(5):
                    writer.write(frame)
                writer.release()
                rec["video_path"] = out_video.name
                rec["duration_s"] = 1.0
            else:
                rec["skipped"] = True
                rec["note"] = "decode failed"
        answers.append(rec)

    manifest = {
        "session_id": session_id,
        "company": st.session_state.get("company", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "aborted": False,
        "answers": answers,
        "questions": questions,
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return base


def _render_session_panel() -> None:
    st.subheader("4. 실시간 모의면접")

    questions = st.session_state.get("questions")
    if IS_CLOUD:
        _render_cloud_session(questions)
    else:
        _render_local_session(questions)

    st.markdown("---")
    st.markdown("**이전에 녹화한 세션**")
    sessions = _list_sessions(RECORDINGS_DIR)
    if not sessions:
        st.caption("아직 녹화된 세션이 없어요.")
        return

    labels = [
        f"{p.name}  ·  {_session_caption(p)}"
        for p in sessions
    ]
    default_idx = 0
    selected_label = st.selectbox(
        "분석할 세션을 선택하세요 (아래 5단계에서 사용돼요)",
        options=labels,
        index=default_idx,
    )
    selected = sessions[labels.index(selected_label)]
    st.session_state["selected_session_dir"] = str(selected)
    st.caption(f"선택됨: `{selected}`")


def _list_sessions(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _session_caption(session_dir: Path) -> str:
    try:
        m = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "(매니페스트를 읽을 수 없음)"
    n_answers = sum(1 for a in m.get("answers", []) if a.get("video_path"))
    flag = " · 중단됨" if m.get("aborted") else ""
    return f"{m.get('company', '?')} · 답변 {n_answers}개{flag}"


# ---------------------------------------------------------------------------
# Phase 6 — Analysis & feedback report
# ---------------------------------------------------------------------------
_PRIORITY_RENDERERS = {
    0: ("🔴", st.error),    # top priority
    1: ("🟠", st.warning),  # second
    2: ("🟡", st.info),     # third
}


def _render_report_panel() -> None:
    st.subheader("5. 분석 및 피드백 리포트")
    selected = st.session_state.get("selected_session_dir")
    if not selected:
        st.info("위 4단계에서 세션을 녹화하거나 기존 세션을 선택하면 분석할 수 있어요.")
        return

    session_dir = Path(selected)
    if not (session_dir / "manifest.json").exists():
        st.error(
            f"{session_dir}에 매니페스트가 없어요. 질문에 답하기 전에 녹화가 "
            "중단되었나요?"
        )
        return

    if not OPENAI_API_KEY:
        st.error(
            "`OPENAI_API_KEY`가 설정되지 않았어요. STT 전사와 리포트 작성에 모두 필요해요.\n\n"
            "- 로컬 실행: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY`를 입력하세요.\n"
            "- 웹 배포: Streamlit Cloud → Manage app → Secrets에 키를 추가하세요."
        )
        return

    with st.form("run-analysis"):
        force = st.checkbox("강제 새로고침 (분석 + 리포트 재실행)", value=False)
        submitted = st.form_submit_button("분석 실행하고 리포트 생성하기")

    if submitted:
        with st.spinner("3축 분석을 실행하는 중이에요 (처음에는 몇 분 걸릴 수 있어요)…"):
            try:
                analysis = analyze_session(session_dir, force_refresh=force)
            except Exception as exc:  # noqa: BLE001
                st.error(f"분석에 실패했어요: {exc}")
                return
        st.session_state["analysis"] = analysis

        with st.spinner("피드백 리포트를 작성하는 중이에요…"):
            try:
                report = build_report(analysis, session_dir=session_dir, force_refresh=force)
            except ReportGenerationError as exc:
                st.error(f"형식에 맞는 리포트를 작성하지 못했어요: {exc}")
                return
            except RuntimeError as exc:
                st.error(str(exc))
                return
        st.session_state["report"] = report
        st.success("완료했어요.")

    analysis = st.session_state.get("analysis")
    report = st.session_state.get("report")
    if not analysis or not report:
        st.caption("위 버튼을 누르면 지표를 계산하고 피드백 리포트를 작성해요.")
        return

    _render_overall_summary(report)
    _render_priorities(report)
    _render_aggregate_kpis(analysis)
    _render_per_question_detail(analysis, report)


def _render_overall_summary(report: dict) -> None:
    st.markdown("### 종합 요약")
    st.write(report.get("overall_summary") or "_(요약 없음)_")


def _render_priorities(report: dict) -> None:
    st.markdown("### 핵심 개선 우선순위")
    priorities = report.get("priorities") or []
    if not priorities:
        st.caption("(없음 — 분석할 수 있는 답변이 없었어요)")
        return
    for i, p in enumerate(priorities):
        icon, renderer = _PRIORITY_RENDERERS.get(i, ("•", st.info))
        area = _AREA_LABELS.get(p.get("area", ""), p.get("area", "?"))
        renderer(
            f"{icon}  **{area}** — "
            f"{p.get('observation', '')}\n\n"
            f"**개선 방법:** {p.get('action', '')}"
        )


def _render_question_feedback(q: dict, transcript: str) -> None:
    """Per-question personalized GPT feedback (Issue 4), cached in session state.

    Shown directly above the metrics table. The GPT call is made once per
    (session, question) and cached in ``st.session_state`` to avoid redundant
    API calls on every Streamlit rerun.
    """
    if not transcript:
        return

    analysis = st.session_state.get("analysis") or {}
    session_id = analysis.get("session_id", "")
    cache: dict = st.session_state.setdefault("question_feedback_cache", {})
    cache_key = f"{session_id}::{q.get('index')}"

    fb = cache.get(cache_key)
    if fb is None:
        # Compact metrics summary for the prompt (benchmark-relevant fields).
        getter = _question_value_getter(q)
        metrics = {mid: getter(mid) for _, _, ids in B.AXES for mid in ids}
        with st.spinner("이 답변에 대한 맞춤 피드백을 작성하는 중이에요…"):
            fb = generate_question_feedback(q, metrics, transcript)
        cache[cache_key] = fb

    if fb.get("status") == "ok":
        st.markdown("**🎯 맞춤 피드백**")
        st.info(fb["text"])
    elif fb.get("status") == "no-api-key":
        st.caption("맞춤 피드백을 보려면 `OPENAI_API_KEY`가 필요해요.")
    elif fb.get("status") == "error":
        st.caption("맞춤 피드백 생성에 실패했어요. 잠시 후 다시 시도해 주세요.")


def _render_metric(metric_id: str, value) -> None:
    """Render one benchmarked metric: `항목명: 측정값 · 기준 범위 · 판정` + advice."""
    verdict = B.evaluate(metric_id, value)
    if verdict is None:
        name = B.name_of(metric_id) or metric_id
        st.markdown(f"**{name}**: 측정 불가")
        return
    st.markdown(verdict.headline())
    st.caption(f"└ {verdict.advice}")


def _render_axis_columns(value_getter) -> None:
    """Lay out all benchmarked metrics in three columns by behavioral axis."""
    cols = st.columns(3)
    for col, (axis_label, emoji, metric_ids) in zip(cols, B.AXES):
        with col:
            st.markdown(f"**{emoji} {axis_label}**")
            for mid in metric_ids:
                _render_metric(mid, value_getter(mid))


def _aggregate_value_getter(agg: dict):
    """Map aggregate metrics to benchmark metric ids."""
    v = agg.get("vision") or {}
    a = agg.get("audio") or {}
    l = agg.get("language") or {}
    answered = agg.get("answered") or 0

    # Long pauses are reported as a session total; convert to a per-answer
    # average so the per-answer benchmark thresholds apply.
    pause_total = a.get("pause_count_3s_total")
    pause_avg = (pause_total / answered) if (pause_total is not None and answered) else None

    mapping = {
        "gaze": v.get("looking_ratio_mean"),
        "expression": {"positive": v.get("positive_ratio_mean"),
                       "tense": v.get("tense_ratio_mean")},
        "head_movement": v.get("head_changes_per_min_mean"),
        "answer_volume": a.get("speech_ratio_mean"),
        "long_pauses": pause_avg,
        "hesitation": a.get("hesitation_mean_s"),
        "speech_rate": a.get("syllables_per_minute_mean"),
        "fillers": l.get("fillers_per_minute_mean"),
        "structure": l.get("structure_score_mean"),
    }
    return lambda mid: mapping.get(mid)


def _question_value_getter(q: dict):
    """Map one question's per-axis metrics to benchmark metric ids."""
    vision = q.get("vision") or {}
    audio = q.get("audio") or {}
    language = q.get("language") or {}
    gaze = vision.get("gaze") or {}
    expr = vision.get("expression") or {}
    head = vision.get("head") or {}
    speech = audio.get("speech") or {}
    rate = audio.get("rate") or {}
    fillers = language.get("fillers") or {}
    structure = language.get("structure") or {}

    mapping = {
        "gaze": gaze.get("looking_ratio"),
        "expression": {"positive": expr.get("positive_ratio"),
                       "tense": expr.get("tense_ratio")},
        "head_movement": head.get("direction_changes_per_min"),
        "answer_volume": speech.get("speech_ratio"),
        "long_pauses": speech.get("pause_count_3s"),
        "hesitation": speech.get("hesitation_before_speech_s"),
        "speech_rate": rate.get("syllables_per_minute"),
        "fillers": fillers.get("per_minute"),
        "structure": structure.get("score"),
    }
    return lambda mid: mapping.get(mid)


def _render_aggregate_kpis(analysis: dict) -> None:
    agg = analysis.get("aggregate") or {}
    if agg.get("status") != "ok":
        return
    st.markdown("### 종합 지표")
    st.caption("각 지표는 [측정값 · 기준 범위 · 🟢/🟡/🔴 판정] 순으로 표시돼요.")
    _render_axis_columns(_aggregate_value_getter(agg))


def _render_per_question_detail(analysis: dict, report: dict) -> None:
    st.markdown("### 질문별 상세")
    comments_by_idx = {
        c.get("index"): c.get("comment", "")
        for c in (report.get("per_question") or [])
    }
    for q in analysis.get("per_question") or []:
        idx = q.get("index", "?")
        cat = _CATEGORY_BADGES.get(q.get("category", ""), q.get("category", ""))
        head = f"Q{idx} · {cat} · {q.get('question', '')}"
        with st.expander(head, expanded=False):
            if q.get("status") != "ok":
                st.caption(f"상태: {q.get('status')}")
                comment = comments_by_idx.get(idx) or ""
                if comment:
                    st.write(comment)
                continue

            language = q.get("language") or {}
            transcript = ((language.get("transcript") or {}).get("text") or "").strip()

            # Personalized GPT feedback block (Issue 4) — directly above metrics.
            _render_question_feedback(q, transcript)

            # Short coach comment from the batch report (Issue 2: 핵심/개선 format).
            comment = comments_by_idx.get(idx) or ""
            if comment:
                st.markdown(f"**코치 코멘트**\n\n{comment}")

            if transcript:
                with st.expander("전사(STT) 결과", expanded=False):
                    st.write(transcript)

            st.markdown("**세부 지표**")
            _render_axis_columns(_question_value_getter(q))

            # Structure rubric reason (Issue 3) — explains the score.
            structure = language.get("structure") or {}
            if structure.get("status") == "ok" and structure.get("reason"):
                st.caption(f"답변 구조 채점 이유: {structure['reason']}")

            # Detected filler words ("말버릇") with their counts.
            _render_filler_words(language.get("fillers") or {})

            # Extra detail not covered by the benchmark table.
            repetition = language.get("repetition") or {}
            top_repeats = repetition.get("top") or []
            if top_repeats:
                rendered = ", ".join(f"{r['word']}×{r['count']}" for r in top_repeats)
                st.caption(f"자주 반복한 단어: {rendered}")


def _render_filler_words(fillers: dict) -> None:
    """Show detected fillers as chips, or a positive note if there were none."""
    counts = fillers.get("counts") or {}
    if not counts:
        st.markdown("✅ 말버릇이 감지되지 않았습니다")
        return
    chips = " ".join(f"`{word} ({n}회)`" for word, n in counts.items())
    st.markdown(f"감지된 말버릇: {chips}")


def _pct(x) -> str:
    if x is None:
        return "–"
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "–"


def _round(x, *, digits: int = 1) -> str:
    if x is None:
        return "–"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "–"


if __name__ == "__main__":
    main()
