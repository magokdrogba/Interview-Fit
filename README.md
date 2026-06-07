# AI Mock Interview

A desktop program that takes a resume and a target company as input, generates
company-tailored interview questions and follow-ups, runs a camera-based mock
interview, and delivers a feedback report analyzing video, audio, and language
habits.

## Status

**Phases 0 – 6 complete.** End-to-end pipeline works: resume + company input →
public-source collection → tailored question generation → live OpenCV
interview session → 3-axis post-hoc analysis → GPT feedback report. 117
hermetic tests cover every module. The single thing you have to do live is
record the session itself.

## End-to-end flow

1. **Streamlit** — `streamlit run app.py`
   - 1) Upload resume + enter target company + paste job posting
   - 2) Click *Collect interview context* (Wikipedia + your URLs + manual notes)
   - 3) Click *Generate questions* (8–12 tailored + 2–3 follow-ups each)
   - 4) Click *Save and prep live interview*
2. **Terminal** — `python run_interview.py --pending`
   - SPACE start/stop · N skip · ESC abort. Per-Q `q{n}.mp4`/`q{n}.wav` and a
     `manifest.json` land under `data/recordings/<session-id>/`.
3. **Streamlit** — refresh the page, pick the session, click *Run analysis
   and generate report*. KPI dashboard + Top-3 priorities + per-question
   detail appear.

## Architecture decisions (confirmed)

- **UI**: Streamlit for input + report; live interview runs in a separate
  OpenCV window. Streamlit cannot host a real-time camera feed cleanly.
- **STT**: OpenAI Whisper API (`whisper-1`). Audio leaves the device only for
  transcription. Video never leaves the device.

## Setup

Requires Python 3.11+.

```bash
cd ~/ai-mock-interview

# 1. Create and activate a venv
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) Pin resolved versions back into requirements.txt
pip freeze > requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env and paste your OPENAI_API_KEY

# 5. Smoke test
python -m pytest tests/ -q
```

## 실행 (Run)

가장 간단한 방법: **`./start.sh` 더블클릭 또는 터미널에서 `./start.sh`**

- macOS Finder에서는 `start.command`를 더블클릭해도 앱이 실행됩니다.
- 두 스크립트 모두 가상환경을 활성화한 뒤 `streamlit run app.py`를 실행합니다.

```bash
./start.sh           # 또는: ./start.command (macOS 더블클릭)
```

온라인 배포(무료, Streamlit Community Cloud)는 [`DEPLOY.md`](DEPLOY.md)를 참고하세요.

## Layout

```
ai-mock-interview/
├── app.py                 # Streamlit entry point (input + report)
├── run_interview.py       # OpenCV-based live interview runner
├── config.py              # Paths, thresholds, model names
├── requirements.txt
├── .env.example
├── src/
│   ├── resume/parser.py
│   ├── crawler/{base,sources}.py
│   ├── question/generator.py
│   ├── interview/{session,overlay}.py
│   ├── analysis/{vision,audio,language}.py
│   └── report/feedback.py
├── data/
│   ├── recordings/        # per-session video + audio (gitignored)
│   └── cache/             # crawl + STT cache (gitignored)
└── tests/
```

## Roadmap

| Phase | Scope                                                      |
|------:|------------------------------------------------------------|
|   0   | Setup: folders, deps, config, stubs                        |
|   1   | Resume parsing + company/job input UI                      |
|   2   | Public-data collection + manual-input fallback             |
|   3   | GPT question + follow-up generation                        |
|   4   | Live webcam/mic interview session + per-question recording |
|   5   | Vision / audio / language post-hoc analysis                |
|   6   | Aggregated feedback report + Streamlit results tab         |

## Privacy

All recorded video and audio stay on the local machine. Only audio is
transmitted externally, and only to the Whisper API for transcription. Facial
imagery never leaves the device.

## Crawling policy

Public pages only, with robots.txt checked before every fetch and a polite
delay between requests. Login walls, CAPTCHAs, and anti-bot systems are never
bypassed; when a source is blocked, the user is shown a textarea to paste
content they collected manually.
