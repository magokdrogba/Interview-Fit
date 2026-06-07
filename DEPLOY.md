# 온라인 배포 가이드 (Streamlit Community Cloud — 무료)

## 준비물
- GitHub 계정
- OpenAI API 키

## 단계

1. GitHub에 이 프로젝트를 push합니다
   주의: .env 파일은 절대 commit하지 마세요 (.gitignore 확인)

2. https://share.streamlit.io 접속 → GitHub으로 로그인

3. "New app" 클릭 → 레포지토리 선택 → Main file: app.py

4. "Advanced settings" → Secrets에 다음 입력:
   OPENAI_API_KEY = "sk-..."

5. Deploy 클릭 → 약 2-3분 후 공개 URL 생성

## 주의사항
- 웹 배포 시 카메라/마이크는 브라우저 권한 허용 필요
- 녹화 파일은 세션 종료 시 삭제됨 (로컬 저장 없음)
- run_interview.py (OpenCV 창)는 로컬에서만 동작
  → 웹 배포 버전에서는 클라우드 환경이 자동 감지되어 브라우저 녹화 모드로 전환됩니다.

## 웹 녹화 방식 (streamlit-webrtc)
- 클라우드에서는 `streamlit-webrtc`로 **영상 + 음성을 실제로 녹화**합니다.
  각 질문에서 START → 답변 → STOP을 누르면 `q{n}.mp4`(음성 포함)가 저장되고,
  저장 시 `q{n}.wav`를 자동 추출해 영상·음성·언어(STT) 3축 분석을 모두 수행합니다.
- WebRTC는 NAT 통과를 위해 STUN 서버가 필요합니다. 기본값으로 Google STUN
  (`stun:stun.l.google.com:19302`)을 사용합니다. 방화벽이 엄격한 환경에서는
  TURN 서버 설정이 추가로 필요할 수 있습니다.
- `streamlit-webrtc`가 설치되어 있지 않으면 자동으로 `st.camera_input()` 스냅샷
  모드로 폴백합니다(이 경우 음성 녹음/음성·언어 분석은 생략).
- requirements.txt에 `streamlit-webrtc`, `aiortc`, `av`가 포함되어 있어야 합니다.
