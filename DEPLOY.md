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

## Supabase 설정 (면접 후기 커뮤니티)
커뮤니티 탭(💬 면접 후기)은 Supabase(무료 PostgreSQL)를 백엔드로 사용합니다.
설정하지 않으면 커뮤니티 탭은 안내 메시지만 보여주고, 나머지 앱은 정상 동작합니다.

1. https://supabase.com 접속 → 무료 프로젝트 생성
2. SQL Editor에서 아래 `CREATE TABLE` 스크립트를 실행
3. Settings → API → **Project URL**과 **anon public key** 복사
4. 키 등록:
   - 로컬: `.env`에 추가
     ```
     SUPABASE_URL=https://xxxx.supabase.co
     SUPABASE_ANON_KEY=eyJ...
     ```
   - Streamlit Cloud: Manage app → Secrets에 추가
     ```
     SUPABASE_URL = "https://xxxx.supabase.co"
     SUPABASE_ANON_KEY = "eyJ..."
     ```

### 테이블 스키마 (SQL Editor에서 실행)
```sql
CREATE TABLE interview_posts (
  id          BIGSERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  company     TEXT NOT NULL,
  role        TEXT NOT NULL,          -- 직무 (예: RA, 전략컨설턴트)
  round       TEXT NOT NULL,          -- 면접 단계 (1차/2차/임원/최종)
  result      TEXT NOT NULL,          -- 합격/불합격/결과대기
  atmosphere  TEXT NOT NULL,          -- 분위기 (편안함/보통/압박)
  questions   TEXT NOT NULL,          -- 받은 질문들 (줄바꿈으로 구분)
  review      TEXT NOT NULL,          -- 전체 후기 본문
  tips        TEXT,                   -- 준비 팁 (선택)
  author_hash TEXT NOT NULL,          -- sha256(nickname) — 익명 식별자
  nickname    TEXT NOT NULL,          -- 표시용 닉네임
  likes       INT DEFAULT 0
);

-- 인덱스
CREATE INDEX idx_company ON interview_posts(company);
CREATE INDEX idx_created ON interview_posts(created_at DESC);

-- 로그인 없이 운영: 누구나 읽기/쓰기/좋아요 가능
ALTER TABLE interview_posts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "public read" ON interview_posts FOR SELECT USING (true);
CREATE POLICY "public insert" ON interview_posts FOR INSERT WITH CHECK (true);
CREATE POLICY "public update likes" ON interview_posts
  FOR UPDATE USING (true) WITH CHECK (true);
```

### 동작 방식
- **Write-gate**: 본인의 면접 경험을 1개 작성해야 다른 후기를 열람할 수 있습니다
  (잡플래닛·블라인드 방식). 세션 단위로 적용됩니다.
- 닉네임은 표시용이며, 익명 식별자(`author_hash`)는 `sha256(nickname+salt)`의
  앞 12자입니다.
