#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "🚀 AI 모의면접 앱을 시작합니다..."
streamlit run app.py
