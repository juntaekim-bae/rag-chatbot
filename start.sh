#!/bin/bash
set -e

# .env 파일 확인
if [ ! -f .env ]; then
  echo "📋 .env 파일 생성 중..."
  cp .env.example .env
  echo "⚠️  .env 파일에 ANTHROPIC_API_KEY를 설정해주세요!"
  echo ""
fi

# 가상환경 생성 및 활성화
if [ ! -d venv ]; then
  echo "🐍 가상환경 생성 중..."
  python3 -m venv venv
fi

source venv/bin/activate

# 패키지 설치
echo "📦 패키지 설치 중..."
pip install -q -r requirements.txt

# 서버 실행
echo ""
echo "🚀 서버 시작: http://localhost:8000"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
