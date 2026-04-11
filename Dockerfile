FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pyhwp 설치 (hwp5txt 명령어 포함)
RUN pip install --no-cache-dir pyhwp

# pdfplumber 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# sentence-transformers 모델 미리 다운로드 (빌드 시 캐시)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 소스 복사
COPY . .

# 문서/DB 디렉터리 생성 (볼륨 마운트 전 기본값)
RUN mkdir -p /data/documents /data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
