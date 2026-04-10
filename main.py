import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from config import DOCUMENTS_DIR, MODEL
from document_processor import SUPPORTED_EXTENSIONS, process_document
from rag_chat import chat_stream
from vector_store import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _index_existing_documents():
    """기존 문서를 백그라운드에서 인덱싱합니다."""
    import threading
    def _run():
        logger.info("기존 문서 인덱싱 시작 (백그라운드)...")
        for fp in DOCUMENTS_DIR.iterdir():
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    process_document(fp, vector_store)
                except Exception as e:
                    logger.error(f"{fp.name} 인덱싱 실패: {e}")
        logger.info(f"인덱싱 완료 — 총 {vector_store.count()}개 청크")
    threading.Thread(target=_run, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _index_existing_documents()
    logger.info("서버 준비 완료 (인덱싱은 백그라운드 진행 중)")
    yield


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 모델 ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class DriveSyncRequest(BaseModel):
    url: str


# ── 기본 라우트 ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path("frontend/index.html")
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>RAG Chatbot</h1>")


# ── 문서 업로드 ───────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"지원하지 않는 형식: {suffix}. 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    filepath = DOCUMENTS_DIR / file.filename
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        count = process_document(filepath, vector_store)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(500, f"문서 처리 실패: {e}")
    return {"filename": file.filename, "chunks": count, "message": f"{count}개 청크로 인덱싱 완료"}


@app.get("/api/documents")
async def list_documents():
    return {
        "documents": vector_store.list_documents(),
        "total_chunks": vector_store.count(),
    }


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    vector_store.delete_document(filename)
    (DOCUMENTS_DIR / filename).unlink(missing_ok=True)
    return {"message": f"{filename} 삭제 완료"}


# ── 채팅 ──────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "질문을 입력해주세요")
    return StreamingResponse(
        chat_stream(request.question, vector_store),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Google Drive 동기화 ───────────────────────────────────────────────────────
@app.post("/api/drive/sync")
async def drive_sync(request: DriveSyncRequest, background_tasks: BackgroundTasks):
    from google_drive import get_status, parse_drive_url, run_sync

    if get_status()["running"]:
        raise HTTPException(409, "이미 동기화가 진행 중입니다.")

    drive_id, link_type = parse_drive_url(request.url)
    if not drive_id:
        raise HTTPException(400, "Google Drive 링크를 인식할 수 없습니다.")

    background_tasks.add_task(run_sync, request.url, DOCUMENTS_DIR, vector_store)
    return {"message": "동기화 시작", "type": link_type, "id": drive_id}


@app.get("/api/drive/status")
async def drive_status():
    from google_drive import get_status
    return get_status()


# ── 헬스체크 ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL, "total_chunks": vector_store.count()}
