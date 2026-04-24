"""
Google Drive 공유 링크에서 파일/폴더를 다운로드하고 자동 인덱싱합니다.

지원 링크 형식:
  - 파일: https://drive.google.com/file/d/FILE_ID/view?usp=...
  - 폴더: https://drive.google.com/drive/folders/FOLDER_ID?usp=...
  - 단축: https://drive.google.com/open?id=ID
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── 전역 동기화 상태 ──────────────────────────────────────────────────────────
_status: dict = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current": "",
    "downloaded": [],
    "errors": [],
    "done": False,
    "message": "대기 중",
    "files_detail": [],  # [{"name": "...", "status": "done"|"error"|"processing"}]
}


# ── Drive URL 영구 저장 ───────────────────────────────────────────────────────
def _config_path() -> Path:
    from config import CHROMA_DIR
    return CHROMA_DIR.parent / "drive_config.json"


def save_drive_url(url: str):
    try:
        _config_path().write_text(json.dumps({"url": url}), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Drive URL 저장 실패: {e}")


def load_drive_url() -> str:
    try:
        p = _config_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("url", "")
    except Exception as e:
        logger.warning(f"Drive URL 로드 실패: {e}")
    return ""


def get_status() -> dict:
    return dict(_status)


def _set(**kwargs):
    _status.update(kwargs)


# ── URL 파싱 ──────────────────────────────────────────────────────────────────
def parse_drive_url(url: str) -> tuple[str, str]:
    """
    Google Drive URL에서 (id, type) 추출.
    type: 'file' | 'folder' | 'unknown'
    """
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1), "file"

    m = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1), "folder"

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1), "unknown"

    return "", "unknown"


# ── Google Drive API 호출 ─────────────────────────────────────────────────────
def _api_list_folder(folder_id: str, api_key: str) -> list[dict]:
    """폴더 내 파일 목록 반환 (재귀 포함)"""
    files = []
    page_token = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "key": api_key,
            "pageSize": 1000,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for f in data.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                # 하위 폴더 재귀
                files.extend(_api_list_folder(f["id"], api_key))
            else:
                files.append(f)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files


def _download_file(file_id: str, filename: str, dest: Path) -> Path:
    """Drive API로 파일 1개 다운로드"""
    # Google Docs 계열은 export, 일반 파일은 alt=media
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    from config import GOOGLE_API_KEY
    params = {"alt": "media", "key": GOOGLE_API_KEY}

    resp = requests.get(url, params=params, stream=True, timeout=60)

    # Google Docs 형식이면 PDF로 export
    if resp.status_code == 403:
        export_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
        params = {"mimeType": "application/pdf", "key": GOOGLE_API_KEY}
        resp = requests.get(export_url, params=params, stream=True, timeout=60)
        filename = Path(filename).stem + ".pdf"

    resp.raise_for_status()

    out_path = dest / filename
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path


# ── 메인 동기화 함수 ──────────────────────────────────────────────────────────
async def run_sync(share_url: str, documents_dir: Path, vector_store) -> None:
    if _status["running"]:
        return

    save_drive_url(share_url)
    _set(running=True, progress=0, total=0, current="",
         downloaded=[], errors=[], done=False, message="초기화 중...", files_detail=[])

    from config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        _set(running=False, done=True,
             message="GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    drive_id, link_type = parse_drive_url(share_url)
    if not drive_id:
        _set(running=False, done=True,
             message="Google Drive 링크를 인식할 수 없습니다.")
        return

    downloaded: list[str] = []
    errors: list[str] = []

    try:
        if link_type == "folder":
            await _sync_folder(drive_id, documents_dir, vector_store, downloaded, errors)
        else:
            await _sync_file(drive_id, documents_dir, vector_store, downloaded, errors)
    except Exception as e:
        logger.exception("Google Drive 동기화 오류")
        errors.append(str(e))

    msg = f"완료: {len(downloaded)}개 다운로드"
    if errors:
        msg += f", {len(errors)}개 실패"
    _set(running=False, done=True, progress=_status["total"],
         downloaded=downloaded, errors=errors, message=msg)


# ── 파일 1개 동기화 ───────────────────────────────────────────────────────────
async def _sync_file(file_id: str, dest: Path, vector_store, downloaded: list, errors: list):
    from config import GOOGLE_API_KEY
    _set(total=1, message="파일 다운로드 중...")

    try:
        # 파일 메타데이터 조회
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"fields": "name,mimeType", "key": GOOGLE_API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        meta = resp.json()
        filename = meta["name"]

        existing = {d["filename"] for d in vector_store.list_documents()}
        if filename in existing:
            _status["files_detail"].append({"name": filename, "status": "skipped"})
            _set(progress=1, message=f"이미 인덱싱된 파일: {filename}")
            return

        _status["files_detail"].append({"name": filename, "status": "processing"})
        _set(current=filename)
        out_path = _download_file(file_id, filename, dest)
        downloaded.append(out_path.name)
        _status["files_detail"][-1]["status"] = "done"
        _set(progress=1)
        _auto_index(str(out_path), vector_store, errors)
    except Exception as e:
        errors.append(f"파일 다운로드 오류: {e}")
        if _status["files_detail"]:
            _status["files_detail"][-1]["status"] = "error"
        logger.error(f"파일 다운로드 실패: {e}")


# ── 폴더 동기화 ───────────────────────────────────────────────────────────────
async def _sync_folder(folder_id: str, dest: Path, vector_store, downloaded: list, errors: list):
    from config import GOOGLE_API_KEY
    from document_processor import SUPPORTED_EXTENSIONS, prepare_chunks
    _set(message="폴더 파일 목록 가져오는 중...")

    try:
        files = _api_list_folder(folder_id, GOOGLE_API_KEY)
    except Exception as e:
        errors.append(f"폴더 목록 조회 실패: {e}")
        return

    if not files:
        errors.append("폴더가 비어있거나 접근할 수 없습니다.")
        return

    existing = {d["filename"] for d in vector_store.list_documents()}

    # 새로 받을 파일만 추림
    to_download = []
    for f in files:
        fname = f["name"]
        if fname in existing:
            _status["files_detail"].append({"name": fname, "status": "skipped"})
        else:
            _status["files_detail"].append({"name": fname, "status": "pending"})
            to_download.append(f)

    _set(total=len(files), message=f"신규 {len(to_download)}개 병렬 다운로드 중...")

    # Phase 1: 4개씩 동시 다운로드
    out_paths: dict[str, Path] = {}  # fname → 저장 경로
    def _dl(f: dict):
        return f["name"], _download_file(f["id"], f["name"], dest)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_dl, f): f for f in to_download}
        done_count = len(existing)
        for future in as_completed(futures):
            f = futures[future]
            fname = f["name"]
            try:
                _, out_path = future.result()
                out_paths[fname] = out_path
                downloaded.append(out_path.name)
                _set_file_status(fname, "downloaded")
            except Exception as e:
                errors.append(f"{fname} 다운로드 실패: {e}")
                _set_file_status(fname, "error")
                logger.error(f"{fname} 다운로드 실패: {e}")
            done_count += 1
            _set(progress=done_count, message=f"다운로드 {done_count}/{len(files)}")

    if not out_paths:
        return

    # Phase 2: 전체 청크 수집 (파싱)
    _set(message=f"{len(out_paths)}개 문서 파싱 중...")
    all_texts, all_metas, all_ids = [], [], []
    for fname, path in out_paths.items():
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        try:
            _set_file_status(fname, "parsing")
            texts, metas, ids = prepare_chunks(path)
            if texts:
                vector_store.delete_document(fname)
                all_texts.extend(texts)
                all_metas.extend(metas)
                all_ids.extend(ids)
                _set_file_status(fname, "parsed")
        except Exception as e:
            errors.append(f"파싱 실패 {fname}: {e}")
            _set_file_status(fname, "error")
            logger.error(f"파싱 오류 {fname}: {e}")

    if not all_texts:
        return

    # Phase 3 & 4: 전체 임베딩 + 저장 한 번에 (Voyage API 128개씩 자동 배치)
    _set(message=f"총 {len(all_texts)}개 청크 임베딩 중...")
    try:
        vector_store.add_documents(all_texts, all_metas, all_ids)
        for fname in out_paths:
            _set_file_status(fname, "done")
        _set(progress=len(files), message=f"완료: {len(out_paths)}개 문서 인덱싱")
    except Exception as e:
        errors.append(f"임베딩/저장 실패: {e}")
        logger.error(f"배치 임베딩 오류: {e}")


def _set_file_status(fname: str, status: str):
    for f in _status["files_detail"]:
        if f["name"] == fname:
            f["status"] = status
            return


# ── 자동 인덱싱 ───────────────────────────────────────────────────────────────
def _auto_index(file_path: str, vector_store, errors: list):
    from document_processor import SUPPORTED_EXTENSIONS, process_document
    from main import _processing_lock

    fp = Path(file_path)

    if fp.suffix.lower() == ".zip":
        _extract_and_index(fp, vector_store, errors)
        return

    if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.info(f"인덱싱 제외 (미지원 형식): {fp.name}")
        return
    try:
        with _processing_lock:
            count = process_document(fp, vector_store)
        logger.info(f"인덱싱 완료: {fp.name} ({count}개 청크)")
    except Exception as e:
        errors.append(f"인덱싱 실패 {fp.name}: {e}")
        logger.error(f"인덱싱 오류: {e}")


def _extract_and_index(zip_path: Path, vector_store, errors: list):
    """ZIP 파일을 압축 해제하고 지원 문서를 인덱싱합니다."""
    import zipfile
    from document_processor import SUPPORTED_EXTENSIONS, process_document

    dest = zip_path.parent
    extracted = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                suffix = Path(member).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    continue
                safe_name = Path(member).name
                out_path = dest / safe_name
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(out_path)

        logger.info(f"ZIP 압축 해제: {zip_path.name} → {len(extracted)}개 파일")
        zip_path.unlink()

    except Exception as e:
        errors.append(f"ZIP 압축 해제 실패 {zip_path.name}: {e}")
        logger.error(f"ZIP 오류: {e}")
        return

    for fp in extracted:
        try:
            count = process_document(fp, vector_store)
            logger.info(f"인덱싱 완료: {fp.name} ({count}개 청크)")
        except Exception as e:
            errors.append(f"인덱싱 실패 {fp.name}: {e}")
            logger.error(f"인덱싱 오류: {e}")
