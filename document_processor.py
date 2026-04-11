import hashlib
import logging
import re
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".hwp", ".hwpx"}



def read_pdf(filepath: Path) -> str:
    """pdfplumber로 페이지별 추출, 타임아웃 보장"""
    import subprocess, sys, json

    script = r"""
import sys, json, warnings, signal
warnings.filterwarnings('ignore')

def handler(signum, frame):
    raise TimeoutError()

filepath = sys.argv[1]
results = []

try:
    import pdfplumber
    with pdfplumber.open(filepath) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)  # 페이지당 10초
            try:
                parts = []
                t = (page.extract_text() or '').strip()
                if t:
                    parts.append(t)
                try:
                    for table in page.extract_tables():
                        rows = [' | '.join(str(c).strip() for c in row if c and str(c).strip()) for row in table]
                        rows = [r for r in rows if r]
                        if rows:
                            parts.append('\n'.join(rows))
                except:
                    pass
                if parts:
                    results.append('\n'.join(parts))
            except TimeoutError:
                pass
            finally:
                signal.alarm(0)
    print(json.dumps({'text': '\n\n'.join(results), 'pages': total}))
except Exception as e:
    print(json.dumps({'text': '', 'pages': 1, 'error': str(e)}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(filepath)],
            capture_output=True, text=True, timeout=180
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            text = data.get("text", "")
            if text.strip():
                logger.info(f"{filepath.name}: {len(text)}자 ({data.get('pages',0)}페이지)")
                return text
    except subprocess.TimeoutExpired:
        logger.warning(f"{filepath.name}: PDF 추출 전체 타임아웃")
    except Exception as e:
        logger.warning(f"{filepath.name}: PDF 추출 오류 ({e})")

    # 최후 수단: pypdf 직접 시도
    try:
        from pypdf import PdfReader
        import warnings as w; w.filterwarnings("ignore")
        reader = PdfReader(str(filepath))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(t.strip() for t in pages if t.strip())
    except Exception:
        return ""


def read_docx(filepath: Path) -> str:
    from docx import Document

    doc = Document(str(filepath))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore")


def read_hwp(filepath: Path) -> str:
    import subprocess
    result = subprocess.run(
        ["hwp5txt", str(filepath)],
        capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise ValueError(f"HWP 변환 실패: {result.stderr.strip()}")
    return result.stdout


def read_hwpx(filepath: Path) -> str:
    import zipfile
    from xml.etree import ElementTree as ET
    parts = []
    with zipfile.ZipFile(filepath, "r") as zf:
        for name in zf.namelist():
            if name.startswith("Contents/") and name.endswith(".xml"):
                with zf.open(name) as f:
                    tree = ET.parse(f)
                    parts.append(" ".join(tree.getroot().itertext()))
    return "\n".join(parts)


def read_document(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(filepath)
    if suffix in (".docx", ".doc"):
        return read_docx(filepath)
    if suffix in (".txt", ".md", ".markdown"):
        return read_text(filepath)
    if suffix == ".hwp":
        return read_hwp(filepath)
    if suffix == ".hwpx":
        return read_hwpx(filepath)
    raise ValueError(f"지원하지 않는 파일 형식: {suffix}")


def normalize_korean(text: str) -> str:
    """공백 정규화"""
    text = re.sub(r'[ \t]+', ' ', text)    # 연속 공백 → 단일 공백
    text = re.sub(r'\n{3,}', '\n\n', text)  # 과도한 줄바꿈 압축
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """한국어 문장 분리 — 문장 끝(다/요/까/죠/네 + 마침표/물음표/느낌표) 기준"""
    # 문장 경계: 한국어 종결어미 뒤 마침표류, 또는 줄바꿈
    pattern = r'(?<=[다요까죠네])[.!?]\s+|(?<=[.!?])\s+(?=[가-힣A-Z])'
    parts = re.split(pattern, text)
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences if sentences else [text]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """문장 경계를 지키면서 청크 분할"""
    text = normalize_korean(text)
    sentences = split_sentences(text)

    chunks = []
    current = ""

    for sent in sentences:
        # 현재 청크에 문장 추가 시 크기 초과
        if current and len(current) + len(sent) + 1 > chunk_size:
            chunks.append(current.strip())
            # 오버랩: 현재 청크 끝부분을 다음 청크 시작으로
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    # 너무 짧은 청크는 다음과 합치기
    merged = []
    for c in chunks:
        if merged and len(c) < 100:
            merged[-1] += " " + c
        else:
            merged.append(c)

    return merged if merged else [text]


def process_document(filepath: Path, vector_store) -> int:
    filename = filepath.name
    logger.info(f"Processing {filename}...")

    text = read_document(filepath)
    if not text.strip():
        logger.warning(f"빈 문서: {filename}")
        return 0

    chunks = chunk_text(text)
    vector_store.delete_document(filename)

    ids, metadatas = [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:30]}".encode()).hexdigest()
        ids.append(chunk_id)
        metadatas.append(
            {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )

    vector_store.add_documents(chunks, metadatas, ids)
    logger.info(f"{filename}: {len(chunks)}개 청크 인덱싱 완료")
    return len(chunks)
