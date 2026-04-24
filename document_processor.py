import hashlib
import logging
import re
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".hwp", ".hwpx"}



_PDF_SIZE_LIMIT = 8 * 1024 * 1024  # 8MB 이상은 pdfplumber 스킵


def read_pdf(filepath: Path) -> str:
    """pdfplumber(소형 PDF)로 페이지별 추출, 대용량은 pypdf 직접 사용"""
    import subprocess, sys, json

    file_size = filepath.stat().st_size
    if file_size > _PDF_SIZE_LIMIT:
        logger.info(f"{filepath.name}: 대용량({file_size//1024//1024}MB), pypdf 직접 사용")
        return _read_pdf_pypdf(filepath)

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
            signal.alarm(5)  # 페이지당 5초
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
            capture_output=True, text=True, timeout=90
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            text = data.get("text", "")
            if text.strip():
                logger.info(f"{filepath.name}: {len(text)}자 ({data.get('pages',0)}페이지)")
                return text
    except subprocess.TimeoutExpired:
        logger.warning(f"{filepath.name}: pdfplumber 타임아웃, pypdf fallback")
    except Exception as e:
        logger.warning(f"{filepath.name}: pdfplumber 오류 ({e}), pypdf fallback")

    return _read_pdf_pypdf(filepath)


def _read_pdf_pypdf(filepath: Path) -> str:
    try:
        from pypdf import PdfReader
        import warnings as w; w.filterwarnings("ignore")
        reader = PdfReader(str(filepath))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(t.strip() for t in pages if t.strip())
        logger.info(f"{filepath.name}: pypdf {len(text)}자 ({len(reader.pages)}페이지)")
        return text
    except Exception as e:
        logger.error(f"{filepath.name}: pypdf 오류 ({e})")
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
    pattern = r'(?<=[다요까죠네])[.!?]\s+|(?<=[.!?])\s+(?=[가-힣A-Z])'
    parts = re.split(pattern, text)
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences if sentences else [text]


# ── 시험 문제 단위 청킹 ───────────────────────────────────────────────────────
# 줄 앞에 오는 문제 번호 패턴: "1.", "13번", "5)" 등
_QNUM_RE = re.compile(r'(?m)^\s*(\d{1,3})\s*[.번\)]\s')
_CHOICE_RE = re.compile(r'[①②③④⑤]')


def _is_exam_doc(text: str) -> bool:
    """시험지 형식 감지: 문제 번호 3개 이상 + 원문자 선지가 문제당 평균 2개 이상"""
    q_count = len(_QNUM_RE.findall(text))
    c_count = len(_CHOICE_RE.findall(text))
    return q_count >= 3 and c_count >= q_count * 2


def _chunk_exam(text: str) -> list[str]:
    """문제 번호 경계로 청킹 — 각 문제(지문+선지)를 하나의 청크로 유지"""
    matches = list(_QNUM_RE.finditer(text))
    if not matches:
        return []
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """시험 문서는 문제 단위로, 일반 문서는 문장 경계 청크 분할"""
    text = normalize_korean(text)

    # 시험지 형식이면 문제 단위로 청킹
    if _is_exam_doc(text):
        exam_chunks = _chunk_exam(text)
        if exam_chunks:
            return exam_chunks

    # 일반 문서: 문장 경계 청킹
    sentences = split_sentences(text)

    chunks = []
    current = ""

    for sent in sentences:
        if current and len(current) + len(sent) + 1 > chunk_size:
            chunks.append(current.strip())
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


def prepare_chunks(filepath: Path) -> tuple[list[str], list[dict], list[str]]:
    """청크 준비만 하고 임베딩/저장은 하지 않음 (배치 처리용)"""
    filename = filepath.name
    text = read_document(filepath)
    if not text.strip():
        logger.warning(f"빈 문서: {filename}")
        return [], [], []

    chunks = chunk_text(text)
    ids: list[str] = []
    metadatas: list[dict] = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:30]}".encode()).hexdigest()
        ids.append(chunk_id)
        metadatas.append({"filename": filename, "chunk_index": i, "total_chunks": len(chunks)})
    logger.info(f"{filename}: {len(chunks)}개 청크 준비 완료")
    return chunks, metadatas, ids


def process_document(filepath: Path, vector_store) -> int:
    filename = filepath.name
    logger.info(f"Processing {filename}...")

    chunks, metadatas, ids = prepare_chunks(filepath)
    if not chunks:
        return 0

    vector_store.delete_document(filename)
    vector_store.add_documents(chunks, metadatas, ids)
    logger.info(f"{filename}: {len(chunks)}개 청크 인덱싱 완료")
    return len(chunks)
