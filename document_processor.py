import hashlib
import logging
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".hwp", ".hwpx"}


def read_pdf(filepath: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


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
