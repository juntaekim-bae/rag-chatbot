import base64
import hashlib
import json
import logging
import time
from typing import Iterator, Optional

from groq import Groq

from config import GROQ_API_KEY, MODEL, TOP_K

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

VISION_MODEL = "llama-3.2-11b-vision-preview"

SYSTEM_PROMPT = """당신은 제공된 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

규칙:
1. 반드시 제공된 문서 컨텍스트에 근거하여 답변하세요.
2. 문서에 없는 내용은 "해당 내용은 문서에 없습니다"라고 명확히 말하세요.
3. 답변은 명확하고 구조적으로 작성하세요.
4. 질문 언어에 맞춰 답변하세요 (한국어 질문 → 한국어 답변)."""


# ── 질문 캐시 ─────────────────────────────────────────────────────────────────
class QuestionCache:
    """자주 묻는 질문을 캐싱합니다 (1시간 유효)."""

    def __init__(self, max_size: int = 200, ttl: int = 3600):
        self._cache: dict = {}
        self._max_size = max_size
        self._ttl = ttl

    def _key(self, question: str) -> str:
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def get(self, question: str) -> Optional[str]:
        key = self._key(question)
        if key in self._cache:
            answer, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                return answer
            del self._cache[key]
        return None

    def set(self, question: str, answer: str):
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[self._key(question)] = (answer, time.time())

    def size(self) -> int:
        return len(self._cache)


question_cache = QuestionCache()


# ── 질문 분해 ─────────────────────────────────────────────────────────────────
def _decompose_question(question: str) -> list[str]:
    """복잡한 질문을 하위 질문들로 분해합니다."""
    # 짧고 단순한 질문은 분해하지 않음
    if len(question) < 40:
        return [question]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "다음 질문을 검색에 유리한 독립적인 하위 질문 1~3개로 분해하세요. "
                    "각 질문은 한 줄에 하나씩, 번호나 기호 없이 작성하세요. "
                    "질문이 단순하면 그대로 한 줄만 출력하세요.\n\n"
                    f"질문: {question}"
                )
            }]
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split('\n') if l.strip()]
        return lines[:3] if lines else [question]
    except Exception as e:
        logger.warning(f"질문 분해 실패: {e}")
        return [question]


# ── 이미지에서 텍스트 추출 ────────────────────────────────────────────────────
def image_to_question(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """이미지에서 질문/텍스트를 추출합니다."""
    b64 = base64.b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": (
                        "이 이미지에 있는 질문이나 텍스트를 추출해주세요. "
                        "이미지가 질문이면 질문을 그대로 출력하고, "
                        "텍스트가 있으면 해당 내용을 요약하여 질문 형태로 만들어주세요. "
                        "설명 없이 질문/텍스트만 출력하세요."
                    )
                }
            ]
        }]
    )
    return resp.choices[0].message.content.strip()


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def _build_context(results: list) -> tuple[str, list[str]]:
    sources: list[str] = []
    parts: list[str] = []
    for r in results:
        fname = r["metadata"].get("filename", "Unknown")
        if fname not in sources:
            sources.append(fname)
        parts.append(f"[출처: {fname}]\n{r['content']}")
    return "\n\n---\n\n".join(parts), sources


def _search_merged(question: str, sub_questions: list[str], vector_store) -> list:
    """여러 하위 질문으로 검색하고 중복 없이 결과를 합칩니다."""
    seen_content: set[str] = set()
    merged: list = []

    for sq in sub_questions:
        results = vector_store.search(sq, n_results=TOP_K)
        for r in results:
            key = r["content"][:80]
            if key not in seen_content:
                seen_content.add(key)
                merged.append(r)

    # 거리 기준 정렬 후 상위 TOP_K*2 반환
    merged.sort(key=lambda x: x.get("distance", 1))
    return merged[:TOP_K * 2]


# ── 메인 스트림 함수 ──────────────────────────────────────────────────────────
def chat_stream(question: str, vector_store) -> Iterator[str]:
    # 1. 캐시 확인
    cached = question_cache.get(question)
    if cached:
        yield f"data: {json.dumps({'type': 'cached', 'content': True})}\n\n"
        yield f"data: {json.dumps({'type': 'text', 'content': cached})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. 질문 분해
    sub_questions = _decompose_question(question)
    if len(sub_questions) > 1:
        yield f"data: {json.dumps({'type': 'decomposed', 'questions': sub_questions})}\n\n"

    # 3. 벡터 검색
    results = _search_merged(question, sub_questions, vector_store)
    if not results:
        yield f"data: {json.dumps({'type': 'text', 'content': '관련 문서를 찾을 수 없습니다. 먼저 문서를 업로드해주세요.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    context, sources = _build_context(results)
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # 4. LLM 스트림
    user_message = (
        f"아래 문서 내용을 참고하여 질문에 답변해주세요.\n\n"
        f"=== 문서 컨텍스트 ===\n{context}\n\n"
        f"=== 질문 ===\n{question}"
    )

    full_answer = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_answer += delta.content
                yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"
    except Exception as e:
        logger.error(f"Groq API 오류: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'AI 오류: {str(e)}'})}\n\n"

    # 5. 캐시 저장
    if full_answer:
        question_cache.set(question, full_answer)

    yield "data: [DONE]\n\n"
