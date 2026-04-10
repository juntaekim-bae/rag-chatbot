import base64
import concurrent.futures
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

SUB_SYSTEM_PROMPT = """당신은 제공된 문서를 기반으로 하위 질문에 간결하게 답변하는 AI 어시스턴트입니다.
3~5문장 이내로 핵심만 답변하세요. 문서에 없으면 "문서에 없음"이라고만 하세요."""

COMBINE_SYSTEM_PROMPT = """당신은 여러 하위 답변을 통합하여 최종 답변을 작성하는 AI 어시스턴트입니다.
하위 답변들을 자연스럽게 통합하여 완성도 높은 답변을 작성하세요.
중복 내용은 제거하고, 논리적으로 흐름이 이어지게 작성하세요."""


# ── 질문 캐시 ─────────────────────────────────────────────────────────────────
class QuestionCache:
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
    if len(question) < 50:
        return [question]
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "다음 질문을 검색에 유리한 독립적인 하위 질문 2~3개로 분해하세요. "
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


# ── 이미지 질문 추출 ──────────────────────────────────────────────────────────
def image_to_question(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                {"type": "text", "text": (
                    "이 이미지에 있는 질문이나 텍스트를 추출해주세요. "
                    "이미지가 질문이면 질문을 그대로 출력하고, "
                    "텍스트가 있으면 해당 내용을 요약하여 질문 형태로 만들어주세요. "
                    "설명 없이 질문/텍스트만 출력하세요."
                )}
            ]
        }]
    )
    return resp.choices[0].message.content.strip()


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _build_context(results: list) -> tuple[str, list[str]]:
    sources: list[str] = []
    parts: list[str] = []
    for r in results:
        fname = r["metadata"].get("filename", "Unknown")
        if fname not in sources:
            sources.append(fname)
        parts.append(f"[출처: {fname}]\n{r['content']}")
    return "\n\n---\n\n".join(parts), sources


def _answer_sub(sub_q: str, vector_store) -> tuple[str, str, list[str]]:
    """하위 질문 1개에 대해 검색 + 답변 생성 (스트리밍 없이 완성 답변 반환)."""
    results = vector_store.search(sub_q, n_results=TOP_K)
    if not results:
        return sub_q, "관련 문서를 찾을 수 없습니다.", []
    context, sources = _build_context(results)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SUB_SYSTEM_PROMPT},
                {"role": "user", "content": f"=== 문서 ===\n{context}\n\n=== 질문 ===\n{sub_q}"}
            ]
        )
        return sub_q, resp.choices[0].message.content.strip(), sources
    except Exception as e:
        return sub_q, f"오류: {e}", []


# ── 메인 스트림 ───────────────────────────────────────────────────────────────
def chat_stream(question: str, vector_store) -> Iterator[str]:
    # 1. 캐시 확인
    cached = question_cache.get(question)
    if cached:
        yield f"data: {json.dumps({'type': 'cached', 'content': True})}\n\n"
        yield f"data: {json.dumps({'type': 'text', 'content': cached})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. 질문 분해 + 기본 검색 병렬
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_decompose = ex.submit(_decompose_question, question)
        f_search = ex.submit(vector_store.search, question, TOP_K)
        sub_questions = f_decompose.result()
        base_results = f_search.result()

    # 3. 단일 질문이면 기존 방식으로 빠르게 처리
    if len(sub_questions) <= 1:
        if not base_results:
            yield f"data: {json.dumps({'type': 'text', 'content': '관련 문서를 찾을 수 없습니다. 먼저 문서를 업로드해주세요.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context, sources = _build_context(base_results)
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        full_answer = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL, max_tokens=2048, stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"=== 문서 ===\n{context}\n\n=== 질문 ===\n{question}"}
                ]
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_answer += delta.content
                    yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'AI 오류: {str(e)}'})}\n\n"

        if full_answer:
            question_cache.set(question, full_answer)
        yield "data: [DONE]\n\n"
        return

    # 4. 복합 질문: 분해된 질문들을 순차적으로 답변 스트리밍
    yield f"data: {json.dumps({'type': 'decomposed', 'questions': sub_questions})}\n\n"

    all_sources: list[str] = []
    sub_answers: list[str] = []

    for i, sq in enumerate(sub_questions):
        # 하위 질문 시작 알림
        yield f"data: {json.dumps({'type': 'sub_start', 'index': i, 'question': sq, 'total': len(sub_questions)})}\n\n"

        results = vector_store.search(sq, n_results=TOP_K)
        if not results:
            sub_text = "관련 내용을 문서에서 찾을 수 없습니다."
            yield f"data: {json.dumps({'type': 'sub_text', 'index': i, 'content': sub_text})}\n\n"
            sub_answers.append(f"Q: {sq}\nA: {sub_text}")
            yield f"data: {json.dumps({'type': 'sub_end', 'index': i})}\n\n"
            continue

        context, sources = _build_context(results)
        for s in sources:
            if s not in all_sources:
                all_sources.append(s)

        sub_text = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL, max_tokens=600, stream=True,
                messages=[
                    {"role": "system", "content": SUB_SYSTEM_PROMPT},
                    {"role": "user", "content": f"=== 문서 ===\n{context}\n\n=== 질문 ===\n{sq}"}
                ]
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    sub_text += delta.content
                    yield f"data: {json.dumps({'type': 'sub_text', 'index': i, 'content': delta.content})}\n\n"
        except Exception as e:
            err = f"오류: {e}"
            yield f"data: {json.dumps({'type': 'sub_text', 'index': i, 'content': err})}\n\n"
            sub_text = err

        sub_answers.append(f"Q: {sq}\nA: {sub_text}")
        yield f"data: {json.dumps({'type': 'sub_end', 'index': i})}\n\n"

    # 5. 출처 전송
    if all_sources:
        yield f"data: {json.dumps({'type': 'sources', 'sources': all_sources})}\n\n"

    # 6. 최종 통합 답변 스트리밍
    yield f"data: {json.dumps({'type': 'final_start'})}\n\n"

    combined_input = "\n\n".join(sub_answers)
    full_answer = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL, max_tokens=1500, stream=True,
            messages=[
                {"role": "system", "content": COMBINE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"원래 질문: {question}\n\n"
                    f"하위 답변들:\n{combined_input}\n\n"
                    "위 내용을 통합하여 자연스럽고 완성도 있는 최종 답변을 작성하세요."
                )}
            ]
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_answer += delta.content
                yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'통합 오류: {str(e)}'})}\n\n"

    # 7. 캐시 저장
    if full_answer:
        question_cache.set(question, full_answer)

    yield "data: [DONE]\n\n"
