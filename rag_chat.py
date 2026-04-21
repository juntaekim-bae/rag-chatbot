import base64
import concurrent.futures
import hashlib
import json
import logging
import re
import time
from typing import Iterator, Optional

from groq import Groq

from config import GROQ_API_KEY, MODEL, TOP_K

logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _trim_context(messages: list, max_doc_chars: int = 6000) -> list:
    """문서 컨텍스트가 너무 길면 잘라서 반환."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if msg["role"] == "user" and "=== 문서 ===" in content:
            doc_start = content.find("=== 문서 ===")
            rest_start = content.find("\n\n=== 질문 ===")
            if doc_start != -1 and rest_start != -1:
                doc_section = content[doc_start:rest_start]
                if len(doc_section) > max_doc_chars:
                    doc_section = doc_section[:max_doc_chars] + "\n...(문서 일부 생략)"
                content = content[:doc_start] + doc_section + content[rest_start:]
            result.append({**msg, "content": content})
        else:
            result.append(msg)
    return result


def _apply_korean_force(messages: list) -> list:
    """fallback 모델용: 시스템 프롬프트 강화 + 한국어 assistant prefill 주입."""
    result = []
    for msg in messages:
        if msg["role"] == "system":
            # 시스템 프롬프트 맨 앞에 한국어 강제 지시 추가
            result.append({**msg, "content": "You MUST respond ONLY in Korean (한국어). No English allowed.\n\n" + msg["content"]})
        elif msg["role"] == "assistant" and msg.get("content") == "":
            # 빈 prefill → 한국어 시작 텍스트로 교체해 모델이 한국어로 이어가도록 강제
            result.append({**msg, "content": "네, 한국어로 답변드리겠습니다.\n\n"})
        else:
            result.append(msg)
    return result


def _groq_stream(messages: list, max_tokens: int):
    """rate limit(429) 또는 요청 초과(413) 시 fallback 모델로 자동 재시도. 그래도 실패하면 컨텍스트를 줄여 재시도."""
    def _is_rate_or_size(e: Exception) -> bool:
        s = str(e).lower()
        return "rate_limit" in s or "429" in s or "413" in s or "request too large" in s

    try:
        return groq_client.chat.completions.create(
            model=MODEL, max_tokens=max_tokens, stream=True, messages=messages
        )
    except Exception as e:
        if not _is_rate_or_size(e):
            raise
        logger.warning(f"Rate/size limit on {MODEL}, switching to {FALLBACK_MODEL}")

    fallback_messages = _apply_korean_force(messages)
    try:
        return groq_client.chat.completions.create(
            model=FALLBACK_MODEL, max_tokens=max_tokens, stream=True, messages=fallback_messages
        )
    except Exception as e:
        if not _is_rate_or_size(e):
            raise
        logger.warning(f"Rate/size limit on {FALLBACK_MODEL}, trimming context and retrying")

    trimmed = _trim_context(fallback_messages)
    return groq_client.chat.completions.create(
        model=FALLBACK_MODEL, max_tokens=max_tokens, stream=True, messages=trimmed
    )

# 한자(CJK) 및 일본어 문자 필터 — 스트리밍 출력에 직접 적용
_FOREIGN_RE = re.compile(
    '[\u4E00-\u9FFF'   # CJK Unified Ideographs
    '\u3400-\u4DBF'    # CJK Extension A
    '\uF900-\uFAFF'    # CJK Compatibility Ideographs
    '\u3040-\u309F'    # 히라가나
    '\u30A0-\u30FF'    # 카타카나
    '\u31F0-\u31FF'    # 카타카나 음성 확장
    '\uFF65-\uFF9F]'   # 반각 카타카나
)

def _strip_foreign(text: str) -> str:
    """한자를 한글 독음으로 변환, 나머지 외국 문자 제거."""
    try:
        import hanja
        text = hanja.translate(text, 'substitution')
    except Exception:
        pass
    cleaned = _FOREIGN_RE.sub('', text)
    return re.sub(r'  +', ' ', cleaned)


_KOREAN_ONLY = (
    "【언어 규칙 — 절대 원칙】"
    "모든 답변은 반드시 한국어로만 작성하라. "
    "영어·중국어·베트남어·일본어·스페인어 등 어떤 외국어도 단 한 글자도 사용하지 말 것. "
    "한자(漢字, 한문 글자)도 절대 사용하지 말 것. 한자가 필요한 경우 한글 독음으로만 표기하라. "
    "문서나 질문에 외국어나 한자가 포함되어 있어도 답변은 한글로만 작성하라. "
    "이 규칙은 어떤 경우에도 예외가 없다."
)

_MCQ_RULES = """【객관식 문제 출력 형식 — 절대 준수】
문제에 선지(①②③④⑤ 또는 1~5번)가 있으면 반드시 아래 3단계 순서대로 출력하라. 순서를 바꾸거나 단계를 생략하지 말 것.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1단계] (가)·(나) 등 빈칸 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
문제에 (가)(나)(다) 등 빈칸이 있는 경우에만 작성한다.
각 빈칸이 무엇을 가리키는지 문서를 근거로 먼저 설명하라.
형식:
(가): [무엇인지] — [근거 설명]
(나): [무엇인지] — [근거 설명]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2단계] 각 선지 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
선지를 번호 순서대로 하나씩 빠짐없이 분석하라.
각 선지마다 내용이 옳은지 틀린지를 문서 근거와 함께 설명하라.
형식:
① [선지 내용] → ✅ 옳다 / ❌ 틀리다
   근거: [문서에서 찾은 내용]
② [선지 내용] → ✅ 옳다 / ❌ 틀리다
   근거: [문서에서 찾은 내용]
(이하 모든 선지 동일하게)

【중요 — 질문 유형 구분】
- "옳은 것은?" → ✅ 옳다인 선지가 정답
- "옳지 않은 것은?" / "틀린 것은?" → ❌ 틀리다인 선지가 정답
질문이 "옳지 않은 것"을 묻는 경우, 2단계 분석에서 ❌ 틀리다로 표시된 선지를 정답으로 선택하라.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[3단계] 최종 정답
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 선지 분석이 끝난 후 마지막에 출력한다. 절대 먼저 쓰지 말 것.
형식:
▶ 정답: X번 (모범답안 기준) 또는 ▶ 정답: X번 (문서 근거)
이유: 왜 이 선지가 답인지 1~2문장으로 요약하라.

【모범답안 우선 원칙 — 반드시 준수】
컨텍스트에 "=== 모범답안 ===" 섹션이 있으면 다음을 따르라.

1. 모범답안에 표기된 정답 번호를 최종 정답으로 반드시 사용하라.
   (모범답안 표기 예: '1번-③', '2. ①', '3번: ②', '③' 등)

2. 분석 결과(2단계)가 모범답안 정답과 일치하면:
   ▶ 정답: X번 (모범답안 일치)

3. 분석 결과와 모범답안 정답이 다르면:
   ▶ 정답: X번 (모범답안 기준)
   ⚠️ 불일치 원인 분석:
   - 분석 결과: Y번 / 모범답안: X번
   - 원인: [왜 분석과 다른지 — 문서 내용 부족, 선지 해석 차이, 모범답안 오류 가능성 등을 구체적으로 설명]

4. 모범답안이 없으면 분석 결과를 그대로 사용하라."""

SYSTEM_PROMPT = f"""{_KOREAN_ONLY}

당신은 제공된 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

{_MCQ_RULES}

규칙:
1. 문서에 직접적인 내용이 없더라도, 관련된 개념이나 유사한 내용이 있으면 그것을 바탕으로 답변하세요.
2. 문서에 전혀 관련 내용이 없을 때만 "문서에서 찾을 수 없습니다"라고 하세요.
3. 답변은 명확하고 구조적으로 작성하세요.
4. 문서 내용을 최대한 활용하여 풍부하게 답변하세요.

{_KOREAN_ONLY}"""

SUB_SYSTEM_PROMPT = f"""{_KOREAN_ONLY}

당신은 제공된 문서를 기반으로 하위 질문에 답변하는 AI 어시스턴트입니다.

{_MCQ_RULES}

핵심 내용을 3~5문장으로 답변하세요.
직접적인 내용이 없어도 관련 내용이 있으면 활용하세요.
정말 아무 관련도 없을 때만 "관련 내용 없음"이라고 하세요.

{_KOREAN_ONLY}"""

COMBINE_SYSTEM_PROMPT = f"""{_KOREAN_ONLY}

당신은 여러 하위 답변을 통합하여 최종 답변을 작성하는 AI 어시스턴트입니다.

{_MCQ_RULES}

하위 답변들을 자연스럽게 통합하여 완성도 높은 답변을 작성하세요.
중복 내용은 제거하고, 논리적으로 흐름이 이어지게 작성하세요.

{_KOREAN_ONLY}"""


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
    # 80자 미만은 단순 질문으로 분해 안 함
    if len(question) < 80:
        return [question]
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,
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
    resp = groq_client.chat.completions.create(
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


# ── 출제원안↔모범답안 페어링 ──────────────────────────────────────────────────
_PAIR_MAP = {"출제원안": "모범답안", "모범답안": "출제원안"}

def _get_paired_chunks(results: list, vector_store) -> list:
    """검색 결과에 포함된 출제원안/모범답안 파일의 짝 파일 청크를 추가로 반환."""
    all_docs = {d["filename"] for d in vector_store.list_documents()}
    extra, seen = [], set()
    for r in results:
        fname = r["metadata"].get("filename", "")
        for src, dst in _PAIR_MAP.items():
            if src in fname:
                paired = fname.replace(src, dst)
                if paired in all_docs and paired not in seen:
                    seen.add(paired)
                    extra.extend(vector_store.get_file_chunks(paired))
                break
    return extra


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _dedup_results(results: list) -> list:
    """거의 동일한 청크 제거 (앞 100자 기준)"""
    seen = set()
    deduped = []
    for r in results:
        key = re.sub(r'\s+', '', r["content"][:100])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _build_context(results: list) -> tuple[str, list[str]]:
    results = _dedup_results(results)
    sources: list[str] = []
    doc_parts: list[str] = []
    mobeom_parts: list[str] = []
    for r in results:
        fname = r["metadata"].get("filename", "Unknown")
        if fname not in sources:
            sources.append(fname)
        if "모범답안" in fname:
            mobeom_parts.append(f"[출처: {fname}]\n{r['content']}")
        else:
            doc_parts.append(f"[출처: {fname}]\n{r['content']}")
    context = "\n\n---\n\n".join(doc_parts)
    if mobeom_parts:
        context += "\n\n=== 모범답안 ===\n" + "\n\n---\n\n".join(mobeom_parts)
    return context, sources


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

    # 출제원안↔모범답안 페어링 청크 추가
    paired = _get_paired_chunks(base_results, vector_store)
    if paired:
        seen_ids = {r["content"][:80] for r in base_results}
        base_results = base_results + [c for c in paired if c["content"][:80] not in seen_ids]

    # 3. 단일 질문이면 빠르게 처리
    if len(sub_questions) <= 1:
        if not base_results:
            yield f"data: {json.dumps({'type': 'text', 'content': '관련 문서를 찾을 수 없습니다. 먼저 문서를 업로드해주세요.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context, sources = _build_context(base_results)
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        full_answer = ""
        try:
            stream = _groq_stream([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"=== 문서 ===\n{context}\n\n=== 질문 ===\n{question}\n\n반드시 한국어로만 답변하세요."},
                    {"role": "assistant", "content": ""}
                ], max_tokens=2048)
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    text = _strip_foreign(text)
                    full_answer += text
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'AI 오류: {str(e)}'})}\n\n"

        if full_answer:
            question_cache.set(question, full_answer)
        yield "data: [DONE]\n\n"
        return

    # 4. 복합 질문: 분해된 질문들 순차 스트리밍
    yield f"data: {json.dumps({'type': 'decomposed', 'questions': sub_questions})}\n\n"

    all_sources: list[str] = []
    sub_answers: list[str] = []

    for i, sq in enumerate(sub_questions):
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
            stream = _groq_stream([
                    {"role": "system", "content": SUB_SYSTEM_PROMPT},
                    {"role": "user", "content": f"=== 문서 ===\n{context}\n\n=== 질문 ===\n{sq}\n\n반드시 한국어로만 답변하세요."},
                    {"role": "assistant", "content": ""}
                ], max_tokens=600)
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    text = _strip_foreign(text)
                    sub_text += text
                    yield f"data: {json.dumps({'type': 'sub_text', 'index': i, 'content': text})}\n\n"
        except Exception as e:
            err = f"오류: {e}"
            yield f"data: {json.dumps({'type': 'sub_text', 'index': i, 'content': err})}\n\n"
            sub_text = err

        sub_answers.append(f"Q: {sq}\nA: {sub_text}")
        yield f"data: {json.dumps({'type': 'sub_end', 'index': i})}\n\n"

    if all_sources:
        yield f"data: {json.dumps({'type': 'sources', 'sources': all_sources})}\n\n"

    yield f"data: {json.dumps({'type': 'final_start'})}\n\n"

    combined_input = "\n\n".join(sub_answers)
    full_answer = ""
    try:
        stream = _groq_stream([
                {"role": "system", "content": COMBINE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"원래 질문: {question}\n\n"
                    f"하위 답변들:\n{combined_input}\n\n"
                    "위 내용을 통합하여 자연스럽고 완성도 있는 최종 답변을 작성하세요. 반드시 한국어로만 답변하세요."
                )},
                {"role": "assistant", "content": ""}
            ], max_tokens=1500)
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                text = _strip_foreign(text)
                full_answer += text
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'통합 오류: {str(e)}'})}\n\n"

    if full_answer:
        question_cache.set(question, full_answer)

    yield "data: [DONE]\n\n"
