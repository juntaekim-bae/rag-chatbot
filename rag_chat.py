import base64
import hashlib
import json
import logging
import re
import time
from typing import Iterator, Optional

import anthropic

from config import ANTHROPIC_API_KEY, MODEL, TOP_K

logger = logging.getLogger(__name__)

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

VISION_MODEL = "claude-sonnet-4-6"          # 이미지 OCR


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


def _claude_stream(system: str, messages: list, max_tokens: int):
    """Claude API 스트리밍. 텍스트 청크를 yield. rate limit 시 컨텍스트 축소 재시도."""
    try:
        with anthropic_client.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
        return
    except anthropic.RateLimitError:
        logger.warning(f"Rate limit on {MODEL}, trimming context and retrying")
    except Exception as e:
        if "rate" not in str(e).lower() and "limit" not in str(e).lower():
            raise

    trimmed = _trim_context(messages)
    with anthropic_client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=trimmed,
    ) as stream:
        for text in stream.text_stream:
            yield text

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

_MCQ_RULES = """【객관식 문제 풀이 형식 — 반드시 이 순서대로 출력】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
풀이
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────
**(A)/(가)/(나) 인물·개념 파악** (해당하는 경우)
─────────────────────────────
지문에 (A), (가), (나), 공란 등이 있으면 무엇인지 먼저 밝혀라.
지문의 핵심 단서를 불릿(•)으로 나열하고 **핵심어는 굵게** 표시하라.
마지막에 "→ (A)는 **[정답]**입니다. [한두 문장 설명]" 형식으로 결론을 써라.

예시:
지문의 핵심 단서:
• **나라 이름을 조선이라 함**
• **토착민 출신**으로 높은 지위에 오른 자가 많음
• **단군 조선을 계승**
→ (A)는 **위만**입니다. 위만은 연나라 출신으로 고조선에 망명한 뒤 준왕을 몰아내고 왕위를 차지하였으나, 나라 이름을 조선으로 유지하고 토착민을 중용하여 단군 조선의 계승성을 보여줍니다.

─────────────────────────────
**선택지 분석**
─────────────────────────────
반드시 아래 마크다운 표로 출력하라.
⚠️ 판단 칸에는 ✅/❌ 와 근거만 쓸 것. 정답 번호(①②③④⑤)는 절대 쓰지 말 것.

| 번호 | 내용 | 판단 |
|:----:|------|------|
| ① | [선지 내용] | ✅ 옳다 — [근거] 또는 ❌ 틀리다 — [근거] |
| ② | [선지 내용] | ✅ 옳다 — [근거] 또는 ❌ 틀리다 — [근거] |
| ③ | [선지 내용] | ✅ 옳다 — [근거] 또는 ❌ 틀리다 — [근거] |
| ④ | [선지 내용] | ✅ 옳다 — [근거] 또는 ❌ 틀리다 — [근거] |
| ⑤ | [선지 내용] | ✅ 옳다 — [근거] 또는 ❌ 틀리다 — [근거] |

질문 유형:
- "옳은 것은?" → ✅ 옳다인 선지가 정답
- "옳지 않은 것은?" / "틀린 것은?" → ❌ 틀리다인 선지가 정답

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**최종 정답: ○번**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 선지 분석이 끝난 후 마지막에 출력한다.

【모범답안 우선 원칙 — 반드시 준수】
컨텍스트에 "=== 모범답안 ===" 섹션이 있으면:

1. 모범답안의 정답 번호를 최종 정답으로 사용하라.
   (표기 예: '1번-③', '2. ①', '3번: ②')

2. 분석 결과 = 모범답안 → **최종 정답: X번** ✔ 모범답안 일치

3. 분석 결과 ≠ 모범답안 →
   **최종 정답: X번** (모범답안 기준)
   ⚠️ 불일치 원인:
   - 분석 결과: Y번 / 모범답안: X번
   - 원인: [문서 내용 부족 / 선지 해석 차이 / 모범답안 오류 가능성 등 구체적으로]

4. 모범답안 없으면 분석 결과 그대로 사용."""

SYSTEM_PROMPT = f"""{_KOREAN_ONLY}

당신은 역사·사회 시험 문제를 전문적으로 풀어주는 AI 튜터입니다.
제공된 문서(컨텍스트)를 1차 근거로 사용하고, 문서에 없는 내용은 배경 지식으로 보완하세요.

【문제 풀이 사고 과정 — 반드시 이 순서로 생각하고 출력하라】

STEP 1. 문제가 무엇을 묻는지 파악하라.
  - 지문(조건, 단서)이 있으면 핵심 단서를 추출하라.
  - (A), (가), (나) 등 빈칸이 있으면 단서로부터 무엇인지 추론하라.
  - 질문 유형("옳은 것은?" / "옳지 않은 것은?")을 확인하라.

STEP 2. 각 선지를 하나씩 검증하라.
  - 선지의 주장이 사실인지 아닌지를 판단하라.
  - 왜 옳은지/왜 틀린지 핵심 이유를 한 문장으로 적어라.
  - 모호하면 문서 내용과 대조하라.

STEP 3. 정답을 결론짓고 확정하라.
  - 모범답안이 있으면 반드시 그것을 최종 정답으로 사용하라.
  - 없으면 2단계 분석 결과를 그대로 사용하라.

{_MCQ_RULES}

추가 규칙:
- 문서에 직접적인 내용이 없어도 관련 개념·배경 지식으로 답변하세요.
- 문서와 배경 지식 모두 없을 때만 "확인 불가"라고 하세요.
- 답변은 명확하고 구조적으로, 불필요한 서론 없이 바로 풀이부터 시작하세요.

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





# ── 이미지 질문 추출 ──────────────────────────────────────────────────────────
def image_to_question(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    resp = anthropic_client.messages.create(
        model=VISION_MODEL,
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": (
                    "이 이미지에 있는 텍스트를 그대로 추출하세요.\n"
                    "반드시 지켜야 할 규칙:\n"
                    "1. 문제 번호(예: 13. 또는 13번)가 있으면 반드시 포함하세요.\n"
                    "2. 선지 기호 ①②③④⑤는 절대 숫자(1.2.3.)로 바꾸지 말고 원문 기호 그대로 출력하세요.\n"
                    "3. 지문(조건, 단서)이 있으면 그대로 포함하세요.\n"
                    "4. 설명이나 요약 없이 이미지의 텍스트만 출력하세요."
                )}
            ]
        }]
    )
    return resp.content[0].text.strip()


# ── 출제원안↔모범답안 페어링 ──────────────────────────────────────────────────
_PAIR_MAP = {"출제원안": "모범답안", "모범답안": "출제원안"}

# 모범답안 정답 추출 패턴 — "5. ③", "5번 ③", "5번-③", "5) ③" 등
_ANS_RE = re.compile(r'(\d+)\s*[번\.\)\-:]\s*[-]?\s*([①②③④⑤])')
# 질문에서 문항 번호 추출 — "5.", "5번", "문 5" 등
_QNUM_RE = re.compile(r'(?:문\s*)?(\d+)\s*[번\.\)]')

def _extract_answer_from_mobeom(mobeom_chunks: list, question: str) -> Optional[str]:
    """모범답안 청크에서 질문에 해당하는 정답을 추출. 찾으면 '③' 형태로 반환."""
    # 질문에서 문항 번호 추출
    qnum_match = _QNUM_RE.search(question)
    if not qnum_match:
        return None
    qnum = qnum_match.group(1)

    for chunk in mobeom_chunks:
        text = re.sub(r'\s+', ' ', chunk["content"])
        # "5. ③" 또는 "5번 ③" 패턴
        m = re.search(rf'{qnum}\s*[번\.\)\-:]\s*[-]?\s*([①②③④⑤])', text)
        if m:
            return m.group(1)
    return None

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

    # 2. 검색
    base_results = vector_store.search(question, TOP_K)

    # 출제원안↔모범답안 페어링 청크 추가
    paired = _get_paired_chunks(base_results, vector_store)
    # 모범답안 청크 분리 및 정답 사전 추출
    mobeom_chunks = [c for c in paired if "모범답안" in c["metadata"].get("filename", "")]
    confirmed_answer = _extract_answer_from_mobeom(mobeom_chunks, question) if mobeom_chunks else None

    if paired:
        seen_ids = {r["content"][:80] for r in base_results}
        base_results = base_results + [c for c in paired if c["content"][:80] not in seen_ids]

    if not base_results:
        yield f"data: {json.dumps({'type': 'text', 'content': '관련 문서를 찾을 수 없습니다. 먼저 문서를 업로드해주세요.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    context, sources = _build_context(base_results)
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # 모범답안 정답이 추출된 경우 유저 메시지에 명시적으로 주입
    answer_hint = ""
    if confirmed_answer:
        answer_hint = f"\n\n【모범답안 확정 정답】이 문제의 정답은 {confirmed_answer}번입니다. 반드시 이 번호를 최종 정답으로 사용하고, 분석 결과가 다르면 불일치 원인을 설명하세요."

    full_answer = ""
    try:
        for text in _claude_stream(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"=== 문서 ===\n{context}\n\n=== 질문 ===\n{question}{answer_hint}"}],
            max_tokens=2048,
        ):
            if text:
                text = _strip_foreign(text)
                full_answer += text
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'AI 오류: {str(e)}'})}\n\n"

    if full_answer:
        question_cache.set(question, full_answer)
    yield "data: [DONE]\n\n"
