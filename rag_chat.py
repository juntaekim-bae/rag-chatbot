import json
import logging
from typing import Iterator

from groq import Groq

from config import GROQ_API_KEY, MODEL, TOP_K

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """당신은 제공된 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

규칙:
1. 반드시 제공된 문서 컨텍스트에 근거하여 답변하세요.
2. 문서에 없는 내용은 "해당 내용은 문서에 없습니다"라고 명확히 말하세요.
3. 답변은 명확하고 구조적으로 작성하세요.
4. 질문 언어에 맞춰 답변하세요 (한국어 질문 → 한국어 답변)."""


def _build_context(results: list) -> tuple[str, list[str]]:
    sources: list[str] = []
    parts: list[str] = []
    for r in results:
        fname = r["metadata"].get("filename", "Unknown")
        if fname not in sources:
            sources.append(fname)
        parts.append(f"[출처: {fname}]\n{r['content']}")
    return "\n\n---\n\n".join(parts), sources


def chat_stream(question: str, vector_store) -> Iterator[str]:
    results = vector_store.search(question, n_results=TOP_K)

    if not results:
        yield f"data: {json.dumps({'type': 'text', 'content': '관련 문서를 찾을 수 없습니다. 먼저 문서를 업로드해주세요.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    context, sources = _build_context(results)

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    user_message = f"""아래 문서 내용을 참고하여 질문에 답변해주세요.

=== 문서 컨텍스트 ===
{context}

=== 질문 ===
{question}"""

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
                yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"
    except Exception as e:
        logger.error(f"Groq API 오류: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'AI 오류: {str(e)}'})}\n\n"

    yield "data: [DONE]\n\n"
