import logging
import re
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

# 한국어 조사 목록 (길이 내림차순 — 긴 것 먼저 매칭)
_PARTICLES = sorted([
    '에서부터', '으로부터', '한테서', '으로서', '에게서',
    '이라고', '이라는', '라고', '라는', '이랑', '이며', '이고',
    '에서', '으로', '에게', '한테', '부터', '까지', '이라',
    '가', '이', '을', '를', '은', '는', '에', '의', '도', '만',
    '과', '와', '랑', '며', '고', '서', '게',
], key=len, reverse=True)

# 동사/형용사/의문사 — 검색 키워드로 부적합
_STOPWORDS = {
    '알려줘', '알려', '가르쳐줘', '가르쳐', '설명해줘', '설명해', '말해줘', '말해',
    '뭐야', '뭐가', '뭔가', '뭔', '무엇', '어떤', '어떻게', '왜', '언제',
    '어디', '누가', '누구', '무슨', '있어', '없어', '알아', '궁금',
    '했어', '했던', '하는', '한다', '했다', '한것', '한일', '할일',
    '됐어', '된것', '되는', '무엇인가', '합니까', '습니까',
}

def _extract_nouns(query: str) -> list[str]:
    """
    공백으로 분리 후 조사 제거, 동사/기능어 필터링 — 명사 위주 키워드만 반환.
    예) '최우가 한일 알려줘' → ['최우']
    """
    tokens = re.split(r'\s+', query.strip())
    result = []
    for token in tokens:
        if not token:
            continue
        # 불용어 전체 매칭
        if token in _STOPWORDS:
            continue
        # 조사 제거
        stripped = token
        for p in _PARTICLES:
            if stripped.endswith(p) and len(stripped) - len(p) >= 2:
                stripped = stripped[:-len(p)]
                break
        # 제거 후 불용어 재확인
        if stripped in _STOPWORDS:
            continue
        # 최소 2글자 이상인 경우만 키워드로 사용
        if len(stripped) >= 2:
            result.append(stripped)
    return result

# 한국어 포함 다국어 지원 모델
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        # 임베딩 모델이 변경된 경우 기존 컬렉션 삭제 후 재생성
        try:
            col = self.client.get_collection(name=COLLECTION_NAME)
            stored_model = col.metadata.get("embed_model", "")
            if stored_model != EMBED_MODEL:
                logger.info(f"임베딩 모델 변경 감지 ({stored_model} → {EMBED_MODEL}), 컬렉션 재생성")
                self.client.delete_collection(COLLECTION_NAME)
                raise ValueError("recreate")
            self.collection = col
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
            )

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]):
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        from document_processor import normalize_korean
        import re

        total = self.collection.count()
        if total == 0:
            return []

        query = normalize_korean(query)
        fetch = min(n_results * 3, total)

        # 1. 벡터 유사도 검색
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch,
        )
        vector_docs: dict[str, dict] = {}
        for i, doc in enumerate(results["documents"][0]):
            dist = results["distances"][0][i]
            vector_docs[results["ids"][0][i]] = {
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "vector_score": 1 - dist,
                "keyword_score": 0.0,
            }

        # 2. 명사 위주 키워드 추출 후 점수 계산
        keywords = _extract_nouns(query)
        query_nospace = re.sub(r'\s+', '', query)
        if keywords:
            for item in vector_docs.values():
                content_norm = normalize_korean(item["content"])
                content_nospace = re.sub(r'\s+', '', content_norm)
                hit = sum(1 for kw in keywords if kw in content_norm or kw in content_nospace)
                if len(query_nospace) >= 4 and query_nospace in content_nospace:
                    hit += len(keywords) * 0.5
                item["keyword_score"] = min(hit / len(keywords), 1.5)

        # 3. 하이브리드 점수로 재정렬 (벡터 60% + 키워드 40%)
        scored = sorted(
            vector_docs.values(),
            key=lambda x: x["vector_score"] * 0.6 + x["keyword_score"] * 0.4,
            reverse=True,
        )

        return [
            {"content": d["content"], "metadata": d["metadata"],
             "distance": 1 - d["vector_score"]}
            for d in scored[:n_results]
        ]

    def get_file_chunks(self, filename: str) -> list[dict]:
        """특정 파일의 모든 청크 반환 (모범답안 전체 로딩용)"""
        results = self.collection.get(where={"filename": filename})
        if not results["documents"]:
            return []
        return [
            {"content": doc, "metadata": meta, "distance": 0.0}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def delete_document(self, filename: str):
        results = self.collection.get(where={"filename": filename})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for {filename}")

    def list_documents(self) -> list[dict]:
        results = self.collection.get()
        if not results["metadatas"]:
            return []
        seen: dict[str, int] = {}
        for meta in results["metadatas"]:
            name = meta.get("filename", "")
            total = meta.get("total_chunks", 0)
            seen[name] = total
        return [{"filename": k, "chunks": v} for k, v in seen.items()]

    def count(self) -> int:
        return self.collection.count()


vector_store = VectorStore()
