import logging
import re
from typing import Any

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, VOYAGE_API_KEY

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
        if token in _STOPWORDS:
            continue
        stripped = token
        for p in _PARTICLES:
            if stripped.endswith(p) and len(stripped) - len(p) >= 2:
                stripped = stripped[:-len(p)]
                break
        if stripped in _STOPWORDS:
            continue
        if len(stripped) >= 2:
            result.append(stripped)
    return result


# Anthropic 공식 파트너 임베딩 모델 (한국어 포함 다국어)
EMBED_MODEL = "voyage-multilingual-2"


class VoyageEmbeddingFunction:
    """Voyage AI 임베딩 — document/query 모드 분리로 검색 품질 향상."""

    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 VOYAGE_API_KEY=<키> 를 추가하세요. "
                "키 발급: https://www.voyageai.com"
            )
        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        self._model = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        """문서 인덱싱용 — input_type='document'"""
        all_embeddings: list[list[float]] = []
        batch_size = 128  # Voyage API 배치 한도
        for i in range(0, len(input), batch_size):
            batch = list(input[i : i + batch_size])
            result = self._client.embed(batch, model=self._model, input_type="document")
            all_embeddings.extend(result.embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """검색 쿼리용 — input_type='query' (document 모드와 구분해 정밀도 향상)"""
        result = self._client.embed([query], model=self._model, input_type="query")
        return result.embeddings[0]


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef = VoyageEmbeddingFunction(model_name=EMBED_MODEL, api_key=VOYAGE_API_KEY)
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

        # 1. 벡터 유사도 검색 — query 전용 임베딩 사용
        query_embedding = self.ef.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
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
