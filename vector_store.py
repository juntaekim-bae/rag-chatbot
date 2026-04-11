import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

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
                "vector_score": 1 - dist,  # 거리 → 유사도
                "keyword_score": 0.0,
            }

        # 2. 키워드 점수: 띄어쓰기 무시하고 키워드 포함 여부 확인
        keywords = [w for w in re.split(r'\s+', query) if len(w) >= 2]
        query_nospace = re.sub(r'\s+', '', query)  # 공백 제거 쿼리
        if keywords:
            for doc_id, item in vector_docs.items():
                content_norm = normalize_korean(item["content"])
                content_nospace = re.sub(r'\s+', '', content_norm)
                # 단어 단위 매칭 + 공백 제거 후 매칭
                hit = sum(1 for kw in keywords if kw in content_norm or kw in content_nospace)
                # 쿼리 전체(공백 제거)가 본문에 있으면 보너스
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
