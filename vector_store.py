import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]):
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, total),
        )
        docs = []
        for i, doc in enumerate(results["documents"][0]):
            docs.append(
                {
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return docs

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
