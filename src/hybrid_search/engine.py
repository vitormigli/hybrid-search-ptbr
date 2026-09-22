"""Ties BM25 + dense retrieval + RRF fusion into one search engine, and loads
the real (local, free) embedding model."""

from functools import lru_cache

from hybrid_search.corpus import DOCUMENTS
from hybrid_search.retrievers import BM25Retriever, DenseRetriever, reciprocal_rank_fusion

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


class SearchEngine:
    def __init__(self, documents: list[dict] | None = None, dense_model=None):
        self.documents = documents if documents is not None else DOCUMENTS
        self.bm25 = BM25Retriever(self.documents)
        self.dense = DenseRetriever(self.documents, dense_model or load_embedding_model())
        self._by_id = {d["id"]: d for d in self.documents}

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=top_k)

    def search_dense(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.dense.search(query, top_k=top_k)

    def search_hybrid(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        bm25_ranked = self.bm25.search(query, top_k=len(self.documents))
        dense_ranked = self.dense.search(query, top_k=len(self.documents))
        fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
        return fused[:top_k]

    def get_document(self, doc_id: str) -> dict | None:
        return self._by_id.get(doc_id)
