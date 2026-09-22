"""BM25 (lexical) and dense (embedding) retrievers, plus Reciprocal Rank Fusion
to combine them. No LLM calls anywhere in this module."""

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._tokenized = [tokenize(d["text"]) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        doc_ids = [d["id"] for d in self.documents]
        ranked = sorted(zip(doc_ids, scores, strict=True), key=lambda x: -x[1])
        return ranked[:top_k]


class DenseRetriever:
    """Wraps a sentence-transformers model. Pass a pre-loaded `model` object
    (or any object exposing `.encode(list[str]) -> np.ndarray`) to avoid
    downloading model weights in tests."""

    def __init__(self, documents: list[dict], model):
        self.documents = documents
        self.model = model
        self._embeddings = np.asarray(model.encode([d["text"] for d in documents]))
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._normalized = self._embeddings / np.clip(norms, 1e-9, None)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = np.asarray(self.model.encode([query]))[0]
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)
        scores = self._normalized @ q_emb
        doc_ids = [d["id"] for d in self.documents]
        ranked = sorted(zip(doc_ids, scores, strict=True), key=lambda x: -x[1])
        return ranked[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Combines multiple ranked lists (id, score) using RRF: score = sum(1/(k+rank))
    across all rankings a document appears in. Higher fused score is better."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
