"""FastAPI service exposing GET /search over the local corpus."""

from fastapi import FastAPI, Query

from hybrid_search.engine import SearchEngine

app = FastAPI(title="Hybrid Search (pt-BR)")
_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = 5, mode: str = "hybrid") -> dict:
    engine = get_engine()
    searchers = {
        "bm25": engine.search_bm25,
        "dense": engine.search_dense,
        "hybrid": engine.search_hybrid,
    }
    ranked = searchers[mode](q, top_k=top_k)
    return {
        "query": q,
        "mode": mode,
        "results": [
            {"id": doc_id, "score": float(score), **engine.get_document(doc_id)}
            for doc_id, score in ranked
        ],
    }
