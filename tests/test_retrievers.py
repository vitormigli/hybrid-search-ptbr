from hybrid_search.retrievers import (
    BM25Retriever,
    DenseRetriever,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = [
    {"id": "a", "text": "gato preto dorme no sofá"},
    {"id": "b", "text": "cachorro late no quintal"},
    {"id": "c", "text": "gato branco caça ratos"},
]


def test_tokenize_lowercases_and_strips_accents():
    assert tokenize("Não é fácil") == ["nao", "e", "facil"]


def test_bm25_ranks_exact_keyword_match_first():
    retriever = BM25Retriever(DOCS)
    ranked = retriever.search("gato")
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] in ("a", "c")
    assert "b" not in ids[:1]


def test_bm25_top_k_respected():
    retriever = BM25Retriever(DOCS)
    ranked = retriever.search("gato", top_k=2)
    assert len(ranked) == 2


def test_dense_retriever_returns_all_documents(fake_model):
    retriever = DenseRetriever(DOCS, fake_model)
    ranked = retriever.search("felino", top_k=3)
    assert {doc_id for doc_id, _ in ranked} == {"a", "b", "c"}


def test_dense_retriever_is_deterministic(fake_model):
    retriever = DenseRetriever(DOCS, fake_model)
    a = retriever.search("gato", top_k=3)
    b = retriever.search("gato", top_k=3)
    assert a == b


def test_rrf_favors_docs_ranked_high_in_both_lists():
    ranking_a = [("x", 5.0), ("y", 3.0), ("z", 1.0)]
    ranking_b = [("y", 9.0), ("x", 4.0), ("z", 0.5)]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    fused_ids = [doc_id for doc_id, _ in fused]
    assert fused_ids[0] in ("x", "y")
    assert fused_ids[-1] == "z"


def test_rrf_document_only_in_one_list_still_included():
    ranking_a = [("x", 1.0)]
    ranking_b = [("y", 1.0)]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert {doc_id for doc_id, _ in fused} == {"x", "y"}
