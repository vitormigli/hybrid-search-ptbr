from hybrid_search.corpus import DOCUMENTS, QUERIES


def test_all_query_relevant_ids_exist_in_corpus():
    doc_ids = {d["id"] for d in DOCUMENTS}
    for q in QUERIES:
        for rel_id in q["relevant_ids"]:
            assert rel_id in doc_ids, f"{rel_id} referenced by query but missing from corpus"


def test_documents_have_unique_ids():
    ids = [d["id"] for d in DOCUMENTS]
    assert len(ids) == len(set(ids))


def test_every_document_has_nonempty_text():
    for d in DOCUMENTS:
        assert d["text"].strip()
