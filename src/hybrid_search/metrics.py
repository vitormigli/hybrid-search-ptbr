"""Retrieval evaluation metrics: recall@k and MRR."""


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = set(ranked_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
