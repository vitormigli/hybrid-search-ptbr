from hybrid_search.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["a", "b", "c"], ["z"], k=3) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b"], ["a", "z"], k=2) == 0.5


def test_recall_at_k_respects_k():
    assert recall_at_k(["z", "a", "b"], ["a"], k=1) == 0.0
    assert recall_at_k(["z", "a", "b"], ["a"], k=2) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["b", "c", "a"], ["a"]) == 1 / 3


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["b", "c"], ["a"]) == 0.0
