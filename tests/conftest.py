"""A deterministic, hash-based fake embedding model so tests never need to
download real model weights or touch the network."""

import hashlib

import numpy as np
import pytest


class FakeEmbeddingModel:
    dim = 32

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).hexdigest()
            rng = np.random.default_rng(int(digest, 16) % (2**32))
            vectors.append(rng.normal(size=self.dim))
        return np.array(vectors)


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()
