# Project instructions for Claude

This project follows the rules of the portfolio master plan:

1. No client code or data — synthetic corpus of internal policies (see
   `src/hybrid_search/corpus.py`).
2. No committed secrets — this project needs none (no paid API is called;
   `gitleaks` still runs on pre-commit and in CI as a safety net).
3. Every project reports numeric evaluation metrics — see `evals/results.md`.
4. Everything runs with a single command: `docker compose up` or `make run`.
5. README in English, with a short "Resumo em português" section at the end.
   Header banner + badges matching the other portfolio repos.
6. Small, descriptive commits using Conventional Commits.
7. Prefer simplicity — `rank-bm25` and `sentence-transformers` directly, no
   vector-database or orchestration framework.

## Layout

- `src/hybrid_search/corpus.py` — synthetic documents + gold query set.
- `src/hybrid_search/retrievers.py` — BM25, dense, and RRF fusion.
- `src/hybrid_search/engine.py` — ties retrievers together, loads the real
  (local, free) embedding model.
- `evals/run_eval.py` — compares BM25-only / dense-only / hybrid on recall@k
  and MRR; the embedding model download is the only network call this project
  ever makes, and it's free.
