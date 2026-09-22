.PHONY: run eval test lint

run:
	uv run uvicorn hybrid_search.api:app --host 0.0.0.0 --port 8000 --reload

eval:
	uv run python evals/run_eval.py

test:
	uv run pytest

lint:
	uv run ruff check .
