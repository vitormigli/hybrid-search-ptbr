"""Evaluates BM25-only, dense-only, and hybrid (RRF) retrieval on the gold query
set, computing recall@3, recall@5, and MRR for each. Writes results.json,
results.md, and a bar chart. No API calls — the embedding model runs locally."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_search.corpus import QUERIES
from hybrid_search.engine import SearchEngine
from hybrid_search.metrics import recall_at_k, reciprocal_rank

EVALS_DIR = Path(__file__).parent

MODES = {
    "bm25": lambda engine, q, k: engine.search_bm25(q, top_k=k),
    "dense": lambda engine, q, k: engine.search_dense(q, top_k=k),
    "hybrid": lambda engine, q, k: engine.search_hybrid(q, top_k=k),
}


def evaluate_mode(engine: SearchEngine, mode: str) -> dict:
    search_fn = MODES[mode]
    recall_3, recall_5, mrr = [], [], []
    for q in QUERIES:
        ranked = search_fn(engine, q["query"], max(5, len(engine.documents)))
        ranked_ids = [doc_id for doc_id, _ in ranked]
        recall_3.append(recall_at_k(ranked_ids, q["relevant_ids"], k=3))
        recall_5.append(recall_at_k(ranked_ids, q["relevant_ids"], k=5))
        mrr.append(reciprocal_rank(ranked_ids, q["relevant_ids"]))
    return {
        "recall@3": statistics.mean(recall_3),
        "recall@5": statistics.mean(recall_5),
        "mrr": statistics.mean(mrr),
        "n_queries": len(QUERIES),
    }


def write_chart(results: dict, output_path: Path) -> None:
    modes = list(results.keys())
    metrics = ["recall@3", "recall@5", "mrr"]
    x = range(len(modes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in modes]
        ax.bar([xi + i * width for xi in x], values, width, label=metric)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval quality: BM25 vs. dense vs. hybrid")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def write_report(results: dict, output_dir: Path) -> None:
    lines = ["# Retrieval Evaluation Results", ""]
    lines.append(f"{len(QUERIES)} gold queries against {15} documents.\n")
    lines.append("| Mode | Recall@3 | Recall@5 | MRR |")
    lines.append("|---|---|---|---|")
    for mode, m in results.items():
        lines.append(f"| {mode} | {m['recall@3']:.1%} | {m['recall@5']:.1%} | {m['mrr']:.3f} |")
    lines.append("")
    lines.append("![Retrieval quality](quality_by_mode.png)")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Loading embedding model (first run downloads ~470MB, cached after that)...")
    engine = SearchEngine()

    results = {mode: evaluate_mode(engine, mode) for mode in MODES}

    (EVALS_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_chart(results, EVALS_DIR / "quality_by_mode.png")
    write_report(results, EVALS_DIR)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
