"""
run_eval.py — Free Evaluation Runner
------------------------------------
Run after ingestion:
    python run_eval.py --mode all
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.local_llm import generate_answer
from src.retrieval.vector_store import VectorStore
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_store"
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_pipeline() -> RAGPipeline:
    if not Path(VECTOR_STORE_DIR).exists():
        logger.error("Vector store not found. Run ingestion first.")
        sys.exit(1)
    store = VectorStore.load(VECTOR_STORE_DIR)
    return RAGPipeline(store, top_k=5, use_query_expansion=True, prompt_version="v2")


def run_adversarial(pipeline: RAGPipeline) -> None:
    evaluator = RAGEvaluator(rag_pipeline=pipeline)
    evaluator.run_all()
    evaluator.print_comparison_table()
    evaluator.save_report("evaluation_adversarial.json")


def run_consistency(pipeline: RAGPipeline) -> None:
    test_queries = [
        "What is the total revenue target for Ghana in 2025?",
        "How many constituencies are in the Greater Accra Region?",
    ]
    results = []
    for query in test_queries:
        answers = []
        for _ in range(3):
            answers.append(pipeline.query(query)["answer"])
        results.append({"query": query, "answers": answers, "all_identical": len(set(answers)) == 1})
    path = OUTPUT_DIR / "consistency_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(), "results": results}, f, indent=2)
    print(f"Consistency report saved to {path}")


def run_rag_vs_llm(pipeline: RAGPipeline) -> None:
    comparison_queries = [
        "What is Ghana's inflation target for 2025?",
        "Which party won the most parliamentary seats in the 2020 elections?",
        "What is the total expenditure in the 2025 budget?",
    ]
    rows = []
    for q in comparison_queries:
        llm_resp = generate_answer(f"Answer this about Ghana without retrieved context: {q}")
        rag_result = pipeline.query(q)
        rows.append({
            "query": q,
            "pure_local_llm": llm_resp,
            "rag": rag_result["answer"],
            "rag_top_score": round(rag_result["similarity_scores"][0], 4) if rag_result["similarity_scores"] else 0,
        })
    md_lines = ["# RAG vs Pure Local LLM Comparison", f"Generated: {datetime.utcnow().isoformat()}", ""]
    for row in rows:
        md_lines += [f"## {row['query']}", f"Top RAG score: {row['rag_top_score']}", "", "Pure local LLM:", row["pure_local_llm"], "", "RAG:", row["rag"], "", "---", ""]
    (OUTPUT_DIR / "rag_vs_llm_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    with open(OUTPUT_DIR / "rag_vs_llm_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(), "comparisons": rows}, f, indent=2)
    print("RAG vs LLM comparison saved in experiments/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghana RAG Evaluation Runner")
    parser.add_argument("--mode", choices=["adversarial", "consistency", "rag_vs_llm", "all"], default="all")
    args = parser.parse_args()
    pipeline = load_pipeline()
    if args.mode in ("adversarial", "all"):
        run_adversarial(pipeline)
    if args.mode in ("consistency", "all"):
        run_consistency(pipeline)
    if args.mode in ("rag_vs_llm", "all"):
        run_rag_vs_llm(pipeline)
    print("Evaluation complete.")
