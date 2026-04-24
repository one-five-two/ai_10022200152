"""
Part E: Evaluation — Free Version
---------------------------------
Compares RAG output against a pure local LLM baseline with no retrieval.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from src.pipeline.local_llm import generate_answer

logger = logging.getLogger(__name__)
EVAL_OUTPUT_DIR = Path("experiments")
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

ADVERSARIAL_QUERIES = [
    {
        "id": "adv_001",
        "type": "ambiguous",
        "query": "Who won?",
        "expected_behaviour": "System should acknowledge ambiguity rather than guessing one winner.",
        "hallucination_trap": "Pure LLM may guess a Ghana election winner without checking the dataset.",
    },
    {
        "id": "adv_002",
        "type": "misleading",
        "query": "Did the government reduce the education budget compared to 2024?",
        "expected_behaviour": "System should not compare unless 2024 evidence is retrieved.",
        "hallucination_trap": "Pure LLM may invent a 2024 baseline.",
    },
]


class RAGEvaluator:
    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.results: list[dict] = []

    def run_pure_llm(self, query: str) -> str:
        prompt = f"Answer this question about Ghana without retrieved context: {query}"
        return generate_answer(prompt)

    def evaluate_query(self, test_case: dict) -> dict:
        query = test_case["query"]
        result = {
            "id": test_case["id"],
            "type": test_case["type"],
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "pure_llm_response": self.run_pure_llm(query),
            "expected_behaviour": test_case["expected_behaviour"],
            "hallucination_trap": test_case["hallucination_trap"],
        }
        if self.rag_pipeline:
            rag_result = self.rag_pipeline.query(query)
            result["rag_response"] = rag_result["answer"]
            result["rag_scores"] = rag_result["similarity_scores"]
            result["rag_chunks_retrieved"] = len(rag_result["retrieved_chunks"])
        else:
            result["rag_response"] = "RAG pipeline not available"
        result["evaluation"] = self._auto_evaluate(result)
        self.results.append(result)
        return result

    def _auto_evaluate(self, result: dict) -> dict:
        rag_resp = result.get("rag_response", "").lower()
        llm_resp = result.get("pure_llm_response", "").lower()
        risky = ["won", "defeated", "increased", "decreased", "ghs", "billion", "2024"]
        llm_hall_count = sum(1 for kw in risky if kw in llm_resp)
        rag_hall_count = sum(1 for kw in risky if kw in rag_resp)
        rag_grounded = any(kw in rag_resp for kw in ["source", "dataset", "document", "page", "retrieved"])
        rag_ack_limits = any(kw in rag_resp for kw in ["insufficient", "ambiguous", "not contain", "limited"])
        return {
            "llm_potential_hallucinations": llm_hall_count,
            "rag_potential_hallucinations": rag_hall_count,
            "rag_is_grounded": rag_grounded,
            "rag_acknowledges_limits": rag_ack_limits,
            "rag_vs_llm_verdict": "RAG BETTER" if rag_grounded or rag_ack_limits or rag_hall_count <= llm_hall_count else "SIMILAR",
        }

    def run_all(self) -> list[dict]:
        for tc in ADVERSARIAL_QUERIES:
            self.evaluate_query(tc)
        return self.results

    def save_report(self, filename: str = "evaluation_report.json") -> None:
        path = EVAL_OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def print_comparison_table(self) -> None:
        print("\nEVALUATION REPORT: RAG vs Pure Local LLM")
        print("=" * 70)
        for result in self.results:
            print(f"\n{result['id']} ({result['type']}): {result['query']}")
            print("PURE LLM:", result.get("pure_llm_response", "N/A"))
            print("RAG:", result.get("rag_response", "N/A"))
            print("AUTO:", result.get("evaluation", {}))
