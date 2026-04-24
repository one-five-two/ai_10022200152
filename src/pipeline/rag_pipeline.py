"""
Part D: Full RAG Pipeline — Free Local Version
----------------------------------------------
Complete flow:
User Query → Retrieval → Context Selection → Prompt → Local LLM → Response

This version uses a free HuggingFace text-generation model through
src.pipeline.local_llm. No Anthropic/Claude API key or paid credits are needed.
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime

from src.retrieval.embedder import embed_query
from src.retrieval.vector_store import VectorStore
from src.retrieval.query_expansion import retrieve_with_expansion
from src.prompts.prompt_builder import build_prompt
from src.pipeline.local_llm import generate_answer

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class RAGPipeline:
    """Central RAG orchestrator."""

    def __init__(
        self,
        vector_store: VectorStore,
        model: str = "google/flan-t5-small",
        top_k: int = 5,
        use_query_expansion: bool = True,
        prompt_version: str = "v2",
        max_tokens: int = 300,
    ):
        self.vector_store = vector_store
        self.model = model
        self.top_k = top_k
        self.use_query_expansion = use_query_expansion
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.session_log: list[dict] = []
        logger.info(
            "RAGPipeline initialised | model=%s top_k=%s expansion=%s prompt=%s",
            model, top_k, use_query_expansion, prompt_version,
        )

    def query(self, user_question: str) -> dict:
        """Runs retrieval, prompt construction, local generation and logging."""
        run_log = {
            "question": user_question,
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "model": self.model,
                "top_k": self.top_k,
                "expansion": self.use_query_expansion,
                "prompt_version": self.prompt_version,
            },
        }

        t0 = time.perf_counter()
        if self.use_query_expansion:
            chunks = retrieve_with_expansion(user_question, self.vector_store, top_k=self.top_k)
        else:
            q_vec = embed_query(user_question)
            chunks = self.vector_store.search(q_vec, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        scores = [c["similarity_score"] for c in chunks]
        run_log["retrieval"] = {
            "num_chunks": len(chunks),
            "similarity_scores": [round(s, 4) for s in scores],
            "top_chunk_preview": chunks[0]["text"][:200] if chunks else "",
            "latency_ms": round(retrieval_ms, 2),
        }

        t1 = time.perf_counter()
        prompt = build_prompt(user_question, chunks, version=self.prompt_version)
        prompt_ms = (time.perf_counter() - t1) * 1000
        run_log["prompt"] = {
            "version": self.prompt_version,
            "char_length": len(prompt),
            "latency_ms": round(prompt_ms, 2),
            "full_prompt": prompt,
        }

        t2 = time.perf_counter()
        answer = self._call_llm(prompt)
        llm_ms = (time.perf_counter() - t2) * 1000
        total_ms = retrieval_ms + prompt_ms + llm_ms

        run_log["generation"] = {
            "answer": answer,
            "latency_ms": round(llm_ms, 2),
            "answer_length": len(answer),
        }
        run_log["total_latency_ms"] = round(total_ms, 2)

        self.session_log.append(run_log)
        self._save_run_log(run_log)

        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "similarity_scores": scores,
            "final_prompt": prompt,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 2),
                "prompt_build": round(prompt_ms, 2),
                "llm": round(llm_ms, 2),
                "total": round(total_ms, 2),
            },
            "timestamp": run_log["timestamp"],
        }

    def _call_llm(self, prompt: str) -> str:
        """Uses a free local model, with extractive fallback if loading fails."""
        return generate_answer(prompt, model_name=self.model, max_new_tokens=self.max_tokens)

    def _save_run_log(self, run_log: dict) -> None:
        log_file = LOG_DIR / "pipeline_runs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            compact = {
                "timestamp": run_log["timestamp"],
                "question": run_log["question"],
                "scores": run_log["retrieval"]["similarity_scores"],
                "num_chunks": run_log["retrieval"]["num_chunks"],
                "answer_length": run_log["generation"]["answer_length"],
                "total_ms": run_log["total_latency_ms"],
            }
            f.write(json.dumps(compact) + "\n")

    def export_session_log(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.session_log, f, indent=2)
        logger.info("Session log exported to %s", filepath)
