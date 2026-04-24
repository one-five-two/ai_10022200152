"""Prompt ablation experiment using the free local model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.local_llm import generate_answer
from src.prompts.prompt_builder import build_prompt
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import embed_query


def main():
    store = VectorStore.load("vector_store")
    question = "What are Ghana's key revenue targets for 2025?"
    chunks = store.search(embed_query(question), top_k=5)
    out = []
    for version in ["v1", "v2", "v3"]:
        prompt = build_prompt(question, chunks, version=version)
        out.append(f"## Prompt {version}\n\n{generate_answer(prompt)}\n")
    Path("experiments/prompt_ablation_results.md").write_text("\n---\n".join(out), encoding="utf-8")
    print("Saved experiments/prompt_ablation_results.md")

if __name__ == "__main__":
    main()
