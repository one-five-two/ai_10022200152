"""
Local free text generation for the RAG pipeline.
------------------------------------------------
This module replaces paid Claude/Anthropic calls with a free local HuggingFace
model. It is not a prebuilt RAG framework: retrieval, chunking, embeddings,
prompting and vector search remain manually implemented in this project.
"""

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "google/flan-t5-small"


@lru_cache(maxsize=1)
def _load_generator(model_name: str = DEFAULT_LOCAL_MODEL):
    """Lazy-load a small local instruction model."""
    try:
        from transformers import pipeline
        logger.info("Loading local generation model: %s", model_name)
        return pipeline("text2text-generation", model=model_name)
    except Exception as exc:
        logger.warning("Could not load local model (%s). Falling back to extractive answer.", exc)
        return None


def _extract_context(prompt: str) -> str:
    match = re.search(r"CONTEXT:\s*(.*?)\n---\s*\nQUESTION:", prompt, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"Context:\s*(.*?)\n\s*Question:", prompt, flags=re.S | re.I)
    return match.group(1).strip() if match else ""


def extractive_answer(prompt: str, max_chars: int = 1400) -> str:
    """
    Safe no-cost fallback: returns a concise grounded answer using retrieved context.
    This avoids hallucination when the local model is unavailable or gives weak output.
    """
    context = _extract_context(prompt)
    if not context:
        return "The provided documents do not contain sufficient information to answer this question."

    passages = re.split(r"\n---\n", context)
    selected = []
    for p in passages[:3]:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        selected.append(p[:450])

    if not selected:
        return "The provided documents do not contain sufficient information to answer this question."

    answer = (
        "Based on the retrieved documents, the most relevant evidence is:\n\n"
        + "\n\n".join(f"- {s}" for s in selected)
        + "\n\nI have limited the answer to the retrieved context to avoid adding unsupported information."
    )
    return answer[:max_chars]


def generate_answer(prompt: str, model_name: str = DEFAULT_LOCAL_MODEL, max_new_tokens: int = 300) -> str:
    """Generate an answer using a free local model, with safe extractive fallback."""
    generator = _load_generator(model_name)
    if generator is None:
        return extractive_answer(prompt)

    compact_prompt = prompt[-6000:]  # keep prompt small enough for FLAN-T5
    try:
        output = generator(
            compact_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            truncation=True,
        )
        text = output[0].get("generated_text", "").strip()
        if not text or len(text) < 20:
            return extractive_answer(prompt)
        return text
    except Exception as exc:
        logger.warning("Local generation failed: %s", exc)
        return extractive_answer(prompt)
