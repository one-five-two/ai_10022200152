"""
Part C: Prompt Engineering
---------------------------
Designs and implements the RAG prompt template.

KEY ANTI-HALLUCINATION TECHNIQUES:
    1. "Only use the provided context" instruction
    2. Explicit fallback: if context is insufficient, say so
    3. Citation anchoring: ask the model to reference source/page
    4. Role framing: positions the model as a factual analyst, not a creative writer
    5. Negative instruction: "Do not add information beyond what is given"

WHY THIS MATTERS:
    LLMs hallucinate most when asked about specific facts (numbers, names,
    percentages) that they partially recall from training data. By grounding
    the model in retrieved context AND explicitly forbidding extrapolation,
    we reduce hallucination rates significantly.

    Experiment run during development (documented in experiments/):
        - Without grounding instruction → model invented 2024 budget figures
          from its training data, ignoring the retrieved 2025 values.
        - With grounding instruction → model correctly cited retrieved numbers.

CONTEXT WINDOW MANAGEMENT:
    The local free generator has a smaller context window. We could dump
    everything in, but that:
        a) increases latency and cost
        b) causes "lost in the middle" failures (Liu et al., 2023) where
           the model ignores chunks in the middle of a long context
    
    Strategy: select top-k chunks by similarity, then apply a hard token
    budget of MAX_CONTEXT_TOKENS. Chunks are truncated from the bottom
    (lowest similarity) if the budget is exceeded.
"""

import logging

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 3000    # ~2400 words; leaves room for instructions + response
CHARS_PER_TOKEN = 4           # rough approximation (English text averages ~4 chars/token)
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN


# ─────────────────────────────────────────────────────────────
# Template variants (for ablation experiment in Part E)
# ─────────────────────────────────────────────────────────────

PROMPT_V1_BASIC = """\
Answer the following question using the context below.

Context:
{context}

Question: {question}
Answer:"""

# This is the prompt used in production (V2). It adds:
#   - explicit role
#   - grounding instruction
#   - citation requirement
#   - fallback instruction
#   - format guidance

PROMPT_V2_GROUNDED = """\
You are a factual research assistant specialising in Ghana's politics and public finance.
Your job is to answer questions strictly based on the provided context passages.

RULES:
1. Only use information from the context below. Do not rely on prior knowledge.
2. If the context does not contain enough information to answer, say:
   "The provided documents do not contain sufficient information to answer this question."
3. When citing facts, mention the source (e.g., "According to the 2025 Budget document, page X..." 
   or "According to the election dataset...").
4. Do not speculate, estimate, or extrapolate beyond what is explicitly stated.
5. Keep your response concise and factual.

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""


# Experiment V3: Chain-of-thought variant (tested in Part E)
PROMPT_V3_COT = """\
You are a factual research assistant for Ghana's elections and budget data.

Think step by step:
1. Identify which parts of the context are relevant to the question.
2. Extract the key facts needed to answer.
3. Compose a clear, grounded answer with citations.

RULES:
- Only use the context. Explicitly say if context is insufficient.
- Do not hallucinate statistics, names, or dates.

---
CONTEXT:
{context}
---

QUESTION: {question}

Let me work through this step by step:"""


# ─────────────────────────────────────────────────────────────
# Context building
# ─────────────────────────────────────────────────────────────

def build_context_string(chunks: list[dict]) -> str:
    """
    Converts retrieved chunks into a formatted context block.

    Each chunk gets a clear separator and source attribution so the
    model can reference them accurately.

    Context window management:
        We enforce MAX_CONTEXT_CHARS by truncating lower-ranked chunks.
        Since chunks are already sorted by similarity (highest first),
        truncating from the bottom minimises information loss.
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        page = chunk.get("metadata", {}).get("page", "")
        score = chunk.get("similarity_score", 0.0)

        # Build source label
        if source == "budget_pdf" and page:
            label = f"[Source: Ghana 2025 Budget, Page {page}]"
        elif source == "election_csv":
            label = "[Source: Ghana Election Dataset]"
        else:
            label = f"[Source: {source}]"

        # Chunk header + text
        chunk_str = f"Passage {i+1} {label} (relevance: {score:.3f}):\n{chunk['text']}\n"

        # Check budget
        if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Context budget exceeded at chunk {i+1}. "
                f"Truncating to {i} chunks ({total_chars} chars)."
            )
            break

        context_parts.append(chunk_str)
        total_chars += len(chunk_str)

    context = "\n---\n".join(context_parts)
    logger.info(f"Context built: {len(context_parts)} chunks, {total_chars} chars")
    return context


def build_prompt(
    question: str,
    chunks: list[dict],
    version: str = "v2",
) -> str:
    """
    Builds the full prompt string from retrieved chunks and a question.

    version: "v1" | "v2" | "v3"  — allows ablation experiments (Part E)
    """
    context = build_context_string(chunks)

    templates = {
        "v1": PROMPT_V1_BASIC,
        "v2": PROMPT_V2_GROUNDED,
        "v3": PROMPT_V3_COT,
    }
    template = templates.get(version, PROMPT_V2_GROUNDED)
    prompt = template.format(context=context, question=question)

    logger.debug(f"Prompt built ({version}): {len(prompt)} chars")
    return prompt


# ─────────────────────────────────────────────────────────────
# Prompt experiment helper
# ─────────────────────────────────────────────────────────────

def run_prompt_experiment(
    question: str,
    chunks: list[dict],
    llm_fn,           # callable: prompt_str → response_str
) -> dict:
    """
    Runs the same question through all three prompt versions and
    returns the results for comparison.

    Used in experiments/prompt_ablation.py (Part E).
    """
    results = {}
    for version in ["v1", "v2", "v3"]:
        prompt = build_prompt(question, chunks, version=version)
        response = llm_fn(prompt)
        results[version] = {
            "prompt_length": len(prompt),
            "response": response,
        }
        logger.info(f"Prompt {version} response length: {len(response)} chars")

    return results
