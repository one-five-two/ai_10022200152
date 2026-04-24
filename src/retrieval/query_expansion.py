"""
Part B: Advanced Retrieval Feature — Free Query Expansion
---------------------------------------------------------
This version does not use any paid API. It manually expands common terms in the
Ghana elections and budget domain, then retrieves and merges results.
"""

import logging
from src.retrieval.embedder import embed_query
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

DOMAIN_SYNONYMS = {
    "school": ["education", "GES", "Ministry of Education"],
    "schools": ["education", "GES", "Ministry of Education"],
    "health": ["healthcare", "hospital", "Ministry of Health"],
    "tax": ["revenue", "levy", "taxation"],
    "money": ["allocation", "expenditure", "budget"],
    "spend": ["expenditure", "allocation", "budget"],
    "government spending": ["public expenditure", "budget allocation"],
    "npp": ["New Patriotic Party"],
    "ndc": ["National Democratic Congress"],
    "votes": ["valid votes", "total votes", "polling results"],
    "won": ["winner", "victory", "elected"],
    "constituency": ["parliamentary constituency", "electoral area"],
}


def expand_query(query: str, n_expansions: int = 2) -> list[str]:
    """Rule-based query expansion for the Ghana election/budget domain."""
    q_lower = query.lower()
    expansions = []

    for key, alternatives in DOMAIN_SYNONYMS.items():
        if key in q_lower:
            for alt in alternatives:
                expansions.append(query + " " + alt)
                if len(expansions) >= n_expansions:
                    break
        if len(expansions) >= n_expansions:
            break

    if not expansions:
        expansions = [
            query + " Ghana elections budget",
            query + " official data source",
        ][:n_expansions]

    logger.info("Query expanded: %r -> %s", query, expansions)
    return [query] + expansions[:n_expansions]


def retrieve_with_expansion(query: str, vector_store: VectorStore, top_k: int = 5, n_expansions: int = 2) -> list[dict]:
    expanded_queries = expand_query(query, n_expansions=n_expansions)
    best_results: dict[str, dict] = {}

    for q in expanded_queries:
        q_vec = embed_query(q)
        candidates = vector_store.search(q_vec, top_k=top_k * 2)
        for chunk in candidates:
            key = chunk["text"][:250]
            existing = best_results.get(key)
            if existing is None or chunk["similarity_score"] > existing["similarity_score"]:
                chunk["retrieved_by_query"] = q
                best_results[key] = chunk

    return sorted(best_results.values(), key=lambda x: x["similarity_score"], reverse=True)[:top_k]
