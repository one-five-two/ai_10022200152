"""
Part B: Embedding Pipeline
---------------------------
Converts text chunks into dense vector representations.

MODEL CHOICE: all-MiniLM-L6-v2
    - 384-dimensional embeddings (compact → fast FAISS search)
    - 256-token sequence limit (our 400-word chunks average ~300 tokens,
      so we truncate at the model level — acceptable for this domain)
    - Strong performance on semantic similarity benchmarks (MTEB)
    - Runs on CPU at ~500 chunks/sec — no GPU required for this dataset size
    - Alternative: OpenAI text-embedding-3-small gives higher quality but
      requires an API key and adds latency/cost per ingestion run.

WHY NOT OpenAI embeddings?
    - Cost: $0.00002/1K tokens. For 50K chunks that's ~$1 per full re-index.
      During development we re-index dozens of times → sentence-transformers
      is free and reproducible offline.
    - Latency: API calls add 100–500ms per batch; local model is <5ms/batch.
    - Offline: local model works without internet access (important for
      deployment in low-connectivity environments like rural Ghana).
"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Global model instance (loaded once, reused across calls)
_model: SentenceTransformer | None = None
EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """Lazy-loads the model the first time it's needed."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Encodes a list of strings into a 2D numpy array of shape (N, 384).

    Batching rationale:
        batch_size=64 is a practical sweet spot for CPU inference.
        Larger batches hit memory limits; smaller batches underutilise
        the model's parallelism.

    Normalisation:
        We L2-normalise every vector so that cosine similarity reduces to
        a simple dot product. This means FAISS IndexFlatIP (inner product)
        gives the same ranking as cosine similarity — which is faster.
    """
    model = get_embedding_model()

    logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # ← L2 normalise
        convert_to_numpy=True,
    )

    logger.info(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> tuple[np.ndarray, list[dict]]:
    """
    Convenience wrapper that extracts texts from chunk dicts,
    embeds them, and returns (embeddings_matrix, chunks_list).

    The chunks list acts as our metadata store — index i in the
    numpy array corresponds to index i in the list.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, batch_size=batch_size)
    return embeddings, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string. Returns shape (384,).

    We apply the same L2 normalisation as during ingestion so the
    dot-product similarity is directly comparable.
    """
    model = get_embedding_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0]  # shape (384,)
