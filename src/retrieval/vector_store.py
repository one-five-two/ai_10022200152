"""
Part B: Vector Store + Retrieval
--------------------------------
Manual vector retrieval wrapper using FAISS when available, with a NumPy fallback.

Cosine similarity is implemented directly. Because embeddings are L2-normalised,
cosine similarity is equivalent to an inner product.
"""

import pickle
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception as exc:
        logger.warning("FAISS unavailable; falling back to NumPy vector search: %s", exc)
        return None


class VectorStore:
    """Stores embeddings and matching chunk metadata for top-k retrieval."""

    def __init__(self, embedding_dim: int = 384, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        self.chunks: list[dict] = []
        self.use_faiss = use_faiss
        self.faiss = _try_import_faiss() if use_faiss else None
        self.index = self.faiss.IndexFlatIP(embedding_dim) if self.faiss else None
        self._matrix: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        if self.index is not None:
            return self.index.ntotal
        return 0 if self._matrix is None else self._matrix.shape[0]

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings but {len(chunks)} chunks")
        vecs = embeddings.astype(np.float32)
        if self.index is not None:
            self.index.add(vecs)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        self.chunks.extend(chunks)
        logger.info("VectorStore now has %s vectors", self.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.ntotal == 0:
            raise RuntimeError("Vector store is empty. Run ingestion first.")
        top_k = min(top_k, self.ntotal)
        q = query_vec.reshape(1, -1).astype(np.float32)

        if self.index is not None:
            distances, indices = self.index.search(q, top_k)
            pairs = zip(distances[0], indices[0])
        else:
            scores = cosine_similarity_numpy(query_vec, self._matrix)
            indices = np.argsort(scores)[::-1][:top_k]
            pairs = [(scores[i], i) for i in indices]

        results = []
        for score, idx in pairs:
            if idx == -1:
                continue
            chunk = self.chunks[int(idx)].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "embedding_dim": self.embedding_dim,
            "chunks": self.chunks,
            "matrix": self._matrix,
            "use_faiss": self.index is not None,
        }
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(metadata, f)

        if self.index is not None:
            self.faiss.write_index(self.index, str(path / "index.faiss"))

        logger.info("VectorStore saved to %s", directory)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        path = Path(directory)
        with open(path / "chunks.pkl", "rb") as f:
            obj = pickle.load(f)

        # Backward compatibility if chunks.pkl contains just a list
        if isinstance(obj, list):
            chunks = obj
            matrix = None
            embedding_dim = 384
        else:
            chunks = obj["chunks"]
            matrix = obj.get("matrix")
            embedding_dim = obj.get("embedding_dim", 384)

        store = cls(embedding_dim=embedding_dim)
        store.chunks = chunks
        store._matrix = matrix

        if store.faiss and (path / "index.faiss").exists():
            store.index = store.faiss.read_index(str(path / "index.faiss"))
        elif matrix is not None:
            store.index = None

        logger.info("VectorStore loaded: %s vectors", store.ntotal)
        return store


def cosine_similarity_numpy(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return d_norms @ q_norm


def top_k_numpy(query_vec: np.ndarray, embeddings_matrix: np.ndarray, chunks: list[dict], k: int = 5) -> list[dict]:
    scores = cosine_similarity_numpy(query_vec, embeddings_matrix)
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        chunk = chunks[int(idx)].copy()
        chunk["similarity_score"] = float(scores[idx])
        results.append(chunk)
    return results
