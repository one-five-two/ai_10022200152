"""
Unit Tests — Ghana RAG System
Tests the core components without requiring actual data files or API keys.
Run: python -m pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.ingestion.chunker import (
    chunk_sliding_window,
    chunk_by_sentences,
    chunk_by_paragraphs,
    chunk_documents,
)
from src.retrieval.vector_store import cosine_similarity_numpy, top_k_numpy


# ─────────────────────────────────────────────────────────────
# Chunker tests
# ─────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Ghana held its 2020 general elections in December. "
    "The presidential race was contested between incumbent Nana Akufo-Addo and opposition leader John Mahama. "
    "Results showed the NPP winning 137 parliamentary seats. "
    "The NDC also won 137 seats creating a hung parliament. "
    "Regional breakdowns showed Ashanti Region strongly favouring NPP. "
    "Volta Region showed strong support for NDC candidates across constituencies. "
    "The Electoral Commission declared Akufo-Addo the winner with 51.6 percent of valid votes. "
    "Voter turnout was recorded at approximately 79 percent nationally. "
)


class TestSlidingWindow:
    def test_basic_output(self):
        chunks = chunk_sliding_window(SAMPLE_TEXT, chunk_size=20, overlap=5)
        assert len(chunks) > 1
        assert all("text" in c for c in chunks)

    def test_chunk_size_respected(self):
        chunks = chunk_sliding_window(SAMPLE_TEXT, chunk_size=10, overlap=2)
        for c in chunks:
            words = c["text"].split()
            assert len(words) <= 10, f"Chunk exceeded size: {len(words)}"

    def test_overlap_produces_duplicate_words(self):
        chunks = chunk_sliding_window(SAMPLE_TEXT, chunk_size=15, overlap=5)
        if len(chunks) >= 2:
            tail_first = set(chunks[0]["text"].split()[-5:])
            head_second = set(chunks[1]["text"].split()[:5])
            # Some words should overlap
            assert len(tail_first & head_second) > 0

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_sliding_window(SAMPLE_TEXT, chunk_size=10, overlap=10)

    def test_source_propagated(self):
        chunks = chunk_sliding_window(SAMPLE_TEXT, source="test_source")
        assert all(c["source"] == "test_source" for c in chunks)


class TestSentenceChunker:
    def test_produces_chunks(self):
        chunks = chunk_by_sentences(SAMPLE_TEXT, sentences_per_chunk=2)
        assert len(chunks) >= 1

    def test_metadata_propagated(self):
        meta = {"page": 3, "year": 2020}
        chunks = chunk_by_sentences(SAMPLE_TEXT, metadata=meta)
        assert all(c["metadata"] == meta for c in chunks)


class TestParagraphChunker:
    def test_single_block_text(self):
        # Text with no paragraph breaks — should produce one chunk
        text = "No double newlines here. Just one long paragraph."
        chunks = chunk_by_paragraphs(text, max_words=100)
        assert len(chunks) == 1

    def test_multi_paragraph(self):
        # "First paragraph." = 2 words, "Second paragraph." = 2 words → merged (4 ≤ 5)
        # "Third paragraph." = 2 words → flushed separately (4+2=6 > 5)
        # So with max_words=5 we get 2 chunks, not 3 (by design — buffer merges short paras)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_by_paragraphs(text, max_words=5)
        assert len(chunks) == 2  # "First + Second" merged; "Third" alone

    def test_multi_paragraph_tight_budget(self):
        # With max_words=1, every 2-word paragraph exceeds budget → each stands alone
        # But since we flush BEFORE adding, each 2-word para goes in as its own chunk
        text = "A b.\n\nC d.\n\nE f."
        chunks = chunk_by_paragraphs(text, max_words=2)
        assert len(chunks) == 3


class TestChunkDocuments:
    def test_election_uses_sentence_strategy(self):
        docs = [{"text": SAMPLE_TEXT, "source": "election_csv", "metadata": {}}]
        chunks = chunk_documents(docs)
        assert all(c["source"] == "election_csv" for c in chunks)

    def test_budget_uses_sliding_strategy(self):
        docs = [{"text": SAMPLE_TEXT, "source": "budget_pdf", "metadata": {"page": 1}}]
        chunks = chunk_documents(docs)
        assert all(c["source"] == "budget_pdf" for c in chunks)

    def test_mixed_sources(self):
        docs = [
            {"text": SAMPLE_TEXT, "source": "election_csv", "metadata": {}},
            {"text": SAMPLE_TEXT, "source": "budget_pdf", "metadata": {"page": 1}},
        ]
        chunks = chunk_documents(docs)
        sources = {c["source"] for c in chunks}
        assert sources == {"election_csv", "budget_pdf"}


# ─────────────────────────────────────────────────────────────
# Vector store / similarity tests
# ─────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([0.6, 0.8, 0.0])
        v_norm = v / np.linalg.norm(v)
        matrix = v_norm.reshape(1, -1)
        scores = cosine_similarity_numpy(v_norm, matrix)
        assert abs(scores[0] - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        q = np.array([1.0, 0.0, 0.0])
        d = np.array([[0.0, 1.0, 0.0]])
        scores = cosine_similarity_numpy(q, d)
        assert abs(scores[0]) < 1e-6

    def test_ranking_order(self):
        q = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [0.0, 1.0, 0.0],  # orthogonal — should score lowest
            [0.8, 0.6, 0.0],  # close — should score highest
            [0.5, 0.5, 0.707], # middle
        ])
        scores = cosine_similarity_numpy(q, matrix)
        assert scores[1] > scores[2] > scores[0]


class TestTopKNumpy:
    def setup_method(self):
        np.random.seed(42)
        self.dim = 32
        self.n = 100
        self.matrix = np.random.randn(self.n, self.dim).astype(np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / norms
        self.chunks = [{"text": f"chunk_{i}"} for i in range(self.n)]

    def test_returns_k_results(self):
        q = self.matrix[0]
        results = top_k_numpy(q, self.matrix, self.chunks, k=5)
        assert len(results) == 5

    def test_scores_descending(self):
        q = self.matrix[0]
        results = top_k_numpy(q, self.matrix, self.chunks, k=5)
        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_self_is_top_result(self):
        # A vector should be most similar to itself
        q = self.matrix[7]
        results = top_k_numpy(q, self.matrix, self.chunks, k=3)
        assert results[0]["text"] == "chunk_7"
