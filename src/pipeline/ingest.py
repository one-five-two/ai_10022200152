"""
Part D: Ingestion Script
-------------------------
Runs the full data engineering + indexing pipeline.

Run this ONCE before starting the Streamlit app:
    python -m src.pipeline.ingest --csv data/ghana_elections.csv --pdf data/ghana_budget_2025.pdf

Output:
    vector_store/index.faiss
    vector_store/chunks.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.data_loader import (
    load_election_csv,
    election_rows_to_text,
    load_budget_pdf,
)
from src.ingestion.chunker import chunk_documents
from src.retrieval.embedder import embed_chunks
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def ingest(csv_path: str, pdf_path: str, output_dir: str = "vector_store") -> None:
    """
    Full ingestion pipeline:
        Load → Clean → Chunk → Embed → Index → Save
    """
    all_documents = []

    # ── Load Election CSV ─────────────────────────────────────────────────
    if csv_path and Path(csv_path).exists():
        logger.info(f"Loading election CSV: {csv_path}")
        df = load_election_csv(csv_path)
        docs = election_rows_to_text(df)
        logger.info(f"  → {len(docs)} election row documents")
        all_documents.extend(docs)
    else:
        logger.warning(f"CSV not found at {csv_path}, skipping")

    # ── Load Budget PDF ───────────────────────────────────────────────────
    if pdf_path and Path(pdf_path).exists():
        logger.info(f"Loading budget PDF: {pdf_path}")
        pages = load_budget_pdf(pdf_path)
        logger.info(f"  → {len(pages)} PDF page documents")
        all_documents.extend(pages)
    else:
        logger.warning(f"PDF not found at {pdf_path}, skipping")

    if not all_documents:
        logger.error("No documents loaded. Check file paths.")
        sys.exit(1)

    # ── Chunk ─────────────────────────────────────────────────────────────
    logger.info("Chunking documents...")
    chunks = chunk_documents(all_documents, chunk_size=400, overlap=80)
    logger.info(f"  → {len(chunks)} chunks total")

    # ── Embed ─────────────────────────────────────────────────────────────
    logger.info("Embedding chunks...")
    embeddings, chunks = embed_chunks(chunks, batch_size=64)
    logger.info(f"  → embedding matrix shape: {embeddings.shape}")

    # ── Index & Save ──────────────────────────────────────────────────────
    logger.info("Building FAISS index...")
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save(output_dir)
    logger.info(f"  → VectorStore saved to {output_dir}/")

    # Summary stats
    election_chunks = sum(1 for c in chunks if c["source"] == "election_csv")
    budget_chunks = sum(1 for c in chunks if c["source"] == "budget_pdf")
    logger.info(
        f"\n{'='*50}\n"
        f"INGESTION COMPLETE\n"
        f"  Election chunks : {election_chunks}\n"
        f"  Budget chunks   : {budget_chunks}\n"
        f"  Total vectors   : {store.index.ntotal}\n"
        f"  Embedding dim   : {embeddings.shape[1]}\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghana RAG Ingestion Pipeline")
    parser.add_argument("--csv", required=True, help="Path to Ghana election CSV")
    parser.add_argument("--pdf", required=True, help="Path to Ghana 2025 Budget PDF")
    parser.add_argument("--output", default="vector_store", help="Output directory")
    args = parser.parse_args()

    ingest(args.csv, args.pdf, args.output)
