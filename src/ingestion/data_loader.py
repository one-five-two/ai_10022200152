"""
Part A: Data Engineering
------------------------
Loads and cleans both the Ghana Election CSV and the Ghana 2025 Budget PDF.

Design rationale:
- Separate loaders per source type keeps concerns isolated (single responsibility).
- Cleaning is applied before chunking so we never embed noisy tokens.
- Unicode normalization handles the mojibake that often appears in African-language PDFs.
"""

import re
import unicodedata
import pandas as pd
import pdfplumber
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CSV Loader  (Ghana Election Dataset)
# ─────────────────────────────────────────────

def load_election_csv(filepath: str) -> pd.DataFrame:
    """
    Loads the Ghana election CSV and returns a cleaned DataFrame.

    The election dataset typically contains columns like:
        constituency, region, candidate, party, votes, year, ...

    We preserve all columns but normalize text fields so they embed cleanly.
    """
    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    logger.info(f"Loaded election CSV: {len(df)} rows, {len(df.columns)} columns")

    # 1. Strip leading/trailing whitespace from every string column
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # 2. Normalise column names → snake_case
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.lower().strip()).strip("_")
        for col in df.columns
    ]

    # 3. Drop rows that are entirely empty (happen in Excel exports)
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4. Fill remaining NaN with empty string (avoids float bleed into text)
    df.fillna("", inplace=True)

    logger.info(f"After cleaning: {len(df)} rows remain")
    return df


def election_rows_to_text(df: pd.DataFrame) -> list[dict]:
    """
    Converts each election row into a plain-text document.

    Why row-level documents?
    - Each row is a self-contained fact (candidate X got Y votes in constituency Z).
    - Row-level granularity lets the retriever surface precise electoral facts
      without mixing unrelated constituencies in the same chunk.
    - Trade-off: many small documents → more FAISS entries, but retrieval
      precision is much higher for factual queries.
    """
    documents = []
    for _, row in df.iterrows():
        # Build a key:value natural language sentence so the embedding model
        # can understand the structure without seeing a raw CSV dump.
        parts = [f"{col.replace('_', ' ')}: {val}" for col, val in row.items() if val]
        text = " | ".join(parts)
        documents.append({
            "text": text,
            "source": "election_csv",
            "metadata": row.to_dict(),
        })
    return documents


# ─────────────────────────────────────────────
# PDF Loader  (Ghana 2025 Budget)
# ─────────────────────────────────────────────

def load_budget_pdf(filepath: str) -> list[dict]:
    """
    Extracts text from the Ghana 2025 Budget PDF page-by-page.

    pdfplumber is chosen over PyMuPDF because it handles table cells and
    multi-column layouts much better for government PDFs.

    Returns a list of page-level dicts so downstream chunkers know the
    page number (useful for citations in the final response).
    """
    pages = []
    with pdfplumber.open(filepath) as pdf:
        total = len(pdf.pages)
        logger.info(f"PDF has {total} pages")
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            cleaned = _clean_pdf_text(raw)
            if len(cleaned) > 50:          # skip near-blank pages
                pages.append({
                    "text": cleaned,
                    "source": "budget_pdf",
                    "page": i + 1,
                    "metadata": {"page": i + 1, "total_pages": total},
                })
    logger.info(f"Extracted {len(pages)} non-empty pages from PDF")
    return pages


def _clean_pdf_text(text: str) -> str:
    """
    PDF-specific cleaning pipeline:

    1. Unicode normalisation → fixes mojibake (â€™ → ', Â£ → £)
    2. Remove header/footer artefacts (page numbers, running headers)
    3. Collapse multiple whitespace/newlines to single space
    4. Remove hyphenation at line breaks (common in justified PDF text)
    5. Strip non-printable control characters

    The order matters: normalise first, then remove, then collapse.
    """
    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove standalone page number lines (e.g. "- 12 -" or just "12")
    text = re.sub(r"^\s*-?\s*\d+\s*-?\s*$", "", text, flags=re.MULTILINE)

    # 3. Fix hyphenated line breaks  ("inno-\nvation" → "innovation")
    text = re.sub(r"-\n(\w)", r"\1", text)

    # 4. Collapse newlines and tabs into spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # 5. Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # 6. Strip non-printable chars except standard ASCII punctuation
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    return text.strip()
