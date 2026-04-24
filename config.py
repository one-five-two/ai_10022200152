"""
config.py — Centralised Configuration
Free local version: no paid API key is required.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    DATA_DIR: Path = Path("data")
    VECTOR_STORE_DIR: Path = Path("vector_store")
    LOG_DIR: Path = Path("logs")
    EXPERIMENT_DIR: Path = Path("experiments")

    CSV_FILENAME: str = "ghana_elections.csv"
    PDF_FILENAME: str = "ghana_budget_2025.pdf"

    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80
    SENTENCES_PER_CHUNK: int = 3
    SENTENCE_OVERLAP: int = 1

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    EMBEDDING_BATCH_SIZE: int = 64

    TOP_K: int = 5
    QUERY_EXPANSION_N: int = 2
    USE_QUERY_EXPANSION: bool = True

    MAX_CONTEXT_TOKENS: int = 3000
    CHARS_PER_TOKEN: int = 4

    # Free local generator. No Claude/Anthropic credits needed.
    LLM_MODEL: str = "google/flan-t5-small"
    LLM_MAX_TOKENS: int = 300
    LLM_TEMPERATURE: float = 0.0

    PROMPT_VERSION: str = "v2"
    MEMORY_MAX_TURNS: int = 5
    USE_MEMORY: bool = True

    def csv_path(self) -> Path:
        return self.DATA_DIR / self.CSV_FILENAME

    def pdf_path(self) -> Path:
        return self.DATA_DIR / self.PDF_FILENAME

    def validate(self) -> None:
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.TOP_K < 1:
            raise ValueError("TOP_K must be at least 1")
        if self.LLM_TEMPERATURE < 0 or self.LLM_TEMPERATURE > 1:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 1")


cfg = Config()
