"""Environment-driven settings for the BioRAG ingestion pipeline.

Every value is read once from the environment (set in docker-compose.override.yml)
with a host-friendly default, so the same code runs inside Airflow or from a shell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_list(name: str, default: str) -> list[str]:
    return [s.strip() for s in _env(name, default).split(",") if s.strip()]


@dataclass(frozen=True)
class Config:
    # --- Storage -------------------------------------------------------------
    db_url: str = field(default_factory=lambda: _env(
        "BIORAG_DB_URL", "postgresql://airflow:airflow@localhost:5432/biorag"))
    qdrant_url: str = field(default_factory=lambda: _env(
        "BIORAG_QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: _env(
        "BIORAG_QDRANT_COLLECTION", "biorag_chunks"))

    # --- Filesystem ----------------------------------------------------------
    raw_dir: Path = field(default_factory=lambda: Path(_env(
        "BIORAG_RAW_DIR", "./var/raw")))
    report_dir: Path = field(default_factory=lambda: Path(_env(
        "BIORAG_REPORT_DIR", "./var/reports")))

    # --- Retrieval / embedding -----------------------------------------------
    embed_model: str = field(default_factory=lambda: _env(
        "BIORAG_EMBED_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"))

    # --- arXiv ingestion (used from Step 3 onward) ---------------------------
    arxiv_categories: list[str] = field(default_factory=lambda: _env_list(
        "BIORAG_ARXIV_CATEGORIES", "q-bio.QM,q-bio.GN,q-bio.NC"))
    arxiv_max_results: int = field(default_factory=lambda: int(_env(
        "BIORAG_ARXIV_MAX_RESULTS", "50")))

    # --- Chunking (mirrors BioRAGEngine defaults) ----------------------------
    chunk_size: int = field(default_factory=lambda: int(_env(
        "BIORAG_CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(_env(
        "BIORAG_CHUNK_OVERLAP", "64")))

    # --- Retention (used in Step 8) ------------------------------------------
    raw_retention_days: int = field(default_factory=lambda: int(_env(
        "BIORAG_RAW_RETENTION_DAYS", "30")))


CONFIG = Config()
