"""Postgres connection handling for the BioRAG pipeline.

Raw psycopg2, no ORM. The schema is four tables and the queries are simple;
an ORM here would only add a dependency that has to stay compatible with
whatever Airflow pins.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PGConnection

from biorag_pipeline.config import CONFIG


def dsn(url: str | None = None) -> str:
    """Return a libpq-compatible DSN.

    BIORAG_DB_URL is written in SQLAlchemy form (``postgresql+psycopg2://``) so the
    same variable can drive SQLAlchemy later. libpq rejects the ``+driver`` suffix,
    so strip it here.
    """
    return (url or CONFIG.db_url).replace("postgresql+psycopg2://", "postgresql://")


@contextlib.contextmanager
def connect(autocommit: bool = False) -> Iterator[PGConnection]:
    """Yield a connection, committing on clean exit and rolling back on error.

    With ``autocommit=False`` (the default) the whole ``with`` block is one
    transaction — that is what makes ``replace_chunks`` atomic.
    """
    conn = psycopg2.connect(dsn())
    conn.autocommit = autocommit
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextlib.contextmanager
def dict_cursor(conn: PGConnection) -> Iterator[psycopg2.extras.RealDictCursor]:
    """Cursor whose rows behave like dicts — used for reads."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        yield cur


def ping() -> str:
    """Return the server version. Used as a connectivity smoke test."""
    with connect(autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        return cur.fetchone()[0]
