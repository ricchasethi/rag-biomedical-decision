"""Apply SQL migrations to the biorag database.

Migrations are plain .sql files in migrations/, applied in filename order. Each
runs inside its own transaction and is recorded in schema_migrations, so running
this repeatedly is a no-op - safe to call on every deploy or DAG start.

    python -m biorag_pipeline.migrate            # apply pending
    python -m biorag_pipeline.migrate --status   # show state, change nothing

Deliberately not Alembic: Airflow already runs its own Alembic instance against
the `airflow` database, and a second migration framework buys nothing for a
four-table schema.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from biorag_pipeline import db

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _applied(conn) -> set[str]:
    """Ensure schema_migrations exists and return the versions already applied."""
    with conn.cursor() as cur:
        cur.execute(BOOTSTRAP)
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def status() -> int:
    """Print each migration file and whether it has been applied."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with db.connect() as conn:
        done = _applied(conn)
    for path in files:
        mark = "applied" if path.stem in done else "PENDING"
        print(f"  [{mark:>7}] {path.name}")
    pending = [p for p in files if p.stem not in done]
    print(f"{len(files)} migration(s), {len(pending)} pending")
    return len(pending)


def migrate() -> int:
    """Apply every pending migration. Returns the number applied."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print(f"no .sql files in {MIGRATIONS_DIR}")
        return 0

    applied = 0
    with db.connect() as conn:
        done = _applied(conn)
        conn.commit()

        for path in files:
            version = path.stem
            if version in done:
                print(f"  [   skip] {path.name}")
                continue
            sql = path.read_text()
            # One transaction per migration: a failure leaves the DB untouched and
            # schema_migrations unchanged, so a corrected file can just be re-run.
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)",
                        (version,),
                    )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"  [ FAILED] {path.name}: {exc}", file=sys.stderr)
                raise
            print(f"  [applied] {path.name}")
            applied += 1

    print(f"{applied} migration(s) applied")
    return applied


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="report only, apply nothing")
    args = ap.parse_args()
    status() if args.status else migrate()
