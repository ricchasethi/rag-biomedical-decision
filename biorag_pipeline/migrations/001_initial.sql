-- ---------------------------------------------------------------------------
-- 001_initial - papers, chunks, ingest_runs, ingest_errors
-- ---------------------------------------------------------------------------

-- == papers =================================================================
-- One row per arXiv paper. arxiv_id is the NATURAL primary key, which is what
-- makes ingestion idempotent: "have I seen this paper?" is a uniqueness
-- constraint enforced by Postgres, not application logic that can drift.
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         TEXT        PRIMARY KEY,      -- '2509.01234', version stripped
    version          INTEGER     NOT NULL DEFAULT 1,

    -- The engine's add_document(doc_id=...) contract, computed by the database so
    -- application code can never disagree about how a doc_id is formed.
    doc_id           TEXT        GENERATED ALWAYS AS ('arxiv_' || arxiv_id) STORED,

    title            TEXT        NOT NULL,
    abstract         TEXT,
    authors          TEXT[]      NOT NULL DEFAULT '{}',
    categories       TEXT[]      NOT NULL DEFAULT '{}',
    primary_category TEXT,
    published_at     TIMESTAMPTZ,
    updated_at_src   TIMESTAMPTZ,                  -- arXiv's own updated timestamp
    pdf_url          TEXT,

    raw_path         TEXT,                         -- where the PDF landed under var/raw
    -- sha256 of the extracted text. A new arXiv version only triggers re-chunking
    -- when the text actually changed, so a metadata-only revision costs nothing.
    content_hash     TEXT,

    -- Lifecycle. Lets a failed run resume instead of restarting: the DAG can ask
    -- "which papers are 'fetched' but not yet 'indexed'?"
    status           TEXT        NOT NULL DEFAULT 'discovered'
                     CHECK (status IN ('discovered','fetched','parsed','indexed','failed','skipped')),
    n_chunks         INTEGER     NOT NULL DEFAULT 0,
    error            TEXT,

    discovered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at       TIMESTAMPTZ,
    first_run_id     TEXT                          -- the run that discovered it
);

CREATE INDEX IF NOT EXISTS papers_status_idx      ON papers (status);
CREATE INDEX IF NOT EXISTS papers_discovered_idx  ON papers (discovered_at DESC);
CREATE INDEX IF NOT EXISTS papers_primary_cat_idx ON papers (primary_category);


-- == chunks =================================================================
-- Mirrors core.rag_engine.Chunk, minus `tokens`. Tokens are deliberately NOT
-- stored: they are a pure function of TextProcessor.tokenize(), so persisting
-- them would silently go stale the next time clean_text() changes. Recompute
-- on load instead.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT        PRIMARY KEY,           -- engine's md5(doc_id:pos)[:12]
    arxiv_id    TEXT        NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    ordinal     INTEGER     NOT NULL,              -- position within the document
    text        TEXT        NOT NULL,
    section     TEXT        NOT NULL DEFAULT 'Body',
    page        INTEGER     NOT NULL DEFAULT 1,
    char_start  INTEGER     NOT NULL DEFAULT 0,
    char_end    INTEGER     NOT NULL DEFAULT 0,
    n_tokens    INTEGER     NOT NULL DEFAULT 0,

    -- Set once the vector is confirmed in Qdrant. Postgres stays the source of
    -- truth for "what should exist"; this column tracks "what actually does".
    embedded    BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- The real natural key. chunk_id is only 48 bits of MD5, so this constraint
    -- is what actually guarantees we never double-insert the same span.
    UNIQUE (arxiv_id, char_start)
);

CREATE INDEX IF NOT EXISTS chunks_arxiv_idx ON chunks (arxiv_id);
-- Partial index = the embedding work queue. Only unembedded rows are indexed,
-- so it stays tiny no matter how large the corpus grows.
CREATE INDEX IF NOT EXISTS chunks_pending_idx ON chunks (arxiv_id) WHERE NOT embedded;


-- == ingest_runs ============================================================
-- One row per Airflow DAG run. This is what the Step 7 daily report reads.
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id       TEXT        PRIMARY KEY,          -- Airflow dag_run_id
    dag_id       TEXT        NOT NULL,
    logical_date TIMESTAMPTZ NOT NULL,
    window_start TIMESTAMPTZ,                      -- arXiv submission window queried
    window_end   TIMESTAMPTZ,
    categories   TEXT[]      NOT NULL DEFAULT '{}',

    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT        NOT NULL DEFAULT 'running'
                 CHECK (status IN ('running','success','failed')),

    n_discovered INTEGER     NOT NULL DEFAULT 0,
    n_fetched    INTEGER     NOT NULL DEFAULT 0,
    n_parsed     INTEGER     NOT NULL DEFAULT 0,
    n_indexed    INTEGER     NOT NULL DEFAULT 0,
    n_chunks     INTEGER     NOT NULL DEFAULT 0,
    n_embedded   INTEGER     NOT NULL DEFAULT 0,
    n_failed     INTEGER     NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ingest_runs_date_idx ON ingest_runs (logical_date DESC);


-- == ingest_errors ==========================================================
-- Per-paper failures, keyed by stage. Mirrors the IngestionError dataclass in
-- ingestion_pubmed.py. One bad PDF must never fail a whole run, so failures are
-- recorded as data rather than raised.
CREATE TABLE IF NOT EXISTS ingest_errors (
    id         BIGSERIAL   PRIMARY KEY,
    run_id     TEXT        REFERENCES ingest_runs(run_id) ON DELETE CASCADE,
    arxiv_id   TEXT,
    stage      TEXT        NOT NULL
               CHECK (stage IN ('discover','fetch','parse','chunk','embed','report','cleanup')),
    reason     TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingest_errors_run_idx   ON ingest_errors (run_id);
CREATE INDEX IF NOT EXISTS ingest_errors_stage_idx ON ingest_errors (stage);
