-- ---------------------------------------------------------------------------
-- 002_add_chunked_status
--
-- Chunking and embedding are two separate writes to two separate stores, so the
-- lifecycle needs a state between them. Without it, a crash after chunking but
-- before embedding leaves a paper that is either re-chunked from scratch or
-- marked 'indexed' while Qdrant holds none of its vectors.
--
--   parsed  -> chunks written to Postgres      -> chunked
--   chunked -> every chunk confirmed in Qdrant -> indexed
--
-- The CHECK constraint is column-level, so Postgres named it papers_status_check.
-- DROP ... IF EXISTS keeps this migration re-runnable on a database where an
-- earlier attempt got part-way.
-- ---------------------------------------------------------------------------

ALTER TABLE papers DROP CONSTRAINT IF EXISTS papers_status_check;

ALTER TABLE papers ADD CONSTRAINT papers_status_check
    CHECK (status IN ('discovered', 'fetched', 'parsed', 'chunked',
                      'indexed', 'failed', 'skipped'));
