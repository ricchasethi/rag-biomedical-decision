# BioRAG — Airflow Infrastructure (Step 1)

Docker Compose stack that runs the BioRAG ingestion pipeline:

| Service | Port | Role |
|---|---|---|
| **Airflow 3.3.1** | 8080 | Scheduler + API server + Celery worker |
| **Postgres 16** | 5432 | Two databases: `airflow` (Airflow's own state) and `biorag` (papers, chunks, ingest runs) |
| **Qdrant 1.19.0** | 6333 / 6334 | Chunk embeddings (REST + gRPC) |
| **Redis 7.2** | — | Celery broker (internal only) |

**Design rules**

1. `docker-compose.yaml` is vendored from upstream, **verbatim and read-only**. All
   customisation lives in `docker-compose.override.yml`, so upgrading Airflow is a
   one-line re-download.
2. The ML stack is **baked into a custom image** (`Dockerfile`), not installed via
   `_PIP_ADDITIONAL_REQUIREMENTS` — that flag reinstalls on every container start.
3. The repo root is bind-mounted at `/opt/biorag` with `PYTHONPATH` pointing at it,
   so pipeline code is editable on the host with **no rebuild required**.
4. BioRAG data lives in the separate `biorag` database, so `airflow db reset`
   can never wipe the corpus.

---

## Prerequisites

- Docker Engine + Compose v2 (`docker --version`, `docker compose version`)
- ~10 GB free disk
- Ports free: 8080, 5432, 6333, 6334

```bash
ss -ltn | grep -E ':(5432|6333|6334|8080) ' || echo "all free"
```

---

## Step 1 — Full setup from scratch

Run from the **repository root** unless a step says otherwise.

### 1.1 Directory skeleton

Airflow needs these to exist before first boot, or Docker creates them as `root`
and Airflow cannot write logs. `var/` is runtime data, kept out of the source tree.

```bash
cd /home/riccha/Documents/practice-agentic-ai/rag-biomedical-decision
mkdir -p airflow/{dags,logs,config,plugins,initdb} var/{raw,reports,hf_cache}
```

Verify: `ls -d airflow/* var/*` → 5 dirs under `airflow/`, 3 under `var/`.

### 1.2 Vendor the official compose file

```bash
curl -Lo airflow/docker-compose.yaml \
  https://airflow.apache.org/docs/apache-airflow/3.3.1/docker-compose.yaml
```

Verify:
```bash
grep -m1 "apache/airflow:" airflow/docker-compose.yaml   # -> apache/airflow:3.3.1
wc -l airflow/docker-compose.yaml                        # -> ~340
```

### 1.3 Dockerfile

```bash
cat > airflow/Dockerfile <<'DOCKERFILE'
FROM apache/airflow:3.3.1

# psycopg2 and some ML wheels need a compiler + libpq headers at install time.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Never pip install as root in the Airflow image.
USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
DOCKERFILE
```

### 1.4 Requirements

The `+cpu` pin is critical: plain `pip install torch` on Linux pulls the CUDA
build (~2.5 GB of unused NVIDIA libraries). The base image runs **Python 3.13**,
and `torch-2.9.1+cpu-cp313` exists on the PyTorch CPU index.

```bash
cat > airflow/requirements-airflow.txt <<'REQS'
# CPU-only torch. Without the +cpu pin and the extra index, sentence-transformers
# drags in the CUDA build and the image goes from ~4.8 GB to ~9 GB.
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.9.1+cpu

sentence-transformers==5.4.1
qdrant-client==1.19.0          # matches the Airflow 3.3.1 constraint set
psycopg2-binary==2.9.12        # already in base image; pinned to survive base upgrades
pypdf==6.16.2
requests>=2.32
REQS
```

Not listed because they ship in `apache/airflow:3.3.1` already (30 providers total):
`apache-airflow-providers-postgres`, `-celery`, `-redis`, `-fab`, `-common-sql`.

### 1.5 Postgres init script

Creates the second database at first boot. Quoted delimiter keeps `$POSTGRES_USER`
literal so Postgres expands it at runtime.

```bash
cat > airflow/initdb/01-create-biorag-db.sh <<'INITDB'
#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE biorag;
EOSQL
INITDB

chmod +x airflow/initdb/01-create-biorag-db.sh
```

Verify: `ls -l airflow/initdb/` → mode `-rwxr-xr-x`.

> **Caveat:** `docker-entrypoint-initdb.d` runs **only when the Postgres data volume
> is empty**. Editing this script later has no effect unless you `docker compose down -v`.

### 1.6 Compose override

```bash
cat > airflow/docker-compose.override.yml <<'OVERRIDE'
# Additions to the vendored docker-compose.yaml. Compose merges this on top:
# `environment` maps merge key-by-key, `volumes`/`ports` lists append.

x-biorag-common: &biorag-common
  build:
    context: .
    dockerfile: Dockerfile
  image: biorag/airflow:3.3.1
  environment:
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    # The pipeline package lives in the repo, mounted at /opt/biorag.
    PYTHONPATH: /opt/biorag
    BIORAG_DB_URL: postgresql+psycopg2://airflow:airflow@postgres:5432/biorag
    BIORAG_QDRANT_URL: http://qdrant:6333
    BIORAG_RAW_DIR: /opt/biorag/var/raw
    BIORAG_REPORT_DIR: /opt/biorag/var/reports
    BIORAG_EMBED_MODEL: pritamdeka/S-PubMedBert-MS-MARCO
    HF_HOME: /opt/biorag/var/hf_cache   # model downloads persist on the host
    TOKENIZERS_PARALLELISM: 'false'     # silences the fork warning under Celery
  volumes:
    - ../:/opt/biorag

services:
  postgres:
    volumes:
      - ./initdb:/docker-entrypoint-initdb.d:ro

  qdrant:
    image: qdrant/qdrant:v1.19.0
    container_name: biorag-qdrant
    ports:
      - "6333:6333"   # REST + dashboard
      - "6334:6334"   # gRPC
    volumes:
      - qdrant-storage:/qdrant/storage
    restart: always

  airflow-apiserver:     *biorag-common
  airflow-scheduler:     *biorag-common
  airflow-dag-processor: *biorag-common
  airflow-triggerer:     *biorag-common
  airflow-init:          *biorag-common
  airflow-cli:           *biorag-common

  airflow-worker:
    <<: *biorag-common
    depends_on:
      qdrant:
        condition: service_started

volumes:
  qdrant-storage:
OVERRIDE
```

Verify:
```bash
cd airflow && docker compose config --services && cd ..
```

Expect exactly **9** services — `qdrant` appearing proves the override merged:

```
postgres  redis  qdrant
airflow-apiserver  airflow-scheduler  airflow-dag-processor
airflow-worker  airflow-triggerer  airflow-init
```

`airflow-cli` and `flower` are hidden behind Compose profiles (`debug`, `flower`)
and correctly do not appear.

> Running this **before** 1.7 prints `WARN ... "FERNET_KEY" variable is not set`.
> Harmless — `.env` does not exist yet. The warnings disappear after 1.7.

### 1.7 Environment file

`AIRFLOW_UID` makes container-written files owned by you rather than root.
`FERNET_KEY` encrypts stored connection passwords. Both keys are generated fresh
and never committed. Note the **unquoted** delimiter here — the substitutions must run.

```bash
cd airflow
cat > .env <<ENVFILE
AIRFLOW_UID=$(id -u)
AIRFLOW_PROJ_DIR=.
FERNET_KEY=$(python3 -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())")
AIRFLOW__API_AUTH__JWT_SECRET=$(python3 -c "import secrets;print(secrets.token_hex(32))")
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
ENVFILE
cd ..
```

Verify — no warnings this time, and the UID resolved:
```bash
cd airflow
docker compose config --services                              # 9 services, 0 warnings
docker compose config | grep -E 'user:|FERNET_KEY' | head -3  # user: "1001:0", key non-empty
cd ..
```

A `user: "50000:0"` means `AIRFLOW_UID` did not take and `logs/` will be root-owned.

### 1.8 Gitignore runtime dirs

```bash
cat >> .gitignore <<'GITIGNORE'

# --- Airflow / pipeline runtime ---
airflow/logs/
airflow/.env
airflow/config/
var/
GITIGNORE
```

Verify: `git status --short | grep -E "airflow/|var/"` — `airflow/.env` and `var/`
must **not** appear.

### 1.9 Build the image

First build downloads ~2.5 GB; expect 5–12 minutes.

```bash
cd airflow
docker compose build
```

Verify:
```bash
docker images biorag/airflow          # ~4.8 GB is correct (base alone is 3.18 GB)

# Definitive check that the CPU pin took — size alone is only a proxy
docker run --rm --entrypoint python biorag/airflow:3.3.1 -c \
  "import torch; print(torch.__version__); print('CUDA compiled in:', torch.version.cuda)"
# -> 2.9.1+cpu  /  CUDA compiled in: None
```

Over ~7 GB, or a `torch.version.cuda` that is not `None`, means the CUDA build slipped in.

### 1.10 Initialise the databases

One-shot container: runs Airflow's schema migrations and creates the admin user.
Postgres also boots here for the first time and runs `01-create-biorag-db.sh`.

```bash
docker compose up airflow-init
```

Verify: output contains `User "airflow" created`, container exits **0**.

### 1.11 Start the stack

```bash
docker compose up -d
sleep 60
docker compose ps        # all running, Airflow services (healthy)
```

### 1.12 Final checkpoint — all four must pass

```bash
# 1. Airflow API responds
curl -s localhost:8080/api/v2/version

# 2. Qdrant is ready
curl -s localhost:6333/readyz

# 3. The biorag database exists alongside airflow's
docker compose exec postgres psql -U airflow -lqt | cut -d'|' -f1 | grep -w biorag

# 4. ML stack importable in the worker AND the repo mount works
docker compose exec airflow-worker python -c \
  "import torch, sentence_transformers, qdrant_client, psycopg2; \
   from core.rag_engine import BioRAGEngine; \
   print('torch', torch.__version__); print('BioRAG import OK')"
```

Expected:

1. `{"version":"3.3.1", ...}`
2. `all shards are ready`
3. ` biorag`
4. `torch 2.9.1+cpu` then `BioRAG import OK`

Check 4 is the important one: it proves the worker can import your existing
`core/rag_engine.py` through the bind mount and has every dependency the DAG needs.

Then open:

- **http://localhost:8080** — Airflow UI, login `airflow` / `airflow`. DAG list
  empty (examples disabled).
- **http://localhost:6333/dashboard** — Qdrant, zero collections.

---

## Ingesting papers

The pipeline runs as three independent stages. Each one takes its work list from
the `status` column in `papers` rather than from data handed over by the previous
stage, so any stage can be re-run on its own and a crashed run resumes from the
database instead of restarting.

```
discover  ->  fetch  ->  parse
 (papers)     (PDFs)     (text)
```

| Stage | Reads | Writes | Status afterwards |
|---|---|---|---|
| `discover` | arXiv API | `papers`, `ingest_runs` | `discovered` |
| `fetch` | papers at `discovered` | `var/raw/<YYYY-MM>/<id>.pdf` | `fetched` |
| `parse` | papers at `fetched` | `var/raw/<YYYY-MM>/<id>.txt`, `content_hash` | `parsed` |

All commands run from the `airflow/` directory. **Add `--dry-run` to any
`discover` call** to see exactly what would be ingested without writing anything.

### 1. Discover

**By category — the daily mode.** Uses `BIORAG_ARXIV_CATEGORIES` from
`docker-compose.override.yml` (default `q-bio.QM,q-bio.GN,q-bio.NC`).

```bash
# Everything new in the configured categories, last 3 days
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --days 3 --max-results 25

# A different category and an explicit window
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --categories q-bio.GN --since 2026-08-01 --until 2026-08-07 --max-results 10
```

**By topic.** `--search` takes a phrase; repeat it to AND several together.
Combine with `--any-category` to search all of arXiv instead of just q-bio.

```bash
# Topic, restricted to the configured categories
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --search "protein structure prediction" --days 365 --max-results 20

# Topic, across all of arXiv
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --search "single cell RNA sequencing" --any-category --days 365 --max-results 20

# Two phrases, both required
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --search "CRISPR" --search "cancer" --any-category --days 365 --max-results 10

# Title only, rather than the whole record
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --search "Alzheimer" --field ti --any-category --days 730 --max-results 10
```

> **`--days` still applies to topic searches.** The topic clause is AND-ed with the
> date window, so `--search "CRISPR"` at the default `--days 2` searches the last
> two days *and* the topic, which usually returns nothing. Use `--days 365` or
> `--since` for topic ingestion.

**By exact identifier.** Ignores topic, category and dates. Versions may be pinned
(`2609.01055v2`); a bare id returns the latest.

```bash
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --arxiv-id 1904.08007 --arxiv-id 2609.01055
```

### The query grammar

Categories are **OR**-ed (a paper in any of them counts); topic terms are
**AND**-ed (each narrows further). Terms are quoted, so multi-word input is
matched as a phrase rather than as loose keywords.

| Command | Generated `search_query` |
|---|---|
| *(default)* | `(cat:q-bio.QM OR cat:q-bio.GN OR cat:q-bio.NC) AND submittedDate:[...]` |
| `--search "protein folding"` | `(cat:q-bio.QM ...) AND (all:"protein folding") AND submittedDate:[...]` |
| `--search X --any-category` | `(all:"X") AND submittedDate:[...]` |
| `--search X --search Y` | `(all:"X" AND all:"Y") AND submittedDate:[...]` |
| `--field ti --search X` | `(ti:"X") AND submittedDate:[...]` |
| `--arxiv-id A --arxiv-id B` | *(none — uses the API's `id_list=A,B` instead)* |

`--field` accepts `all` (default), `ti` title, `abs` abstract, `au` author,
`co` comment, `jr` journal ref, `cat` category, `rn` report number.

### 2. Fetch

Downloads PDFs for everything at `discovered`, into `var/raw/<YYYY-MM>/`.
Deliberately paced at one request every 3 seconds, so 25 papers take ~75 s.

```bash
docker compose exec airflow-worker python -m biorag_pipeline.fetch --limit 25

# Only specific papers (they must already be in the papers table)
docker compose exec airflow-worker python -m biorag_pipeline.fetch \
  --arxiv-id 2609.01055 --arxiv-id 2609.01228
```

Papers already on disk are re-marked `fetched` without hitting the network.
Transient failures (timeout, 5xx) stay at `discovered` so the next run retries
them; permanent ones (404, oversized, no `pdf_url`) become `failed` and are not
retried. Either way the reason lands in `ingest_errors`.

### 3. Parse

Extracts text with pypdf, normalises section headings, drops the bibliography,
and writes the result beside the PDF as `.txt` — so re-chunking later needs
neither a re-download nor a re-extraction.

```bash
docker compose exec airflow-worker python -m biorag_pipeline.parse --limit 25

# Show the first 1500 characters of what was extracted - the real quality signal
docker compose exec airflow-worker python -m biorag_pipeline.parse --limit 5 --show

# Keep the reference list instead of truncating at it
docker compose exec airflow-worker python -m biorag_pipeline.parse --limit 5 --keep-references
```

A PDF that yields too little text (scanned or image-only) falls back to the
abstract already stored from the arXiv metadata, so the paper stays searchable
rather than being dropped. The fallback is recorded in `ingest_errors`.

### Worked example: ingest a topic end to end

```bash
cd airflow

# 1. Look before you write
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --search "protein language model" --any-category --days 365 \
  --max-results 15 --dry-run

# 2. Commit it
docker compose exec airflow-worker python -m biorag_pipeline.discover \
  --search "protein language model" --any-category --days 365 --max-results 15

# 3. Download and extract
docker compose exec airflow-worker python -m biorag_pipeline.fetch --limit 15
docker compose exec airflow-worker python -m biorag_pipeline.parse --limit 15 --show
```

### Inspecting what landed

```bash
# Where every paper is in the lifecycle
docker compose exec postgres psql -U airflow -d biorag -c \
"SELECT status, count(*) FROM papers GROUP BY status ORDER BY 2 DESC;"

# Most recent papers
docker compose exec postgres psql -U airflow -d biorag -c \
"SELECT arxiv_id, version, status, primary_category, left(title,45) AS title
 FROM papers ORDER BY discovered_at DESC LIMIT 10;"

# Run history
docker compose exec postgres psql -U airflow -d biorag -c \
"SELECT run_id, status, n_discovered, n_failed, categories
 FROM ingest_runs ORDER BY started_at DESC LIMIT 10;"

# Failures grouped by stage
docker compose exec postgres psql -U airflow -d biorag -c \
"SELECT stage, count(*), min(left(reason,60)) FROM ingest_errors GROUP BY stage;"

# What is on disk
du -sh ../var/raw && find ../var/raw -name '*.pdf' | wc -l
```

### Re-running is safe

Every stage is idempotent, which is the whole point of the Postgres schema:

| Re-run | What happens |
|---|---|
| `discover` with the same window | `unchanged: N`, **zero rows written** — `upsert_paper` only writes when the arXiv version is higher |
| `discover` after a paper is revised | `updated: 1`, status reset to `discovered` so it re-fetches and re-chunks |
| `fetch` | Papers already `fetched` are no longer in the queue; a PDF already on disk skips the network |
| `parse` | Same text ⇒ same `content_hash` ⇒ skips straight to `indexed` without re-chunking |

### Retrying failures

```bash
# See what failed and why
docker compose exec postgres psql -U airflow -d biorag -c \
"SELECT arxiv_id, status, left(error,80) FROM papers WHERE status='failed';"

# Put them back in the queue after fixing the cause
docker compose exec postgres psql -U airflow -d biorag -c \
"UPDATE papers SET status='discovered', error=NULL WHERE status='failed';"
```

---

## Day-to-day operations

All commands run from the `airflow/` directory.

```bash
docker compose up -d                    # start
docker compose stop                     # stop, keep data
docker compose ps                       # status
docker compose logs -f airflow-worker   # follow one service
docker compose logs --tail=100 airflow-scheduler
docker compose restart airflow-scheduler
```

Shell access:
```bash
docker compose exec airflow-worker bash
docker compose exec airflow-worker airflow dags list
docker compose exec postgres psql -U airflow -d biorag
```

After editing `Dockerfile` or `requirements-airflow.txt` a rebuild is required.
Editing Python under `/opt/biorag` (the repo) needs **no** rebuild.
```bash
docker compose build && docker compose up -d
```

User management — no external account exists; the admin is created by
`airflow-init` from `_AIRFLOW_WWW_USER_*` in `.env`:
```bash
docker compose exec airflow-apiserver airflow users list
docker compose exec airflow-apiserver airflow users reset-password \
  --username airflow --password NEW_PASSWORD
```

Disk usage:
```bash
docker system df        # true usage; SIZE in `docker images` double-counts shared layers
docker builder prune    # reclaim build cache (safe)
```

---

## Teardown

```bash
docker compose down     # stop + remove containers, KEEP volumes
docker compose down -v  # ALSO delete Postgres + Qdrant data (destructive)
```

`down -v` wipes the corpus and every embedding. It is also the only way to make
`initdb/01-create-biorag-db.sh` run again.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARN "FERNET_KEY" variable is not set` | `.env` missing or wrong directory | Run 1.7; `docker compose` must run from `airflow/` |
| `user: "50000:0"` in `config` | `AIRFLOW_UID` unset | Re-run 1.7; check `grep AIRFLOW_UID .env` |
| Permission denied writing logs | dirs created as root before 1.7 | `sudo chown -R $(id -u):0 logs config plugins` |
| `airflow-init` exits 1 | missing dirs, or bad UID | Re-run 1.1, verify `.env`, retry |
| Image ~9 GB | CUDA torch installed | Confirm `--extra-index-url` + `+cpu` in requirements, rebuild |
| `database "biorag" does not exist` | Postgres volume pre-existed 1.5 | `docker compose down -v` then re-run 1.10 |
| Port already allocated | conflict on 8080/5432/6333 | Stop the other process, or remap in the override |
| Qdrant unreachable from worker | wrong host | Inside containers use `http://qdrant:6333`, not `localhost` |

---

## Tuning and open questions

Every value below is an environment variable in `docker-compose.override.yml`, so
changing one needs a `docker compose up -d` and a re-run - never a code change.
Nothing here is a bug; they are decisions worth revisiting once the pipeline is
stable, ideally measured rather than guessed.

### 1. `BIORAG_CHUNK_SIZE` is in **characters**, not tokens (open)

**Current:** `512` chars, giving ~428-char chunks averaging **54 tokens** and
**113 chunks per paper**.

`DocumentChunker` measures in characters (`core/rag_engine.py`, `sent_len = len(sent)`),
but almost every RAG guide quotes chunk sizes in *tokens*, where 512 is a common
default. The coincidence makes `chunk_size=512` look conventional while actually
being about a quarter of the usual size.

| | Current | Typical RAG |
|---|---|---|
| Chunk size | 428 chars ~ **54 tokens** | 1,000-2,000 chars ~ 250-500 tokens |
| Chunks per paper | **113** | 25-50 |
| Embedding context used | **~11%** of PubMedBERT's 512-token window | 50-100% |

**Why the default was fine and no longer is.** `BioRAGEngine(chunk_size=512)` was
chosen when the corpus was four hand-written documents in `data/sample_corpus.py`,
each a few thousand characters. A full arXiv PDF is ~51,000 characters - a 10-25x
jump the default was never sized for.

**What it costs:** 4x the vectors (6,654 chunks ~ 20 MB today; ~350 MB at 1,000
papers), 4x the embedding time, and thin context per hit - a retrieved chunk is
often a single sentence with unresolved pronouns. Against that, small chunks give
sharper lexical precision, which is part of why the BM25 path scores well.

**To change:**

```yaml
# docker-compose.override.yml, under x-biorag-common -> environment
    BIORAG_CHUNK_SIZE: '1800'
    BIORAG_CHUNK_OVERLAP: '200'
```

```bash
docker compose up -d
docker compose exec postgres psql -U airflow -d biorag -c \
  "UPDATE papers SET status='parsed' WHERE status IN ('chunked','indexed');"
docker compose exec airflow-worker python -m biorag_pipeline.chunk --limit 500
docker compose exec airflow-worker python -m biorag_pipeline.embed --limit 20000
```

Re-chunking needs **no re-download and no re-parse** - the extracted text is
already on disk. That is what the three-tier storage split was for.

> **Measure, do not guess.** `evals/retrieval_eval.py` reports MRR and NDCG@K and
> is exactly the instrument for this. Run it at 512, run it at 1800, compare.
> Changing chunk size on intuition is how RAG systems quietly get worse.

### 2. `BIORAG_EMBED_MODEL` is PubMed-tuned, the corpus is arXiv (open)

**Current:** `pritamdeka/S-PubMedBert-MS-MARCO`, 768-dim, trained on PubMed.

A good fit while ingesting `q-bio.*`. If the corpus drifts toward `cs.LG` /
`stat.ML` - which happens easily, since many q-bio papers are cross-listed and
`--any-category` topic searches ignore categories entirely - a general model such
as `BAAI/bge-base-en-v1.5` would likely retrieve better.

Changing the model changes the vector dimension, so it needs a **new collection**,
not just a re-embed:

```yaml
    BIORAG_EMBED_MODEL: 'BAAI/bge-base-en-v1.5'
    BIORAG_QDRANT_COLLECTION: 'biorag_chunks_bge'
```
```bash
docker compose exec postgres psql -U airflow -d biorag -c \
  "UPDATE chunks SET embedded = FALSE;"
docker compose exec airflow-worker python -m biorag_pipeline.embed --limit 20000
```

Keeping the old collection means you can A/B the two with the same eval harness.

### 3. Reference lists are dropped at parse time (deliberate, revisit if needed)

`parse.py` truncates each document at its References heading. Bibliographies are
roughly a third of a paper and are pure lexical noise under BM25 - author names
and title fragments that match almost any query while answering none.

`--keep-references` disables it per run. If citation-graph features are ever
wanted, this is the decision to revisit, and it would want its own table rather
than being folded back into chunk text.

### 4. `lookback_days` defaults to 2, not 1 (deliberate)

arXiv's search index lags announcement, so a strict 24-hour window silently drops
papers. The overlap is free: `upsert_paper()` returns `unchanged` and writes
nothing for anything already stored. Widen it further if a run ever reports fewer
papers than arXiv's listing page shows.

### 5. Fetch is not parallelised (deliberate)

The 3-second arXiv throttle in `fetch.py` is **process-local**, so mapping the
fetch task would give each mapped instance its own timer and defeat the rate
limit. One task, one throttle. For bulk historical ingestion use arXiv's S3 bulk
access rather than raising this ceiling.

---

## Where the build has got to

| Step | Status | Delivers |
|---|---|---|
| 1 Infrastructure | done | Airflow + Postgres + Qdrant, custom worker image |
| 2 Postgres schema | done | `papers`, `chunks`, `ingest_runs`, `ingest_errors`, migration runner |
| 3 arXiv client | done | Date/topic/id search, `discover` stage |
| 4 Fetch + parse | done | PDFs to `var/raw/`, sectioned text, `content_hash` |
| 5 Chunk + embed | pending | `DocumentChunker` → `chunks` table → Qdrant vectors |
| 6 The DAG | pending | Wires stages 3–5 with dynamic task mapping |
| 7 Daily report | pending | Markdown + HTML into `var/reports/` |
| 8 Cleanup | pending | Raw-blob retention, orphaned Qdrant points, old logs |
| 9 Serving cutover | pending | `server.py` reads Postgres + Qdrant, not `sample_corpus.py` |

**No DAG exists yet.** Everything above is driven from the CLI; Airflow is running
but has an empty DAG list. The stages are deliberately built as an importable
package (`biorag_pipeline/`) rather than inside `dags/`, so they run and are
tested without Airflow. Step 6 turns them into thin `@task` wrappers.

**Qdrant is still empty** — nothing is embedded until Step 5.

The existing embedded Qdrant store at `../qdrant_data/` (used by the `--hybrid`
CLI flag on the old `cli.py`) is untouched and keeps working. Embedded mode holds
an exclusive file lock, so the server here uses a separate `qdrant-storage` volume
that the DAG will populate.

### Running the test suites

```bash
docker compose exec airflow-worker python /opt/biorag/tests/test_pipeline_db.py    # 25 checks
docker compose exec airflow-worker python /opt/biorag/tests/test_arxiv_client.py   # 32 checks
```

The first needs the database; the second is fully offline.
