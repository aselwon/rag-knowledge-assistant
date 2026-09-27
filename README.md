# DocuPilot — cited handbook RAG assistant

DocuPilot is an offline-first MVP for asking questions about a small company handbook. It ingests Markdown and plain-text documents, retrieves evidence from PostgreSQL/pgvector, answers with source quotes and citations, refuses questions below a similarity threshold, and records per-request usage units.

The default `MOCK_LLM=1` path is deterministic and needs no API key at runtime. It is intended for a reproducible local demo and evaluation, rather than as a production language model.

## Public demo

[Open the live DocuPilot demo](https://docupilot-demo.pages.dev).


## Quick demo with Docker Compose

Requirements: Docker with Compose, and network access for the first image build. Runtime requests in mock mode are offline.

```bash
cp .env.example .env                 # optional; these are the mock defaults
make demo
```

The demo starts PostgreSQL + pgvector, the FastAPI service, and the React UI, ingests `data/handbook`, and runs the evaluation gate. Open:

- UI: http://127.0.0.1:18765
- API: http://127.0.0.1:18080
- API health: http://127.0.0.1:18080/api/health
- API docs (Swagger): http://127.0.0.1:18080/docs

The host ports can be changed with `WEB_PORT`, `API_PORT`, and `POSTGRES_PORT`. Stop the stack with `make down`. Compose stores database data in the `pgdata` volume.

## Local development

The supported local runtime is Python 3.12 with `uv` and Node.js 22 with npm. PostgreSQL/pgvector is still required for the API and integration tests.

```bash
cp .env.example .env
uv sync --frozen
(cd frontend && npm ci)
make db
make ingest
make api       # another terminal: uv run uvicorn app.main:app --reload --port 18080
make web       # another terminal: cd frontend && npm run dev
```

The local Vite UI is normally http://127.0.0.1:5173. No login or demo credentials are required. Compose development database credentials are `docupilot` / `docupilot` for database `docupilot`.

Useful checks:

```bash
make test      # pytest
make lint      # ruff
make eval      # writes reports/eval.json and reports/eval.md
```

`make eval` uses `--check` and exits non-zero if the documented MVP gates are missed. The evaluator creates a temporary isolated PostgreSQL schema, loads the fixed handbook corpus, then drops that schema; it does not overwrite uploaded documents in the normal database. Integration tests use `TEST_DATABASE_URL`; if it is unset, database-dependent tests are skipped. For acceptance and CI, provide an explicit `TEST_DATABASE_URL` so a missing database cannot be mistaken for a passing run.

## API examples

```bash
curl http://127.0.0.1:18080/api/health
curl http://127.0.0.1:18080/api/documents
curl -F file=@data/handbook/people.md http://127.0.0.1:18080/api/ingest
curl -s http://127.0.0.1:18080/api/query \
  -H 'content-type: application/json' \
  -d '{"question":"What is the annual learning budget?"}'
```

The query response contains `answer`, `citations` (`chunk_id`, exact `quote`, `source`, `section`, and score), `refused`, `retrieved_sources`, and `usage`. Unsupported or low-similarity questions return the refusal text with no citations. Usage totals are available at `GET /api/usage`.

## Architecture and retrieval

1. `app/ingest.py` walks the handbook and calls the service; uploads accept `.md` and `.txt` up to 1 MB.
2. `app/chunking.py` splits on Markdown headings and creates section-aware windows of 180 words with 30 words of overlap. Chunk IDs are content-derived, and re-ingest is idempotent by source plus content hash.
3. `app/providers.py` creates embeddings. Mock embeddings are deterministic signed hashed term vectors of dimension 384. Provider mode calls an OpenAI-compatible `/embeddings` endpoint.
4. `app/service.py` performs exact cosine similarity search in pgvector, ordered by distance, with configurable `TOP_K` and `SIMILARITY_THRESHOLD` (defaults 3 and 0.16). Exact search suits this small corpus and avoids approximate-index recall trade-offs. In mock mode, a second filter removes hits with no shared real query terms, preventing hash collisions from becoming evidence.
5. The mock generator cites each retrieved chunk in full so that conditions and exceptions remain visible. Provider mode calls an OpenAI-compatible `/chat/completions` endpoint with deterministic settings and `max_tokens=700`; JSON output and exact quote validation reject invalid or unverifiable responses.
6. `request_usage` records completed ingest/query operations, mode, units, refusal status, and elapsed time. Mock units are word-count-based estimates; provider units are reported token usage. Transport/provider failures do not create a usage row, so real billing may occur before a failed request is observed.

The browser UI displays expandable citations, quote text, source/section, similarity, refusal state, and the session's usage total.

## Environment

`.env.example` is a safe mock configuration:

```dotenv
MOCK_LLM=1
DATABASE_URL=postgresql://docupilot:docupilot@localhost:56432/docupilot
SIMILARITY_THRESHOLD=0.16
TOP_K=3
EMBEDDING_DIM=384
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

Set `MOCK_LLM=0` only when an OpenAI-compatible provider and `OPENAI_API_KEY` are available. Changing embedding model or dimension requires a fresh database/volume because the schema stores the embedding signature. Keep API keys out of Git.

In provider mode, question text and retrieved chunks are sent to the configured OpenAI-compatible endpoint. Use an endpoint and data policy appropriate for the documents being indexed.

## Evaluation

The fixed set contains 26 questions: 23 supported questions with expected sources and 3 deliberately unsupported questions. The check gates are:

- source hit rate ≥ 0.90;
- faithfulness/quote proxy = 1.00 (each returned quote is an exact stored-chunk substring);
- expected answer match rate ≥ 0.80;
- refusal accuracy = 1.00 for the three negative cases.

The quote proxy checks citation integrity; it does not prove that every generated claim is entailed. The verified mock run produced `reports/eval.json` and `reports/eval.md` with: source hit rate `1.000`, answer match rate `1.000`, faithfulness proxy `1.000`, and refusal accuracy `1.000` across all 26 questions.

The full verified checks also reported `ruff check` passing and 25 pytest tests passing. Pytest emitted one upstream Starlette/httpx deprecation warning; it did not fail the run. The real provider path was not exercised against a paid service; its adapter was tested with mocks. CI configuration was added but was not run on a remote CI service.

## Scope and limitations

This is a trusted, single-tenant local MVP with no authentication, authorization, tenant isolation, audit controls, or production deployment hardening. It supports Markdown and TXT; PDF ingestion is outside the implemented MVP. The mock provider is useful for deterministic demos but is not a substitute for model quality testing. First-time dependency and image installation requires network access, while mock runtime queries do not.

The project does not include multi-agent orchestration, fine-tuning, or a mobile application.

## Project layout

`app/` contains the FastAPI, ingestion, chunking, provider, pgvector, and service code; `frontend/` contains the React/Vite UI; `data/handbook/` is the demo corpus; `eval/` contains the benchmark runner and questions; `tests/` contains unit and PostgreSQL integration tests; `compose.yaml` defines the local stack.
