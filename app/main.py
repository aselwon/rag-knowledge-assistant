import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db import connection, initialize
from app.schemas import IngestResponse, QueryResponse
from app.service import RAGService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    initialize(settings)
    app.state.rag = RAGService(settings)
    yield


app = FastAPI(title="DocuPilot", version="0.1.0", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Enter a question of at least 3 characters")
        return value


@app.exception_handler(psycopg.Error)
async def database_error(request, exc):
    logger.error("Database request failed: %s", type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})


@app.exception_handler(httpx.HTTPError)
async def provider_error(request, exc):
    logger.error("Provider request failed: %s", type(exc).__name__)
    return JSONResponse(status_code=502, content={"detail": "Language provider unavailable"})


@app.get("/api/health")
def health():
    settings = get_settings()
    with connection(settings) as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "mode": "mock" if settings.mock_llm else "provider"}


@app.get("/api/documents")
def documents():
    with connection(get_settings()) as conn:
        return conn.execute(
            "SELECT d.source, d.updated_at, count(c.id) AS chunks FROM documents d "
            "LEFT JOIN chunks c ON c.source=d.source GROUP BY d.source ORDER BY d.source"
        ).fetchall()


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile):
    source = Path((file.filename or "").replace("\\", "/")).name
    if not source or len(source) > 200 or Path(source).suffix.lower() not in {".md", ".txt"}:
        await file.close()
        raise HTTPException(422, "Upload a Markdown (.md) or text (.txt) file")
    try:
        raw = await file.read(get_settings().max_upload_bytes + 1)
    finally:
        await file.close()
    if len(raw) > get_settings().max_upload_bytes:
        raise HTTPException(413, "Maximum document size is 1 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "Document must be UTF-8 text") from None
    if "\x00" in text:
        raise HTTPException(422, "Document must contain plain text")
    try:
        return await run_in_threadpool(app.state.rag.ingest, source, text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/query", response_model=QueryResponse)
def query(body: Question):
    try:
        return app.state.rag.query(body.question)
    except ValueError:
        raise HTTPException(502, "Provider returned an invalid response") from None


@app.get("/api/usage")
def usage():
    with connection(get_settings()) as conn:
        return conn.execute(
            "SELECT mode, operation, count(*) AS requests, sum(units) AS units, "
            "count(*) FILTER (WHERE refused) AS refusals FROM request_usage "
            "GROUP BY mode, operation ORDER BY mode, operation"
        ).fetchall()
