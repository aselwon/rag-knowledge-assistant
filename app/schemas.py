from typing import Literal

from pydantic import BaseModel


class Usage(BaseModel):
    units: int
    unit: Literal["mock_units", "tokens"]


class Citation(BaseModel):
    chunk_id: str
    source: str
    section: str
    quote: str
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation]
    refused: bool
    usage: Usage
    retrieved_sources: list[str]


class IngestResponse(BaseModel):
    request_id: str
    source: str
    chunks: int
    unchanged: bool
    usage: Usage
