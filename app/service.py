import hashlib
import time
import uuid

import numpy as np

from app.chunking import CHUNKER_VERSION, chunk_document
from app.config import Settings
from app.db import connection
from app.providers import Provider, terms

REFUSAL = "I couldn't find enough evidence in the documents to answer this question."


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = Provider(settings)

    def _usage(self, conn, request_id, operation, units, started, refused=False):
        conn.execute(
            "INSERT INTO request_usage (id, operation, mode, units, refused, elapsed_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                request_id,
                operation,
                "mock" if self.settings.mock_llm else "provider",
                units,
                refused,
                int((time.monotonic() - started) * 1000),
            ),
        )

    def ingest(self, source: str, text: str) -> dict:
        started, request_id = time.monotonic(), uuid.uuid4()
        chunks = chunk_document(source, text)
        if not chunks:
            raise ValueError("Document must contain text beyond headings")
        digest = hashlib.sha256((CHUNKER_VERSION + text).encode()).hexdigest()
        with connection(self.settings) as conn:
            # Serializes writes to one logical source; replacement and usage commit atomically.
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (source,))
            previous = conn.execute(
                "SELECT content_hash FROM documents WHERE source = %s", (source,)
            ).fetchone()
            unchanged = previous and previous["content_hash"] == digest
            units = 0
            if not unchanged:
                vectors, units = self.provider.embed([f"{c.section}\n{c.text}" for c in chunks])
                conn.execute(
                    "INSERT INTO documents(source, content_hash) VALUES (%s, %s) "
                    "ON CONFLICT (source) DO UPDATE SET content_hash=EXCLUDED.content_hash, "
                    "updated_at=now()",
                    (source, digest),
                )
                conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
                with conn.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (c.id, c.source, c.section, c.text, c.ordinal, np.array(v))
                            for c, v in zip(chunks, vectors, strict=True)
                        ],
                    )
            self._usage(conn, request_id, "ingest", units, started)
        return {
            "request_id": str(request_id),
            "source": source,
            "chunks": len(chunks),
            "unchanged": bool(unchanged),
            "usage": self._usage_payload(units),
        }

    def _usage_payload(self, units):
        return {"units": units, "unit": "mock_units" if self.settings.mock_llm else "tokens"}

    def query(self, question: str) -> dict:
        started, request_id = time.monotonic(), uuid.uuid4()
        vectors, units = self.provider.embed([question])
        vector = np.array(vectors[0])
        with connection(self.settings) as conn:
            # Exact cosine search is appropriate for the small handbook corpus; no ANN recall loss.
            hits = []
            if np.linalg.norm(vector) > 0:
                hits = conn.execute(
                    "SELECT id, source, section, body AS text, "
                    "1 - (embedding <=> %s) AS score FROM chunks "
                    "WHERE 1 - (embedding <=> %s) >= %s "
                    "ORDER BY embedding <=> %s, id LIMIT %s",
                    (
                        vector,
                        vector,
                        self.settings.similarity_threshold,
                        vector,
                        self.settings.top_k,
                    ),
                ).fetchall()
            if self.settings.mock_llm:
                # Hash collisions are not evidence. Mock retrieval must share actual terms.
                query_terms = set(terms(question))
                hits = [
                    hit
                    for hit in hits
                    if query_terms.intersection(terms(hit["section"] + " " + hit["text"]))
                ]
            citations, answer, refused = [], REFUSAL, True
            if hits:
                generated = self.provider.generate(question, hits)
                units += generated.units
                answer, refused = generated.answer, generated.refused
                by_id = {hit["id"]: hit for hit in hits}
                for citation in generated.citations:
                    hit = by_id[citation["chunk_id"]]
                    citations.append(
                        {
                            **citation,
                            "source": hit["source"],
                            "section": hit["section"],
                            "score": float(hit["score"]),
                        }
                    )
            self._usage(conn, request_id, "query", units, started, refused)
        return {
            "request_id": str(request_id),
            "answer": answer,
            "citations": citations,
            "refused": refused,
            "usage": self._usage_payload(units),
            "retrieved_sources": list(dict.fromkeys(h["source"] for h in hits)),
        }
