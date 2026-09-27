from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import Settings


def initialize(settings: Settings) -> None:
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""CREATE TABLE IF NOT EXISTS configuration (
            key text PRIMARY KEY, value text NOT NULL)""")
        conn.execute(
            "INSERT INTO configuration VALUES ('embedding', %s) ON CONFLICT DO NOTHING",
            (settings.embedding_signature,),
        )
        signature = conn.execute(
            "SELECT value FROM configuration WHERE key = 'embedding'"
        ).fetchone()["value"]
        if signature != settings.embedding_signature:
            raise ValueError("Embedding configuration differs from database; use a fresh database")
        conn.execute("""CREATE TABLE IF NOT EXISTS documents (
            source text PRIMARY KEY, content_hash text NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now())""")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
            id text PRIMARY KEY,
            source text NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
            section text NOT NULL, body text NOT NULL, ordinal integer NOT NULL,
            embedding vector({settings.embedding_dim}) NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source)")
        conn.execute("""CREATE TABLE IF NOT EXISTS request_usage (
            id uuid PRIMARY KEY, operation text NOT NULL, mode text NOT NULL,
            units integer NOT NULL CHECK (units >= 0), refused boolean NOT NULL DEFAULT false,
            elapsed_ms integer NOT NULL, created_at timestamptz NOT NULL DEFAULT now())""")


@contextmanager
def connection(settings: Settings):
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn
