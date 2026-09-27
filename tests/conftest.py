import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.config import Settings
from app.db import initialize


@pytest.fixture
def database_settings():
    url = os.environ.get("TEST_DATABASE_URL", Settings().database_url)
    schema = "test_" + uuid.uuid4().hex
    try:
        admin = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError:
        if os.environ.get("TEST_DATABASE_URL"):
            pytest.fail("Configured TEST_DATABASE_URL is unavailable")
        pytest.skip("PostgreSQL unavailable: run make db or set TEST_DATABASE_URL")
    try:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        settings = Settings(
            database_url=make_conninfo(url, options=f"-csearch_path={schema},public"), mock_llm=True
        )
        initialize(settings)
        yield settings
    finally:
        admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        admin.close()
