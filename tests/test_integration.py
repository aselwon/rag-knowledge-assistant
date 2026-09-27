from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import connection, initialize
from app.main import app
from app.service import RAGService

pytestmark = pytest.mark.integration


def test_ingest_query_replace_and_persist(database_settings):
    service = RAGService(database_settings)
    first = service.ingest("people.md", "# Annual leave\nAnnual leave is 25 days.")
    second = service.ingest("people.md", "# Annual leave\nAnnual leave is 25 days.")
    assert not first["unchanged"] and second["unchanged"]
    assert second["usage"]["units"] == 0
    result = service.query("How many days of annual leave?")
    assert not result["refused"] and "25 days" in result["answer"]
    assert result["citations"][0]["source"] == "people.md"
    old_id = result["citations"][0]["chunk_id"]
    service.ingest("people.md", "# Annual leave\nAnnual leave is 30 days.")
    restarted = RAGService(database_settings)
    result = restarted.query("Annual leave days?")
    assert "30 days" in result["answer"]
    assert result["citations"][0]["chunk_id"] != old_id
    with connection(database_settings) as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM request_usage").fetchone()["n"] == 5


def test_refusal_and_zero_query_are_logged(database_settings):
    service = RAGService(database_settings)
    service.ingest("guide.md", "# Equipment\nLaptop monitor keyboard desk.")
    for question in ["Galactic volcano recipes?", "the and is"]:
        result = service.query(question)
        assert result["refused"] and not result["citations"]
    with connection(database_settings) as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM request_usage WHERE refused").fetchone()["n"]
            == 2
        )


def test_threshold_is_configurable(database_settings):
    service = RAGService(database_settings.model_copy(update={"similarity_threshold": 1.0}))
    service.ingest("guide.md", "# Leave\nAnnual leave allowance is 25 days.")
    assert service.query("leave")["refused"]


def test_concurrent_ingest_is_idempotent(database_settings):
    def ingest(_):
        return RAGService(database_settings).ingest("same.md", "# One\nA stable document body.")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ingest, range(4)))
    assert sum(not r["unchanged"] for r in results) == 1
    with connection(database_settings) as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 1


def test_embedding_mode_mismatch_fails(database_settings):
    with pytest.raises(ValueError, match="Embedding configuration differs"):
        initialize(database_settings.model_copy(update={"embedding_dim": 768}))


def test_failed_replacement_preserves_previous_document(database_settings, monkeypatch):
    service = RAGService(database_settings)
    service.ingest("a.md", "# Leave\nAnnual leave is 25 days.")

    def fail(_):
        raise ValueError("invalid provider result")

    monkeypatch.setattr(service.provider, "embed", fail)
    with pytest.raises(ValueError):
        service.ingest("a.md", "# Leave\nAnnual leave is 40 days.")
    with connection(database_settings) as conn:
        assert "25 days" in conn.execute("SELECT body FROM chunks").fetchone()["body"]


def test_http_upload_query_usage_and_validation(database_settings, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "get_settings", lambda: database_settings)
    get_settings.cache_clear()
    with TestClient(app) as client:
        assert client.get("/api/health").json()["mode"] == "mock"
        assert client.post("/api/query", json={"question": "   "}).status_code == 422
        assert client.post("/api/ingest", files={"file": ("x.pdf", b"not pdf")}).status_code == 422
        assert client.post("/api/ingest", files={"file": ("x.txt", b"\xff")}).status_code == 422
        assert client.post("/api/ingest", files={"file": ("x.txt", b"\x00")}).status_code == 422
        assert client.post("/api/ingest", files={"file": ("x.md", b"# Empty")}).status_code == 422
        assert (
            client.post("/api/ingest", files={"file": ("x.txt", b"a" * 1_000_001)}).status_code
            == 413
        )
        response = client.post(
            "/api/ingest", files={"file": ("../../people.md", b"# Leave\nAnnual leave is 25 days.")}
        )
        assert response.status_code == 200
        assert response.json()["source"] == "people.md"
        answer = client.post("/api/query", json={"question": "Annual leave days?"}).json()
        assert answer["citations"][0]["quote"] == "Annual leave is 25 days."
        assert len(client.get("/api/documents").json()) == 1
        assert sum(row["requests"] for row in client.get("/api/usage").json()) == 2


def test_evaluation_isolated_from_uploaded_documents(database_settings):
    from eval.run import evaluate, evaluation_database, metrics

    service = RAGService(database_settings)
    service.ingest("people.md", "# Custom policy\nOur private policy stays intact.")
    with evaluation_database(database_settings) as isolated:
        summary = metrics(evaluate(isolated))
    assert summary["questions"] == 26
    assert summary["source_hit_rate"] >= 0.9
    with connection(database_settings) as conn:
        assert conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"] == 1
        assert "private policy" in conn.execute("SELECT body FROM chunks").fetchone()["body"]


def test_mock_hash_collision_without_shared_words_refuses(database_settings, monkeypatch):
    service = RAGService(database_settings)
    service.ingest("guide.md", "# Internet\nHome internet reimbursement is 30 EUR.")
    # Force maximum vector similarity for completely unrelated words.
    vectors, _ = service.provider.embed(["Internet Home internet reimbursement is 30 EUR."])
    monkeypatch.setattr(service.provider, "embed", lambda _: (vectors, 4))
    result = service.query("Intergalactic chess championship")
    assert result["refused"]
    assert result["citations"] == []
