import numpy as np
import pytest

from app.config import Settings
from app.providers import Provider


def test_mock_embeddings_are_stable_normalized_and_discriminate():
    provider = Provider(Settings(mock_llm=True))
    vectors, units = provider.embed(["annual leave allowance", "annual leave", "volcanic lava"])
    assert provider.embed(["annual leave allowance"])[0][0] == vectors[0]
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert np.dot(vectors[0], vectors[1]) > np.dot(vectors[0], vectors[2])
    assert units == 7
    assert not any(provider.embed(["the and is"])[0][0])


def test_mock_answer_uses_exact_evidence():
    provider = Provider(Settings(mock_llm=True))
    text = "Welcome to the team. Annual leave is 25 days."
    response = provider.generate("annual leave?", [{"id": "abc", "text": text}])
    assert response.citations == [{"chunk_id": "abc", "quote": text}]
    assert "[1]" in response.answer


@pytest.mark.parametrize(
    "content",
    [
        '{"answer":"Yes","citations":[{"chunk_id":"invented","quote":"No"}]}',
        '{"answer":"Yes","citations":[{"chunk_id":"abc","quote":"invented quote"}]}',
        '{"answer":"Yes","citations":[]}',
        '{"answer":"Yes","citations":[null]}',
        "not json",
    ],
)
def test_real_provider_fails_closed_on_unverifiable_citations(monkeypatch, content):
    provider = Provider(Settings(mock_llm=False, openai_api_key="test-not-a-real-key"))
    monkeypatch.setattr(
        provider,
        "_post",
        lambda *args: {
            "usage": {"total_tokens": 12},
            "choices": [{"message": {"content": content}}],
        },
    )
    response = provider.generate("Leave?", [{"id": "abc", "text": "25 days."}])
    assert response.refused
    assert response.citations == []
    assert response.units == 12


def test_real_provider_accepts_exact_quote(monkeypatch):
    provider = Provider(Settings(mock_llm=False, openai_api_key="test-not-a-real-key"))
    monkeypatch.setattr(
        provider,
        "_post",
        lambda *args: {
            "usage": {"total_tokens": 8},
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer":"25 days [1]",'
                            '"citations":[{"chunk_id":"abc","quote":"25 days."}]}'
                        )
                    }
                }
            ],
        },
    )
    response = provider.generate("Leave?", [{"id": "abc", "text": "25 days."}])
    assert not response.refused


def test_real_mode_requires_key():
    with pytest.raises(ValueError, match="requires OPENAI_API_KEY"):
        Provider(Settings(mock_llm=False, openai_api_key=""))


def test_real_embeddings_preserve_input_order_and_meter_tokens(monkeypatch):
    provider = Provider(
        Settings(mock_llm=False, openai_api_key="test-not-a-real-key", embedding_dim=32)
    )

    def response(route, payload):
        assert route == "/embeddings"
        assert payload["input"] == ["first", "second"]
        assert payload["dimensions"] == 32
        return {
            "data": [{"index": 1, "embedding": [0.5] * 32}, {"index": 0, "embedding": [0.25] * 32}],
            "usage": {"total_tokens": 4},
        }

    monkeypatch.setattr(provider, "_post", response)
    vectors, units = provider.embed(["first", "second"])
    assert vectors[0] == [0.25] * 32
    assert units == 4


def test_real_embeddings_reject_wrong_dimensions(monkeypatch):
    provider = Provider(Settings(mock_llm=False, openai_api_key="test-not-a-real-key"))
    monkeypatch.setattr(
        provider,
        "_post",
        lambda *args: {
            "data": [{"index": 0, "embedding": [1.0]}],
            "usage": {"total_tokens": 1},
        },
    )
    with pytest.raises(ValueError, match="Invalid embedding"):
        provider.embed(["test"])
