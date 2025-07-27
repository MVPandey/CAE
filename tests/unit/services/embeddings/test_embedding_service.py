"""Unit tests for embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schema.llm.message import Message
from app.services.embeddings.embedding_service import EmbeddingService


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = AsyncMock()

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.usage.total_tokens = 10

    client.embeddings.create = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    manager = AsyncMock()
    manager.get_json = AsyncMock(return_value=None)
    manager.set_json = AsyncMock(return_value=True)

    async def mock_scan_keys(pattern, count=100):
        for key in ["key1", "key2"]:
            yield key

    manager.scan_keys = mock_scan_keys
    manager.delete = AsyncMock(return_value=True)
    return manager


@pytest.fixture
async def embedding_service(mock_openai_client, mock_redis_manager):
    """Create embedding service with mocks."""
    with patch("app.services.embeddings.embedding_service.AsyncOpenAI", return_value=mock_openai_client):
        with patch("app.services.embeddings.embedding_service.redis_manager", mock_redis_manager):
            service = EmbeddingService()
            return service


class TestEmbeddingService:
    """Test embedding service functionality."""

    def test_get_embedding_dimension(self):
        """Test getting embedding dimensions for different models."""
        with patch("app.services.embeddings.embedding_service.AsyncOpenAI"):
            with patch("app.services.embeddings.embedding_service.app_settings") as mock_settings:
                mock_settings.EMBEDDING_MODEL_NAME = "text-embedding-3-small"
                service = EmbeddingService()
                assert service.embedding_dimension == 1536

                mock_settings.EMBEDDING_MODEL_NAME = "text-embedding-3-large"
                service = EmbeddingService()
                assert service.embedding_dimension == 3072

                mock_settings.EMBEDDING_MODEL_NAME = "unknown-model"
                service = EmbeddingService()
                assert service.embedding_dimension == 3072

    def test_hash_text(self, embedding_service):
        """Test text hashing functionality."""
        text = "Hello, world!"
        hash1 = embedding_service._hash_text(text)
        hash2 = embedding_service._hash_text(text)

        assert hash1 == hash2

        hash3 = embedding_service._hash_text("Different text")
        assert hash1 != hash3

    def test_prepare_conversation_text(self, embedding_service):
        """Test conversation text preparation."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]

        result = embedding_service._prepare_conversation_text(messages)
        expected = "user: Hello\nassistant: Hi there!\nuser: How are you?"

        assert result == expected

    async def test_embed_text_no_cache(self, embedding_service, mock_openai_client):
        """Test embedding text without cache."""
        text = "Test text"

        result = await embedding_service.embed_text(text, use_cache=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)  # Based on mock embedding
        assert np.array_equal(result, np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))

        mock_openai_client.embeddings.create.assert_called_once()

        assert embedding_service._stats["total_requests"] == 1
        assert embedding_service._stats["api_calls"] == 1
        assert embedding_service._stats["cache_hits"] == 0

    async def test_embed_text_with_cache_miss(self, embedding_service, mock_redis_manager):
        """Test embedding text with cache miss."""
        text = "Test text"
        mock_redis_manager.get_json.return_value = None  # Cache miss

        with patch("app.services.embeddings.embedding_service.redis_manager", mock_redis_manager):
            result = await embedding_service.embed_text(text, use_cache=True)

        assert isinstance(result, np.ndarray)

        cache_key = f"embedding:{embedding_service.model_name}:{embedding_service._hash_text(text)}"
        mock_redis_manager.get_json.assert_called_with(cache_key)
        mock_redis_manager.set_json.assert_called_once()

    async def test_embed_text_with_cache_hit(self, embedding_service, mock_redis_manager):
        """Test embedding text with cache hit."""
        text = "Test text"
        cached_embedding = {"embedding": [0.6, 0.7, 0.8], "text_hash": "hash"}
        mock_redis_manager.get_json.return_value = cached_embedding

        result = await embedding_service.embed_text(text, use_cache=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)  # Based on mock

    async def test_embed_text_api_failure(self, embedding_service, mock_openai_client):
        """Test embedding text when API fails."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")

        result = await embedding_service.embed_text("Test text", use_cache=False)

        assert result is None

    async def test_embed_texts_batch(self, embedding_service, mock_openai_client):
        """Test batch embedding of texts."""
        texts = ["Text 1", "Text 2", "Text 3"]

        mock_embeddings = []
        for i in range(3):
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1 * i, 0.2 * i, 0.3 * i]
            mock_embeddings.append(mock_embedding)

        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_response.usage.total_tokens = 30
        mock_openai_client.embeddings.create.return_value = mock_response

        results = await embedding_service.embed_texts(texts, use_cache=False)

        assert len(results) == 3
        assert all(isinstance(r, np.ndarray) for r in results)

    async def test_embed_texts_with_mixed_cache(self, embedding_service, mock_redis_manager, mock_openai_client):
        """Test batch embedding with mixed cache."""
        texts = ["Cached 1", "Uncached", "Cached 2"]

        mock_embeddings = []
        for i in range(3):
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)]
            mock_embeddings.append(mock_embedding)

        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_response.usage.total_tokens = 30
        mock_openai_client.embeddings.create.return_value = mock_response

        results = await embedding_service.embed_texts(texts, use_cache=False)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, np.ndarray)
            expected = np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)], dtype=np.float32)
            assert np.array_equal(result, expected)

    async def test_embed_texts_large_batch(self, embedding_service, mock_openai_client):
        """Test batch embedding with size larger than batch_size."""
        texts = [f"Text {i}" for i in range(250)]  # Larger than default batch_size of 100

        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(100)], usage=MagicMock(total_tokens=100)
        )

        results = await embedding_service.embed_texts(texts, use_cache=False, batch_size=100)

        assert len(results) == 250
        assert mock_openai_client.embeddings.create.call_count == 3

    async def test_embed_conversation(self, embedding_service):
        """Test embedding a conversation."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]

        result = await embedding_service.embed_conversation(messages, use_cache=False)

        assert isinstance(result, np.ndarray)

    def test_cosine_similarity(self, embedding_service):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = embedding_service.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = embedding_service.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = embedding_service.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0)

        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = embedding_service.cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_get_stats(self, embedding_service):
        """Test getting service statistics."""
        embedding_service._stats = {
            "total_requests": 100,
            "cache_hits": 30,
            "api_calls": 70,
            "total_tokens": 1000,
        }

        stats = embedding_service.get_stats()

        assert stats["total_requests"] == 100
        assert stats["cache_hits"] == 30
        assert stats["api_calls"] == 70
        assert stats["total_tokens"] == 1000
        assert stats["cache_hit_rate"] == 0.3
        assert stats["model"] == embedding_service.model_name
        assert stats["dimension"] == embedding_service.embedding_dimension

    async def test_clear_cache(self, embedding_service, mock_redis_manager):
        """Test clearing embedding cache."""

        async def mock_scan_keys(pattern, count=100):
            for key in ["key1", "key2", "key3"]:
                yield key

        mock_redis_manager.scan_keys = mock_scan_keys
        mock_redis_manager.delete.side_effect = [True, True, False]  # 2 successful, 1 failed

        with patch("app.services.embeddings.embedding_service.redis_manager", mock_redis_manager):
            count = await embedding_service.clear_cache()

        assert count == 2

    async def test_clear_cache_with_pattern(self, embedding_service, mock_redis_manager):
        """Test clearing cache with specific pattern."""
        pattern = "user_*"

        async def mock_scan_keys(pattern_arg, count=100):
            if "user_*" in pattern_arg:
                yield "key1"

        mock_redis_manager.scan_keys = mock_scan_keys
        mock_redis_manager.delete.return_value = True

        with patch("app.services.embeddings.embedding_service.redis_manager", mock_redis_manager):
            count = await embedding_service.clear_cache(pattern)

        assert count == 1

    async def test_retry_on_api_failure(self, embedding_service, mock_openai_client):
        """Test retry logic on API failures."""
        with patch("app.services.embeddings.embedding_service.wait_exponential", return_value=0):
            mock_openai_client.embeddings.create.side_effect = [
                Exception("Temporary failure"),
                Exception("Another failure"),
                MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])], usage=MagicMock(total_tokens=10)),
            ]

            result = await embedding_service.embed_text("Test", use_cache=False)

            assert result is not None
            assert mock_openai_client.embeddings.create.call_count == 3
