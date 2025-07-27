"""Edge case tests for embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.services.embeddings.embedding_service import EmbeddingService


class TestEmbeddingServiceEdgeCases:
    """Test edge cases for embedding service."""

    @pytest.mark.asyncio
    @patch("app.services.embeddings.embedding_service.redis_manager")
    @patch("app.services.embeddings.embedding_service.logger")
    async def test_clear_cache_exception_handling(self, mock_logger, mock_redis):
        """Test clear_cache handles exceptions gracefully."""
        service = EmbeddingService()

        mock_redis.scan_keys.side_effect = Exception("Redis error")

        result = await service.clear_cache()
        assert result == 0

        mock_logger.error.assert_called_once_with("Error clearing cache: Redis error")

    @pytest.mark.asyncio
    @patch("app.services.embeddings.embedding_service.redis_manager")
    async def test_embed_texts_empty_list(self, mock_redis):
        """Test embed_texts with empty list."""
        service = EmbeddingService()

        result = await service.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    @patch("app.services.embeddings.embedding_service.AsyncOpenAI")
    async def test_embed_text_api_returns_none(self, mock_openai):
        """Test embed_text when API returns None."""
        service = EmbeddingService()
        service._client = AsyncMock()
        service._client.embeddings.create.side_effect = Exception("API error")

        result = await service.embed_text("test", use_cache=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_cosine_similarity_zero_norm(self):
        """Test cosine_similarity with zero norm vectors."""
        service = EmbeddingService()

        embedding1 = np.array([0, 0, 0])
        embedding2 = np.array([1, 2, 3])

        similarity = service.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    @pytest.mark.asyncio
    @patch("app.services.embeddings.embedding_service.AsyncOpenAI")
    @patch("app.services.embeddings.embedding_service.redis_manager")
    async def test_embed_texts_partial_cache_hit(self, mock_redis, mock_openai_class):
        """Test embed_texts with partial cache hits."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[4.0, 5.0, 6.0])]
        mock_response.usage.total_tokens = 10
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        service = EmbeddingService()

        mock_redis.get_json = AsyncMock(
            side_effect=[
                {"embedding": [1.0, 2.0, 3.0]},  # First text cached
                None,  # Second text not cached
            ]
        )
        mock_redis.set_json = AsyncMock(return_value=True)

        result = await service.embed_texts(["text1", "text2"])

        assert len(result) == 2
        assert np.array_equal(result[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert np.array_equal(result[1], np.array([4.0, 5.0, 6.0], dtype=np.float32))

    @pytest.mark.asyncio
    async def test_get_stats_calculation(self):
        """Test get_stats calculations."""
        service = EmbeddingService()

        service._stats = {
            "total_requests": 100,
            "cache_hits": 60,
            "api_calls": 40,
            "total_tokens": 1000,
        }

        stats = service.get_stats()

        assert stats["total_requests"] == 100
        assert stats["cache_hits"] == 60
        assert stats["api_calls"] == 40
        assert stats["total_tokens"] == 1000
        assert stats["cache_hit_rate"] == 0.6
        assert stats["model"] == service.model_name
        assert stats["dimension"] == service.embedding_dimension
