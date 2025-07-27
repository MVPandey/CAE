"""Unit tests for semantic cache."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schema.llm.message import Message
from app.services.cache.semantic_cache import CacheEntry, SemanticCache


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    manager = AsyncMock()
    manager.get_json = AsyncMock(return_value=None)
    manager.set_json = AsyncMock(return_value=True)
    manager.scan_keys = AsyncMock(return_value=[])
    manager.delete = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = AsyncMock()
    service.embed_conversation = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    service.cosine_similarity = MagicMock(return_value=0.9)
    service.clear_cache = AsyncMock(return_value=5)
    return service


@pytest.fixture
def semantic_cache(mock_redis_manager, mock_embedding_service):
    """Create semantic cache with mocks."""
    with patch("app.services.cache.semantic_cache.redis_manager", mock_redis_manager):
        with patch("app.services.cache.semantic_cache.embedding_service", mock_embedding_service):
            cache = SemanticCache()
            yield cache


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you!"),
        Message(role="user", content="Can you help me with Python?"),
    ]


@pytest.fixture
def sample_cache_data():
    """Sample cache data."""
    return {
        "conversation_hash": "abc123",
        "embedding": [0.1, 0.2, 0.3],
        "response": "I'd be happy to help with Python!",
        "metadata": {"message_count": 3},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "access_count": 0,
    }


class TestSemanticCache:
    """Test semantic cache functionality."""

    def test_generate_conversation_hash(self, semantic_cache, sample_messages):
        """Test conversation hash generation."""
        hash1 = semantic_cache._generate_conversation_hash(sample_messages)
        hash2 = semantic_cache._generate_conversation_hash(sample_messages)

        assert hash1 == hash2
        assert len(hash1) == 16

        different_messages = [Message(role="user", content="Different content")]
        hash3 = semantic_cache._generate_conversation_hash(different_messages)
        assert hash1 != hash3

    def test_generate_cache_key(self, semantic_cache):
        """Test cache key generation."""
        hash_val = "abc123"

        key = semantic_cache._generate_cache_key(hash_val)
        assert key == "mcts_cache:abc123"

        key = semantic_cache._generate_cache_key(hash_val, "simulation")
        assert key == "mcts_cache:abc123:simulation"

    async def test_store_in_index(self, semantic_cache, mock_redis_manager):
        """Test storing entry in similarity index."""
        hash_val = "abc123"
        embedding = np.array([0.1, 0.2, 0.3])
        metadata = {"message_count": 3}

        await semantic_cache._store_in_index(hash_val, embedding, metadata)

        mock_redis_manager.set_json.assert_called_once()
        call_args = mock_redis_manager.set_json.call_args
        assert call_args[0][0] == "mcts_index:abc123"
        assert "embedding" in call_args[0][1]
        assert call_args[0][1]["embedding"] == [0.1, 0.2, 0.3]

    async def test_find_similar_entries(self, semantic_cache, mock_redis_manager, mock_embedding_service):
        """Test finding similar entries."""

        async def mock_scan_keys(pattern, count=100):
            for key in ["mcts_index:hash1", "mcts_index:hash2", "mcts_index:entries"]:
                yield key

        mock_redis_manager.scan_keys = mock_scan_keys

        async def mock_get_json(key):
            if key == "mcts_index:hash1":
                return {"hash": "hash1", "embedding": [0.1, 0.2, 0.3]}
            elif key == "mcts_index:hash2":
                return {"hash": "hash2", "embedding": [0.4, 0.5, 0.6]}
            return None

        mock_redis_manager.get_json.side_effect = mock_get_json

        mock_embedding_service.cosine_similarity.side_effect = [0.95, 0.90]

        embedding = np.array([0.1, 0.2, 0.3])

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            results = await semantic_cache._find_similar_entries(embedding)

        assert len(results) == 2
        assert results[0] == ("hash1", 0.95)
        assert results[1] == ("hash2", 0.90)

    async def test_find_similar_entries_below_threshold(
        self, semantic_cache, mock_redis_manager, mock_embedding_service
    ):
        """Test finding similar entries filters by threshold."""

        async def mock_scan_keys(pattern, count=100):
            yield "mcts_index:hash1"

        mock_redis_manager.scan_keys = mock_scan_keys
        mock_redis_manager.get_json.return_value = {"hash": "hash1", "embedding": [0.1, 0.2, 0.3]}

        mock_embedding_service.cosine_similarity.return_value = 0.5
        semantic_cache.similarity_threshold = 0.85

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            results = await semantic_cache._find_similar_entries(np.array([0.1, 0.2, 0.3]))

        assert len(results) == 0

    async def test_get_exact_match(self, semantic_cache, mock_redis_manager, sample_messages, sample_cache_data):
        """Test getting exact match from cache."""

        async def mock_get_json(key):
            if key.endswith(":simulation"):
                return {"simulation": "data"}
            elif key.endswith(":score"):
                return {"score": 0.9}
            else:
                return sample_cache_data

        mock_redis_manager.get_json.side_effect = mock_get_json

        result = await semantic_cache.get(sample_messages)

        assert result is not None
        assert isinstance(result, CacheEntry)
        assert result.response == "I'd be happy to help with Python!"
        assert semantic_cache._stats["exact_hits"] == 1

    async def test_get_exact_match_response_only(
        self, semantic_cache, mock_redis_manager, sample_messages, sample_cache_data
    ):
        """Test getting exact match with response_only flag."""
        mock_redis_manager.get_json.return_value = sample_cache_data

        result = await semantic_cache.get(sample_messages, response_only=True)

        assert result is not None
        assert result.response == "I'd be happy to help with Python!"
        assert result.simulation_data == {}
        assert result.score_data == {}

    async def test_get_similarity_match(
        self, semantic_cache, mock_redis_manager, mock_embedding_service, sample_messages, sample_cache_data
    ):
        """Test getting similar match from cache."""

        async def mock_scan_keys(pattern, count=100):
            if "mcts_index:" in pattern:
                yield "mcts_index:similar_hash"

        mock_redis_manager.scan_keys = mock_scan_keys

        async def mock_get_json(key):
            if "mcts_cache:" in key and key.endswith(":simulation"):
                return {"simulation": "data"}
            elif "mcts_cache:" in key and key.endswith(":score"):
                return {"score": 0.9}
            elif "mcts_cache:" in key and not ("simulation" in key or "score" in key):
                if not hasattr(mock_get_json, "_exact_called"):
                    mock_get_json._exact_called = True
                    return None
                return sample_cache_data
            elif "mcts_index:" in key:
                return {"hash": "similar_hash", "embedding": [0.1, 0.2, 0.3]}
            return None

        mock_redis_manager.get_json.side_effect = mock_get_json

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            result = await semantic_cache.get(sample_messages)

        assert result is not None
        assert result.metadata.get("similarity") == 0.9
        assert semantic_cache._stats["similarity_hits"] == 1

    async def test_get_cache_miss(self, semantic_cache, mock_redis_manager, mock_embedding_service, sample_messages):
        """Test cache miss."""
        mock_redis_manager.get_json.return_value = None

        async def mock_scan_keys(pattern, count=100):
            return
            yield

        mock_redis_manager.scan_keys = mock_scan_keys

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            result = await semantic_cache.get(sample_messages)

        assert result is None
        assert semantic_cache._stats["misses"] == 1

    async def test_get_embedding_failure(
        self, semantic_cache, mock_redis_manager, mock_embedding_service, sample_messages
    ):
        """Test handling embedding failure."""
        mock_redis_manager.get_json.return_value = None
        mock_embedding_service.embed_conversation.return_value = None

        async def mock_scan_keys(pattern, count=100):
            if False:
                yield

        mock_redis_manager.scan_keys = mock_scan_keys

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            result = await semantic_cache.get(sample_messages)

        assert result is None
        assert semantic_cache._stats["misses"] == 1

    async def test_store_success(self, semantic_cache, mock_redis_manager, mock_embedding_service, sample_messages):
        """Test storing entry in cache."""
        response = "Test response"
        simulation_data = {"simulation": "data"}
        score_data = {"score": 0.9}

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            success = await semantic_cache.store(
                sample_messages,
                response,
                simulation_data,
                score_data,
            )

        assert success is True
        assert semantic_cache._stats["stores"] == 1

        assert mock_redis_manager.set_json.call_count == 4

    async def test_store_with_metadata(
        self, semantic_cache, mock_redis_manager, mock_embedding_service, sample_messages
    ):
        """Test storing entry with custom metadata."""
        metadata = {"custom_field": "value"}

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            success = await semantic_cache.store(
                sample_messages,
                "Response",
                {},
                {},
                metadata,
            )

        assert success is True

        main_call = mock_redis_manager.set_json.call_args_list[0]
        stored_data = main_call[0][1]
        assert stored_data["metadata"]["custom_field"] == "value"
        assert stored_data["metadata"]["message_count"] == 3

    async def test_store_embedding_failure(
        self, semantic_cache, mock_redis_manager, mock_embedding_service, sample_messages
    ):
        """Test handling embedding failure during store."""
        mock_embedding_service.embed_conversation.return_value = None

        with patch("app.services.cache.semantic_cache.embedding_service", mock_embedding_service):
            success = await semantic_cache.store(
                sample_messages,
                "Response",
                {},
                {},
            )

        assert success is False

    async def test_update_access_stats(self, semantic_cache, mock_redis_manager):
        """Test updating access statistics."""
        hash_val = "abc123"
        existing_data = {
            "access_count": 5,
            "other_field": "value",
        }
        mock_redis_manager.get_json.return_value = existing_data

        await semantic_cache._update_access_stats(hash_val)

        updated_call = mock_redis_manager.set_json.call_args
        updated_data = updated_call[0][1]
        assert updated_data["access_count"] == 6
        assert "last_accessed" in updated_data

    async def test_invalidate(self, semantic_cache, mock_redis_manager, sample_messages):
        """Test invalidating cache entries."""
        mock_redis_manager.delete.side_effect = [True, True, False, True]

        success = await semantic_cache.invalidate(sample_messages)

        assert success is True
        assert mock_redis_manager.delete.call_count == 4

    async def test_clear_all(self, semantic_cache, mock_redis_manager):
        """Test clearing all cache entries."""
        call_count = 0

        async def mock_scan_keys(pattern, count=100):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                for key in ["key1", "key2", "key3"]:
                    yield key
            elif call_count == 2:
                for key in ["key4", "key5"]:
                    yield key

        mock_redis_manager.scan_keys = mock_scan_keys
        mock_redis_manager.delete.return_value = True

        count = await semantic_cache.clear_all()

        assert count == 5
        assert call_count == 2

    def test_get_stats(self, semantic_cache):
        """Test getting cache statistics."""
        semantic_cache._stats = {
            "exact_hits": 50,
            "similarity_hits": 20,
            "misses": 30,
            "stores": 70,
        }

        stats = semantic_cache.get_stats()

        assert stats["exact_hits"] == 50
        assert stats["similarity_hits"] == 20
        assert stats["misses"] == 30
        assert stats["stores"] == 70
        assert stats["total_requests"] == 100
        assert stats["hit_rate"] == 0.7

    def test_get_stats_no_requests(self, semantic_cache):
        """Test getting stats with no requests."""
        stats = semantic_cache.get_stats()

        assert stats["total_requests"] == 0
        assert stats["hit_rate"] == 0

    async def test_warm_cache(self, semantic_cache, mock_redis_manager, mock_embedding_service):
        """Test cache warming functionality."""
        conversation_patterns = [
            [Message(role="user", content="Pattern 1")],
            [Message(role="user", content="Pattern 2")],
        ]

        mock_redis_manager.get_json.return_value = None

        async def mock_scan_keys(pattern, count=100):
            return
            yield

        mock_redis_manager.scan_keys = mock_scan_keys

        async def mock_generator(messages):
            return {
                "response": f"Response for {messages[0].content}",
                "simulation_data": {},
                "score_data": {},
            }

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            count = await semantic_cache.warm_cache(conversation_patterns, mock_generator)

        assert count == 2
        assert semantic_cache._stats["stores"] == 2

    async def test_warm_cache_skip_existing(self, semantic_cache, mock_redis_manager, mock_embedding_service):
        """Test cache warming skips existing entries."""
        conversation_patterns = [
            [Message(role="user", content="Pattern 1")],
        ]

        mock_redis_manager.get_json.return_value = {
            "response": "Cached",
            "embedding": [0.1, 0.2, 0.3],
            "conversation_hash": "abc123",
            "metadata": {"message_count": 1},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        async def mock_generator(messages):
            return {"response": "New", "simulation_data": {}, "score_data": {}}

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            count = await semantic_cache.warm_cache(conversation_patterns, mock_generator)

        assert count == 0

    async def test_warm_cache_generator_failure(self, semantic_cache, mock_redis_manager, mock_embedding_service):
        """Test cache warming handles generator failures."""
        conversation_patterns = [
            [Message(role="user", content="Pattern 1")],
        ]

        mock_redis_manager.get_json.return_value = None

        async def mock_scan_keys(pattern, count=100):
            return
            yield

        mock_redis_manager.scan_keys = mock_scan_keys

        async def mock_generator(messages):
            raise Exception("Generator error")

        with patch("app.services.embeddings.embedding_service.embedding_service", mock_embedding_service):
            count = await semantic_cache.warm_cache(conversation_patterns, mock_generator)

        assert count == 0
