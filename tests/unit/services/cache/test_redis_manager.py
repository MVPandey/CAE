"""Unit tests for Redis manager with mocked async operations."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from app.services.cache.redis_manager import RedisManager


@pytest.fixture
def mock_redis_pool():
    """Mock Redis connection pool."""
    pool = MagicMock()
    pool.disconnect = AsyncMock()
    return pool


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.setex = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=1)

    async def async_iter(items):
        for item in items:
            yield item

    client.scan_iter = AsyncMock(return_value=async_iter(["key1", "key2"]))
    client.info = AsyncMock(
        return_value={
            "redis_version": "7.0.0",
            "used_memory_human": "10M",
            "connected_clients": 5,
            "total_connections_received": 100,
            "instantaneous_ops_per_sec": 50,
        }
    )
    client.close = AsyncMock()
    return client


@pytest.fixture
async def redis_manager(mock_redis_pool, mock_redis_client):
    """Create Redis manager with mocks."""
    manager = RedisManager()

    with patch("app.services.cache.redis_manager.redis.ConnectionPool", return_value=mock_redis_pool):
        with patch("app.services.cache.redis_manager.redis.Redis", return_value=mock_redis_client):
            with patch.object(manager, "_health_check_loop", new_callable=AsyncMock):
                await manager.initialize()

    return manager


class TestRedisManager:
    """Test Redis manager functionality."""

    async def test_initialize_success(self, mock_redis_pool, mock_redis_client):
        """Test successful initialization."""
        manager = RedisManager()

        with patch("app.services.cache.redis_manager.redis.ConnectionPool", return_value=mock_redis_pool):
            with patch("app.services.cache.redis_manager.redis.Redis", return_value=mock_redis_client):
                with patch.object(manager, "_health_check_loop", new_callable=AsyncMock):
                    await manager.initialize()

        assert manager._pool is not None
        assert manager._client is not None
        assert manager._is_healthy is True
        mock_redis_client.ping.assert_called_once()

    async def test_initialize_failure(self, mock_redis_pool, mock_redis_client):
        """Test initialization failure with graceful degradation."""
        manager = RedisManager()
        mock_redis_client.ping.side_effect = RedisError("Connection failed")

        with patch("app.services.cache.redis_manager.redis.ConnectionPool", return_value=mock_redis_pool):
            with patch("app.services.cache.redis_manager.redis.Redis", return_value=mock_redis_client):
                await manager.initialize()

        assert manager._is_healthy is False

    async def test_get_success(self, redis_manager):
        """Test successful get operation."""
        redis_manager._client.get.return_value = "test_value"

        result = await redis_manager.get("test_key")

        assert result == "test_value"
        redis_manager._client.get.assert_called_with("test_key")

    async def test_get_unhealthy(self, redis_manager):
        """Test get operation when Redis is unhealthy."""
        redis_manager._is_healthy = False

        result = await redis_manager.get("test_key")

        assert result is None
        redis_manager._client.get.assert_not_called()

    async def test_get_with_retry(self, redis_manager):
        """Test get operation with retry on failure."""
        redis_manager._client.get.side_effect = RedisError("Temp failure")

        result = await redis_manager.get("test_key")

        assert result is None
        assert redis_manager._client.get.call_count >= 1

    async def test_set_success(self, redis_manager):
        """Test successful set operation."""
        result = await redis_manager.set("test_key", "test_value")

        assert result is True
        redis_manager._client.set.assert_called_with("test_key", "test_value")

    async def test_set_with_ttl(self, redis_manager):
        """Test set operation with TTL."""
        result = await redis_manager.set("test_key", "test_value", ttl=3600)

        assert result is True
        redis_manager._client.setex.assert_called_with("test_key", 3600, "test_value")

    async def test_set_json(self, redis_manager):
        """Test setting JSON values."""
        test_dict = {"key": "value", "number": 42}

        result = await redis_manager.set("test_key", test_dict)

        assert result is True
        expected_json = json.dumps(test_dict)
        redis_manager._client.set.assert_called_with("test_key", expected_json)

    async def test_delete_success(self, redis_manager):
        """Test successful delete operation."""
        result = await redis_manager.delete("test_key")

        assert result is True
        redis_manager._client.delete.assert_called_with("test_key")

    async def test_delete_key_not_found(self, redis_manager):
        """Test delete when key doesn't exist."""
        redis_manager._client.delete.return_value = 0

        result = await redis_manager.delete("test_key")

        assert result is False

    async def test_exists_success(self, redis_manager):
        """Test exists check."""
        result = await redis_manager.exists("test_key")

        assert result is True
        redis_manager._client.exists.assert_called_with("test_key")

    async def test_scan_keys(self, redis_manager):
        """Test scanning keys by pattern."""
        expected_keys = ["cache:key1", "cache:key2", "cache:key3"]

        async def async_iter():
            for item in expected_keys:
                yield item

        redis_manager._client.scan_iter = MagicMock(return_value=async_iter())

        result = []
        async for key in redis_manager.scan_keys("cache:*"):
            result.append(key)

        assert result == expected_keys
        redis_manager._client.scan_iter.assert_called_with(match="cache:*", count=100)

    async def test_get_json_success(self, redis_manager):
        """Test getting JSON values."""
        test_dict = {"key": "value", "number": 42}
        redis_manager._client.get.return_value = json.dumps(test_dict)

        result = await redis_manager.get_json("test_key")

        assert result == test_dict

    async def test_get_json_invalid(self, redis_manager):
        """Test getting invalid JSON."""
        redis_manager._client.get.return_value = "invalid json"

        result = await redis_manager.get_json("test_key")

        assert result is None

    async def test_set_json_success(self, redis_manager):
        """Test setting JSON values."""
        test_dict = {"key": "value", "number": 42}

        result = await redis_manager.set_json("test_key", test_dict, ttl=3600)

        assert result is True
        expected_json = json.dumps(test_dict)
        redis_manager._client.setex.assert_called_with("test_key", 3600, expected_json)

    async def test_get_info_healthy(self, redis_manager):
        """Test getting Redis info when healthy."""
        result = await redis_manager.get_info()

        assert result["status"] == "healthy"
        assert result["version"] == "7.0.0"
        assert result["used_memory"] == "10M"

    async def test_get_info_unhealthy(self, redis_manager):
        """Test getting Redis info when unhealthy."""
        redis_manager._is_healthy = False

        result = await redis_manager.get_info()

        assert result["status"] == "unhealthy"

    async def test_health_check_loop(self, mock_redis_client):
        """Test health check loop functionality."""
        manager = RedisManager()
        manager._client = mock_redis_client
        manager._is_healthy = False

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:  # Stop after one ping
                raise asyncio.CancelledError()
            return

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(asyncio.CancelledError):
                await manager._health_check_loop()

        assert mock_redis_client.ping.call_count >= 1

    async def test_close(self, redis_manager):
        """Test closing Redis connections."""

        async def dummy_task():
            await asyncio.sleep(100)

        redis_manager._health_check_task = asyncio.create_task(dummy_task())

        await redis_manager.close()

        assert redis_manager._health_check_task.cancelled()
        redis_manager._client.close.assert_called_once()
        redis_manager._pool.disconnect.assert_called_once()

    async def test_context_manager_healthy(self, redis_manager):
        """Test context manager when healthy."""
        async with redis_manager.get_client() as client:
            assert client is redis_manager._client

    async def test_context_manager_unhealthy(self, redis_manager):
        """Test context manager when unhealthy."""
        redis_manager._is_healthy = False

        async with redis_manager.get_client() as client:
            assert client is None

    async def test_context_manager_redis_error(self, redis_manager):
        """Test context manager marking unhealthy on error."""
        redis_manager._is_healthy = True

        redis_manager._client.get.side_effect = RedisError("Test error")

        async with redis_manager.get_client() as client:
            if client:
                result = await redis_manager.get("test_key")
                assert result is None

        assert redis_manager._is_healthy is False
