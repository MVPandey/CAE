"""Simple unit tests for Redis manager without actual Redis dependency."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.cache.redis_manager import RedisManager


class TestRedisManagerBatchMethods:
    """Test batch operations in Redis manager."""

    @pytest.mark.asyncio
    async def test_batch_get_success(self):
        """Test successful batch get."""
        manager = RedisManager()
        manager._is_healthy = True

        mock_pipeline = AsyncMock()
        mock_pipeline.get = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[b"value1", None, b"value3"])

        mock_client = AsyncMock()
        mock_client.pipeline.return_value = mock_pipeline
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_pipeline_ctx():
            yield mock_pipeline

        manager.pipeline = mock_pipeline_ctx

        result = await manager.batch_get(["key1", "key2", "key3"])

        assert result == {"key1": b"value1", "key2": None, "key3": b"value3"}
        assert mock_pipeline.get.call_count == 3
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_get_unhealthy(self):
        """Test batch get when Redis is unhealthy."""
        manager = RedisManager()
        manager._is_healthy = False

        result = await manager.batch_get(["key1", "key2"])
        assert result == {"key1": None, "key2": None}

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_batch_set_success(self, mock_redis):
        """Test successful batch set."""
        mock_client = MagicMock()  # Use MagicMock for client
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, True])

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client.pipeline.return_value = mock_pipeline

        mock_redis.ConnectionPool.return_value = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping = AsyncMock()

        manager = RedisManager()
        await manager.initialize()
        manager._is_healthy = True

        await manager.batch_set({"key1": "value1", "key2": "value2"}, ttl=60)

        assert mock_pipeline.setex.call_count == 2
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_set_unhealthy(self):
        """Test batch set when Redis is unhealthy."""
        manager = RedisManager()
        manager._is_healthy = False

        await manager.batch_set({"key1": "value1"})

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_batch_delete_success(self, mock_redis):
        """Test successful batch delete."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.delete.return_value = 2
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        manager = RedisManager()
        await manager.initialize()
        manager._is_healthy = True
        manager._client = mock_client

        result = await manager.batch_delete(["key1", "key2", "key3"])

        assert result == 2
        mock_client.delete.assert_called_once_with("key1", "key2", "key3")

    @pytest.mark.asyncio
    async def test_batch_delete_unhealthy(self):
        """Test batch delete when Redis is unhealthy."""
        manager = RedisManager()
        manager._is_healthy = False

        result = await manager.batch_delete(["key1", "key2"])
        assert result == 0

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_scan_keys_success(self, mock_redis):
        """Test successful key scanning."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        async def mock_scan_iter(match=None, count=None):
            for key in [b"test:key1", b"test:key2"]:
                yield key

        mock_client.scan_iter = mock_scan_iter

        manager = RedisManager()
        await manager.initialize()
        manager._is_healthy = True
        manager._client = mock_client

        keys = []
        async for key in manager.scan_keys("test:*"):
            keys.append(key)

        assert keys == ["test:key1", "test:key2"]

    @pytest.mark.asyncio
    async def test_scan_keys_unhealthy(self):
        """Test key scanning when Redis is unhealthy."""
        manager = RedisManager()
        manager._is_healthy = False

        keys = []
        async for key in manager.scan_keys("test:*"):
            keys.append(key)

        assert keys == []

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_exists_many_success(self, mock_redis):
        """Test checking existence of multiple keys."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.exists.return_value = 2  # 2 keys exist
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        manager = RedisManager()
        await manager.initialize()
        manager._is_healthy = True
        manager._client = mock_client

        result = await manager.exists_many(["key1", "key2", "key3"])

        assert result == {"key1": True, "key2": True, "key3": False}
        mock_client.exists.assert_called_once_with("key1", "key2", "key3")

    @pytest.mark.asyncio
    async def test_exists_many_unhealthy(self):
        """Test exists many when Redis is unhealthy."""
        manager = RedisManager()
        manager._is_healthy = False

        result = await manager.exists_many(["key1", "key2"])
        assert result == {"key1": False, "key2": False}

    def test_get_connection_info(self):
        """Test getting connection info."""
        manager = RedisManager()
        manager._is_healthy = True
        manager._pool = MagicMock()
        manager._pool.connection_kwargs = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
        }
        manager._pool.max_connections = 20

        info = manager.get_connection_info()

        assert info["host"] == "localhost"
        assert info["port"] == 6379
        assert info["db"] == 0
        assert info["pool_size"] == 20
        assert info["is_healthy"] is True
