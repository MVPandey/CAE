"""Advanced tests for Redis manager pipeline and locking functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.cache.redis_manager import RedisManager


class TestRedisManagerPipeline:
    """Test Redis manager pipeline functionality."""

    @pytest.fixture
    async def redis_manager(self):
        """Create Redis manager instance."""
        manager = RedisManager()
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_pipeline_success(self, mock_redis):
        """Test successful pipeline operations."""
        mock_client = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, b"value1", 1])

        mock_redis.ConnectionPool.return_value = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping = AsyncMock()
        mock_client.pipeline.return_value = mock_pipeline

        manager = RedisManager()
        await manager.initialize()
        manager._is_healthy = True

        async with manager.pipeline() as pipe:
            assert pipe is not None
            pipe.set("key1", "value1")
            pipe.get("key1")
            pipe.incr("counter")
            await pipe.execute()

        mock_client.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_unhealthy(self, redis_manager):
        """Test pipeline when Redis is unhealthy."""
        redis_manager._is_healthy = False

        async with redis_manager.pipeline() as pipe:
            assert pipe is None

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.logger")
    async def test_pipeline_error_handling(self, mock_logger):
        """Test pipeline error handling."""
        from redis.exceptions import RedisError

        manager = RedisManager()
        manager._is_healthy = True

        mock_client = MagicMock()
        mock_client.pipeline = MagicMock(side_effect=RedisError("Pipeline error"))

        manager._client = mock_client

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_get_client():
            yield mock_client

        manager.get_client = mock_get_client

        async with manager.pipeline() as pipe:
            assert pipe is None

        mock_logger.error.assert_called_with("Redis pipeline failed: Pipeline error")

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.logger")
    @patch("app.services.cache.redis_manager.redis")
    async def test_pipeline_metrics_tracking(self, mock_redis, mock_logger):
        """Test that pipeline operations track metrics."""
        mock_client = AsyncMock()
        mock_pipeline = MagicMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=None)

        mock_redis.from_url.return_value = mock_client
        mock_client.pipeline.return_value = mock_pipeline
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        async with manager.pipeline() as pipe:
            pass

        assert pipe is not None


class TestRedisManagerLocking:
    """Test Redis manager distributed locking."""

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_acquire_lock_success(self, mock_redis):
        """Test successful lock acquisition."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.set.return_value = True

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        result = await manager.acquire_lock("test_lock", timeout=10)

        assert result is True
        mock_client.set.assert_called_once()

        call_args = mock_client.set.call_args
        assert call_args[0][0] == "lock:test_lock"
        assert "nx" in call_args[1] and call_args[1]["nx"] is True
        assert "ex" in call_args[1] and call_args[1]["ex"] == 10

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_acquire_lock_already_held(self, mock_redis):
        """Test lock acquisition when lock is already held."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.set.return_value = False

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        result = await manager.acquire_lock("test_lock", blocking_timeout=0)

        assert result is False

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.asyncio.sleep")
    @patch("app.services.cache.redis_manager.redis")
    async def test_acquire_lock_with_blocking(self, mock_redis, mock_sleep):
        """Test lock acquisition with blocking."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client

        mock_client.set.side_effect = [False, False, True]

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        result = await manager.acquire_lock("test_lock", blocking_timeout=1.0)

        assert result is True
        assert mock_client.set.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_release_lock_success(self, mock_redis):
        """Test successful lock release."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.get.return_value = b"lock_id_123"
        mock_client.eval.return_value = 1

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        result = await manager.release_lock("test_lock", "lock_id_123")

        assert result is True
        mock_client.eval.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_release_lock_not_owner(self, mock_redis):
        """Test lock release when not the owner."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.get.return_value = b"different_lock_id"
        mock_client.eval.return_value = 0

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        result = await manager.release_lock("test_lock", "lock_id_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_lock_unhealthy_redis(self):
        """Test locking when Redis is unhealthy."""
        manager = RedisManager()
        manager._is_healthy = False

        assert await manager.acquire_lock("test_lock") is False
        assert await manager.release_lock("test_lock", "lock_id") is False

    @pytest.mark.asyncio
    async def test_lock_metrics_tracking(self):
        """Test that lock operations track metrics."""
        with patch("app.services.cache.cache_metrics.cache_metrics.timer") as mock_timer:
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__ = MagicMock(return_value=mock_timer_context)
            mock_timer_context.__exit__ = MagicMock(return_value=None)
            mock_timer.return_value = mock_timer_context

            from app.services.cache.redis_manager import RedisManager

            with patch("app.services.cache.redis_manager.redis") as mock_redis:
                mock_client = MagicMock()
                mock_redis.from_url.return_value = mock_client
                mock_client.set.return_value = True
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)

                manager = RedisManager()
                await manager.initialize()
                manager._client = mock_client
                manager._is_healthy = True

                await manager.acquire_lock("test_lock")

                mock_timer.assert_called_with("acquire_lock", "redis")


class TestRedisManagerBatchOperations:
    """Test Redis manager batch operations."""

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_batch_get(self, mock_redis):
        """Test batch get operation."""
        mock_client = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=["value1", None, "value3"])

        mock_redis.ConnectionPool.return_value = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping = AsyncMock()
        mock_client.pipeline.return_value = mock_pipeline

        manager = RedisManager()
        await manager.initialize()
        manager._is_healthy = True

        keys = ["key1", "key2", "key3"]
        results = await manager.batch_get(keys)

        assert results == {"key1": "value1", "key2": None, "key3": "value3"}
        mock_client.pipeline.assert_called_once()
        assert mock_pipeline.get.call_count == 3

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_batch_set_with_pipeline(self, mock_redis):
        """Test batch set using pipeline."""
        mock_client = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, True, True])

        mock_redis.from_url.return_value = mock_client
        mock_client.pipeline.return_value = mock_pipeline

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        await manager.batch_set(data, ttl=3600)

        mock_client.pipeline.assert_called_once()
        assert mock_pipeline.setex.call_count == 3
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cache.redis_manager.redis")
    async def test_batch_delete(self, mock_redis):
        """Test batch delete operation."""
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.delete.return_value = 2

        manager = RedisManager()
        await manager.initialize()
        manager._client = mock_client
        manager._is_healthy = True

        keys = ["key1", "key2", "key3"]
        result = await manager.batch_delete(keys)

        assert result == 2
        mock_client.delete.assert_called_once_with(*keys)
