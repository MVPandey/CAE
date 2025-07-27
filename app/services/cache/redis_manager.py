"""Production-ready Redis connection manager with pooling and circuit breaker."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...utils.config import app_settings
from ...utils.logger import logger
from .cache_metrics import cache_metrics, track_cache_operation


class RedisManager:
    """
    Manages Redis connections with production features:
    - Connection pooling with health checks
    - Automatic retry with exponential backoff
    - Circuit breaker pattern for failover
    - Graceful degradation when Redis is unavailable
    """

    def __init__(self):
        self._pool: ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._is_healthy = True
        self._health_check_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool(
                host=app_settings.REDIS_HOST,
                port=app_settings.REDIS_PORT,
                password=app_settings.REDIS_PASSWORD,
                db=app_settings.REDIS_DB,
                max_connections=app_settings.REDIS_MAX_CONNECTIONS,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            self._client = redis.Redis(connection_pool=self._pool)

            await self._client.ping()
            self._is_healthy = True

            self._health_check_task = asyncio.create_task(self._health_check_loop())

            cache_metrics.set_cache_info(
                redis_host=app_settings.REDIS_HOST,
                redis_port=str(app_settings.REDIS_PORT),
                max_connections=str(app_settings.REDIS_MAX_CONNECTIONS),
                pool_size=str(app_settings.REDIS_POOL_SIZE),
            )
            cache_metrics.update_connection_count(1)

            logger.info(
                "Redis initialized successfully",
                extra={
                    "host": app_settings.REDIS_HOST,
                    "port": app_settings.REDIS_PORT,
                    "pool_size": app_settings.REDIS_POOL_SIZE,
                },
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            cache_metrics.record_connection_error("initialization_failed")
            self._is_healthy = False

    async def close(self) -> None:
        """Close Redis connections and cleanup."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.close()

        if self._pool:
            await self._pool.disconnect()

        logger.info("Redis connections closed")

    async def _health_check_loop(self) -> None:
        """Periodic health check for Redis connection."""
        while True:
            try:
                await asyncio.sleep(30)
                await self._client.ping()
                if not self._is_healthy:
                    logger.info("Redis connection restored")
                    self._is_healthy = True
            except Exception:
                if self._is_healthy:
                    logger.error("Redis health check failed")
                    self._is_healthy = False

    @asynccontextmanager
    async def get_client(self):
        """Get Redis client with health check."""
        if not self._is_healthy:
            yield None
            return

        try:
            yield self._client
        except RedisError as e:
            logger.error(f"Redis operation failed: {e}")
            self._is_healthy = False
            yield None

    @retry(
        retry=retry_if_exception_type(RedisError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    @track_cache_operation("get")
    async def get(self, key: str) -> str | None:
        """Get value from Redis with retry logic."""
        if not self._is_healthy:
            return None

        try:
            async with self.get_client() as client:
                if client:
                    return await client.get(key)
            return None
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None

    @retry(
        retry=retry_if_exception_type(RedisError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    @track_cache_operation("set")
    async def set(
        self,
        key: str,
        value: str | dict,
        ttl: int | None = None,
    ) -> bool:
        """Set value in Redis with retry logic."""
        if not self._is_healthy:
            return False

        try:
            async with self.get_client() as client:
                if client:
                    if isinstance(value, dict):
                        value = json.dumps(value)

                    if ttl:
                        await client.setex(key, ttl, value)
                    else:
                        await client.set(key, value)
                    return True
            return False
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._is_healthy:
            return False

        try:
            async with self.get_client() as client:
                if client:
                    result = await client.delete(key)
                    return result > 0
            return False
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._is_healthy:
            return False

        try:
            async with self.get_client() as client:
                if client:
                    return await client.exists(key) > 0
            return False
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False

    async def scan_keys(self, pattern: str, count: int = 100):
        """Scan for keys matching pattern. Yields keys as they are found."""
        if not self._is_healthy:
            return

        try:
            async with self.get_client() as client:
                if client:
                    async for key in client.scan_iter(match=pattern, count=count):
                        yield key.decode("utf-8") if isinstance(key, bytes) else key
        except Exception as e:
            logger.error(f"Redis scan failed for pattern {pattern}: {e}")
            return

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Get JSON value from Redis."""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON for key {key}")
        return None

    async def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Set JSON value in Redis."""
        return await self.set(key, value, ttl)

    async def get_info(self) -> dict[str, Any]:
        """Get Redis server info for monitoring."""
        if not self._is_healthy:
            return {"status": "unhealthy"}

        try:
            async with self.get_client() as client:
                if client:
                    info = await client.info()
                    stats = {
                        "status": "healthy",
                        "version": info.get("redis_version"),
                        "used_memory": info.get("used_memory_human"),
                        "connected_clients": info.get("connected_clients"),
                        "total_connections_received": info.get("total_connections_received"),
                        "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
                    }
                    if info.get("connected_clients"):
                        cache_metrics.update_connection_count(info["connected_clients"])
                    return stats
            return {"status": "unhealthy"}
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"status": "unhealthy", "error": str(e)}

    @asynccontextmanager
    async def pipeline(self):
        """Get a Redis pipeline for batch operations."""
        if not self._is_healthy:
            yield None
            return

        try:
            async with self.get_client() as client:
                if client:
                    pipe = client.pipeline()
                    yield pipe
                else:
                    yield None
        except RedisError as e:
            logger.error(f"Redis pipeline failed: {e}")
            cache_metrics.record_connection_error("pipeline_failed")
            self._is_healthy = False
            yield None

    @track_cache_operation("batch_get")
    async def batch_get(self, keys: list[str]) -> dict[str, str | None]:
        """Get multiple keys in a single operation."""
        if not self._is_healthy or not keys:
            return {key: None for key in keys}

        try:
            async with self.pipeline() as pipe:
                if pipe:
                    for key in keys:
                        pipe.get(key)
                    results = await pipe.execute()
                    return dict(zip(keys, results))
            return {key: None for key in keys}
        except Exception as e:
            logger.error(f"Batch get failed: {e}")
            return {key: None for key in keys}

    @track_cache_operation("batch_set")
    async def batch_set(self, items: dict[str, str], ttl: int | None = None) -> dict[str, bool]:
        """Set multiple keys in a single operation."""
        if not self._is_healthy or not items:
            return {key: False for key in items}

        try:
            async with self.pipeline() as pipe:
                if pipe:
                    for key, value in items.items():
                        if ttl:
                            pipe.setex(key, ttl, value)
                        else:
                            pipe.set(key, value)
                    results = await pipe.execute()
                    return dict(zip(items.keys(), [bool(r) for r in results]))
            return {key: False for key in items}
        except Exception as e:
            logger.error(f"Batch set failed: {e}")
            return {key: False for key in items}

    async def batch_delete(self, keys: list[str]) -> int:
        """Delete multiple keys and return count of deleted keys."""
        if not self._is_healthy or not keys:
            return 0

        try:
            async with self.get_client() as client:
                if client:
                    return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error in batch delete: {e}")
            return 0

    async def exists_many(self, keys: list[str]) -> dict[str, bool]:
        """Check existence of multiple keys."""
        if not self._is_healthy or not keys:
            return {key: False for key in keys}

        try:
            async with self.get_client() as client:
                if client:
                    count = await client.exists(*keys)
                    result = {}
                    for i, key in enumerate(keys):
                        result[key] = i < count
                    return result
            return {key: False for key in keys}
        except Exception as e:
            logger.error(f"Error checking key existence: {e}")
            return {key: False for key in keys}

    def get_connection_info(self) -> dict[str, Any]:
        """Get Redis connection information."""
        info = {
            "is_healthy": self._is_healthy,
            "host": "unknown",
            "port": 0,
            "db": 0,
            "pool_size": 0,
        }

        if self._pool and hasattr(self._pool, "connection_kwargs"):
            conn_kwargs = self._pool.connection_kwargs
            info.update(
                {
                    "host": conn_kwargs.get("host", "unknown"),
                    "port": conn_kwargs.get("port", 0),
                    "db": conn_kwargs.get("db", 0),
                    "pool_size": getattr(self._pool, "max_connections", 0),
                }
            )

        return info

    @track_cache_operation("acquire_lock")
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking_timeout: float = 0.1,
    ) -> bool:
        """
        Acquire a distributed lock using Redis.

        Args:
            lock_name: Name of the lock
            timeout: Lock timeout in seconds
            blocking_timeout: Time to wait for lock acquisition

        Returns:
            True if lock acquired, False otherwise
        """
        if not self._is_healthy:
            return False

        lock_key = f"lock:{lock_name}"
        identifier = f"{id(self)}:{asyncio.get_event_loop().time()}"

        try:
            async with self.get_client() as client:
                if client:
                    acquired = await client.set(
                        lock_key,
                        identifier,
                        nx=True,
                        ex=timeout,
                    )
                    if acquired:
                        logger.debug(f"Acquired lock: {lock_name}")
                        return True

                    if blocking_timeout > 0:
                        max_attempts = 3
                        attempt = 0
                        while attempt < max_attempts - 1:
                            await asyncio.sleep(0.1)
                            acquired = await client.set(
                                lock_key,
                                identifier,
                                nx=True,
                                ex=timeout,
                            )
                            if acquired:
                                logger.debug(f"Acquired lock: {lock_name} after {attempt + 2} attempts")
                                return True
                            attempt += 1

            return False
        except Exception as e:
            logger.error(f"Failed to acquire lock {lock_name}: {e}")
            return False

    async def release_lock(self, lock_name: str, lock_id: str = None) -> bool:
        """Release a distributed lock with optional lock ID verification."""
        if not self._is_healthy:
            return False

        lock_key = f"lock:{lock_name}"

        try:
            async with self.get_client() as client:
                if client:
                    if lock_id:
                        lua_script = """
                        if redis.call("GET", KEYS[1]) == ARGV[1] then
                            return redis.call("DEL", KEYS[1])
                        else
                            return 0
                        end
                        """
                        deleted = await client.eval(lua_script, 1, lock_key, lock_id)
                    else:
                        deleted = await client.delete(lock_key)

                    if deleted:
                        logger.debug(f"Released lock: {lock_name}")
                    return bool(deleted)
            return False
        except Exception as e:
            logger.error(f"Failed to release lock {lock_name}: {e}")
            return False

    @asynccontextmanager
    async def distributed_lock(self, lock_name: str, timeout: int = 10):
        """Context manager for distributed locking."""
        acquired = await self.acquire_lock(lock_name, timeout)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release_lock(lock_name)


redis_manager = RedisManager()
