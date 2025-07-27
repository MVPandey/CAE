"""Semantic cache for conversation analysis with similarity search."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ...schema.llm.message import Message
from ...utils.config import app_settings
from ...utils.logger import logger
from ..embeddings.embedding_service import embedding_service
from .redis_manager import redis_manager


@dataclass
class CacheEntry:
    """Represents a cached conversation analysis result."""

    key: str
    conversation_hash: str
    embedding: np.ndarray
    response: str
    simulation_data: dict[str, Any]
    score_data: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime | None = None


class SemanticCache:
    """
    Semantic cache for MCTS conversation analysis.

    Features:
    - Two-tier caching: exact match + similarity search
    - Configurable similarity thresholds
    - TTL-based expiration
    - Access pattern tracking
    - Cache warming strategies
    """

    def __init__(self):
        self.cache_prefix = "mcts_cache"
        self.index_prefix = "mcts_index"
        self.similarity_threshold = app_settings.CACHE_SIMILARITY_THRESHOLD
        self.ttl_seconds = app_settings.CACHE_TTL_SECONDS
        self._stats = {
            "exact_hits": 0,
            "similarity_hits": 0,
            "misses": 0,
            "stores": 0,
        }

    def _generate_conversation_hash(self, messages: list[Message]) -> str:
        """Generate stable hash for conversation."""
        text = ""
        for msg in messages:
            text += f"{msg.role}:{msg.content}\n"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _generate_cache_key(self, conversation_hash: str, suffix: str = "") -> str:
        """Generate Redis key for cache entry."""
        if suffix:
            return f"{self.cache_prefix}:{conversation_hash}:{suffix}"
        return f"{self.cache_prefix}:{conversation_hash}"

    async def _store_in_index(
        self,
        conversation_hash: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """Store entry in similarity index."""
        index_entry = {
            "hash": conversation_hash,
            "embedding": embedding.tolist(),
            "created_at": datetime.now(UTC).isoformat(),
            "message_count": metadata.get("message_count", 0),
        }

        await redis_manager.set_json(
            f"{self.index_prefix}:{conversation_hash}",
            index_entry,
            ttl=self.ttl_seconds,
        )

    async def _find_similar_entries(
        self,
        embedding: np.ndarray,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar conversations using embeddings."""
        similar_entries = []

        pattern = f"{self.index_prefix}:*"
        async for key in redis_manager.scan_keys(pattern, count=1000):
            if key == f"{self.index_prefix}:entries":
                continue

            entry = await redis_manager.get_json(key)
            if entry and "embedding" in entry:
                stored_embedding = np.array(entry["embedding"], dtype=np.float32)
                from ..embeddings.embedding_service import embedding_service

                similarity = embedding_service.cosine_similarity(embedding, stored_embedding)

                if similarity >= self.similarity_threshold:
                    similar_entries.append((entry["hash"], similarity))

        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries[:max_results]

    async def get(
        self,
        messages: list[Message],
        response_only: bool = False,
    ) -> CacheEntry | None:
        """
        Get cached result for conversation.

        Args:
            messages: Conversation messages
            response_only: If True, only return cached response (faster)

        Returns:
            CacheEntry if found, None otherwise
        """
        conversation_hash = self._generate_conversation_hash(messages)
        cache_key = self._generate_cache_key(conversation_hash)

        cached_data = await redis_manager.get_json(cache_key)
        if cached_data:
            self._stats["exact_hits"] += 1
            logger.info(
                "Cache exact hit",
                extra={"conversation_hash": conversation_hash},
            )

            await self._update_access_stats(conversation_hash)

            if response_only:
                return CacheEntry(
                    key=cache_key,
                    conversation_hash=conversation_hash,
                    embedding=np.array(cached_data["embedding"]),
                    response=cached_data["response"],
                    simulation_data={},
                    score_data={},
                    metadata=cached_data["metadata"],
                    created_at=datetime.fromisoformat(cached_data["created_at"]),
                )

            simulation_data = await redis_manager.get_json(self._generate_cache_key(conversation_hash, "simulation"))
            score_data = await redis_manager.get_json(self._generate_cache_key(conversation_hash, "score"))

            return CacheEntry(
                key=cache_key,
                conversation_hash=conversation_hash,
                embedding=np.array(cached_data["embedding"]),
                response=cached_data["response"],
                simulation_data=simulation_data or {},
                score_data=score_data or {},
                metadata=cached_data["metadata"],
                created_at=datetime.fromisoformat(cached_data["created_at"]),
                access_count=cached_data.get("access_count", 0),
            )

        embedding = await embedding_service.embed_conversation(messages)
        if embedding is None:
            self._stats["misses"] += 1
            return None

        similar_entries = await self._find_similar_entries(embedding)
        if similar_entries:
            best_match_hash, similarity = similar_entries[0]
            logger.info(
                "Cache similarity hit",
                extra={
                    "similarity": similarity,
                    "threshold": self.similarity_threshold,
                },
            )
            self._stats["similarity_hits"] += 1

            cache_key = self._generate_cache_key(best_match_hash)
            cached_data = await redis_manager.get_json(cache_key)
            if cached_data:
                await self._update_access_stats(best_match_hash)

                if response_only:
                    return CacheEntry(
                        key=cache_key,
                        conversation_hash=best_match_hash,
                        embedding=np.array(cached_data["embedding"]),
                        response=cached_data["response"],
                        simulation_data={},
                        score_data={},
                        metadata={**cached_data["metadata"], "similarity": similarity},
                        created_at=datetime.fromisoformat(cached_data["created_at"]),
                    )

                simulation_data = await redis_manager.get_json(self._generate_cache_key(best_match_hash, "simulation"))
                score_data = await redis_manager.get_json(self._generate_cache_key(best_match_hash, "score"))

                return CacheEntry(
                    key=cache_key,
                    conversation_hash=best_match_hash,
                    embedding=np.array(cached_data["embedding"]),
                    response=cached_data["response"],
                    simulation_data=simulation_data or {},
                    score_data=score_data or {},
                    metadata={**cached_data["metadata"], "similarity": similarity},
                    created_at=datetime.fromisoformat(cached_data["created_at"]),
                )

        self._stats["misses"] += 1
        return None

    async def store(
        self,
        messages: list[Message],
        response: str,
        simulation_data: dict[str, Any],
        score_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store analysis result in cache.

        Args:
            messages: Conversation messages
            response: Generated response
            simulation_data: Simulation results
            score_data: Scoring results
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        conversation_hash = self._generate_conversation_hash(messages)
        cache_key = self._generate_cache_key(conversation_hash)

        embedding = await embedding_service.embed_conversation(messages)
        if embedding is None:
            logger.error("Failed to generate embedding for cache storage")
            return False

        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "message_count": len(messages),
                "last_role": messages[-1].role if messages else None,
            }
        )

        main_data = {
            "conversation_hash": conversation_hash,
            "embedding": embedding.tolist(),
            "response": response,
            "metadata": metadata,
            "created_at": datetime.now(UTC).isoformat(),
            "access_count": 0,
        }

        success = await redis_manager.set_json(cache_key, main_data, ttl=self.ttl_seconds)

        if success:
            await redis_manager.set_json(
                self._generate_cache_key(conversation_hash, "simulation"),
                simulation_data,
                ttl=self.ttl_seconds,
            )
            await redis_manager.set_json(
                self._generate_cache_key(conversation_hash, "score"),
                score_data,
                ttl=self.ttl_seconds,
            )

            await self._store_in_index(conversation_hash, embedding, metadata)

            self._stats["stores"] += 1
            logger.info(
                "Stored in cache",
                extra={
                    "conversation_hash": conversation_hash,
                    "response_length": len(response),
                },
            )

        return success

    async def _update_access_stats(self, conversation_hash: str) -> None:
        """Update access statistics for cache entry."""
        cache_key = self._generate_cache_key(conversation_hash)
        cached_data = await redis_manager.get_json(cache_key)

        if cached_data:
            cached_data["access_count"] = cached_data.get("access_count", 0) + 1
            cached_data["last_accessed"] = datetime.now(UTC).isoformat()
            await redis_manager.set_json(cache_key, cached_data, ttl=self.ttl_seconds)

    async def invalidate(self, messages: list[Message]) -> bool:
        """Invalidate cache entry for conversation."""
        conversation_hash = self._generate_conversation_hash(messages)
        keys_to_delete = [
            self._generate_cache_key(conversation_hash),
            self._generate_cache_key(conversation_hash, "simulation"),
            self._generate_cache_key(conversation_hash, "score"),
            f"{self.index_prefix}:{conversation_hash}",
        ]

        deleted = 0
        for key in keys_to_delete:
            if await redis_manager.delete(key):
                deleted += 1

        logger.info(f"Invalidated {deleted} cache entries")
        return deleted > 0

    async def clear_all(self) -> int:
        """Clear all cache entries."""
        patterns = [
            f"{self.cache_prefix}:*",
            f"{self.index_prefix}:*",
        ]

        total_deleted = 0
        for pattern in patterns:
            async for key in redis_manager.scan_keys(pattern):
                if await redis_manager.delete(key):
                    total_deleted += 1

        logger.info(f"Cleared {total_deleted} cache entries")
        return total_deleted

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum(
            [
                self._stats["exact_hits"],
                self._stats["similarity_hits"],
                self._stats["misses"],
            ]
        )

        hit_rate = (
            (self._stats["exact_hits"] + self._stats["similarity_hits"]) / total_requests if total_requests > 0 else 0
        )

        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
        }

    async def warm_cache(
        self,
        conversation_patterns: list[list[Message]],
        generator_func: Any,
    ) -> int:
        """
        Warm cache with common conversation patterns.

        Args:
            conversation_patterns: List of common conversations
            generator_func: Function to generate analysis results

        Returns:
            Number of entries warmed
        """
        warmed = 0
        for messages in conversation_patterns:
            if await self.get(messages, response_only=True):
                continue

            try:
                result = await generator_func(messages)
                if result:
                    success = await self.store(
                        messages,
                        result["response"],
                        result["simulation_data"],
                        result["score_data"],
                        {"warmed": True},
                    )
                    if success:
                        warmed += 1
            except Exception as e:
                logger.error(f"Failed to warm cache entry: {e}")

        logger.info(f"Warmed {warmed} cache entries")
        return warmed

    async def health_check(self) -> bool:
        """Check if semantic cache is healthy."""
        try:
            test_key = "health_check_test"
            await redis_manager.set_json(test_key, {"test": True}, ttl=1)
            result = await redis_manager.get_json(test_key)
            await redis_manager.delete(test_key)
            return result is not None
        except Exception:
            return False


semantic_cache = SemanticCache()
