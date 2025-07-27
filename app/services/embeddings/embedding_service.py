"""Production-ready embedding service with batching and caching."""

import hashlib
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...schema.llm.message import Message
from ...utils.config import app_settings
from ...utils.logger import logger
from ..cache.redis_manager import redis_manager


class EmbeddingService:
    """
    Handles text embeddings with production features:
    - Batch processing for efficiency
    - Caching of embeddings in Redis
    - Retry logic for API failures
    - Cost tracking and monitoring
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=app_settings.EMBEDDING_MODEL_API_KEY,
            base_url=app_settings.EMBEDDING_MODEL_BASE_URL,
        )
        self.model_name = app_settings.EMBEDDING_MODEL_NAME
        self.embedding_dimension = self._get_embedding_dimension()
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "total_tokens": 0,
        }

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model_name, 3072)

    def _hash_text(self, text: str) -> str:
        """Create stable hash for text caching."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _prepare_conversation_text(self, messages: list[Message]) -> str:
        """Convert messages to text for embedding."""
        text_parts = []
        for msg in messages:
            text_parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(text_parts)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embedding API with retry logic."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float",
            )

            embeddings = [item.embedding for item in response.data]

            self._stats["api_calls"] += 1
            self._stats["total_tokens"] += response.usage.total_tokens

            logger.info(
                "Embedding API call completed",
                extra={
                    "texts_count": len(texts),
                    "tokens_used": response.usage.total_tokens,
                    "model": self.model_name,
                },
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise

    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray | None:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use Redis cache

        Returns:
            Embedding vector or None if failed
        """
        self._stats["total_requests"] += 1

        if use_cache:
            cache_key = f"embedding:{self.model_name}:{self._hash_text(text)}"
            cached = await redis_manager.get_json(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                return np.array(cached["embedding"], dtype=np.float32)

        try:
            embeddings = await self._call_embedding_api([text])
            embedding = embeddings[0]

            if use_cache:
                await redis_manager.set_json(
                    cache_key,
                    {"embedding": embedding, "text_hash": self._hash_text(text)},
                    ttl=86400,  # 24 hours
                )

            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    async def embed_texts(
        self,
        texts: list[str],
        use_cache: bool = True,
        batch_size: int = 100,
    ) -> list[np.ndarray | None]:
        """
        Get embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use Redis cache
            batch_size: Maximum texts per API call

        Returns:
            List of embeddings (or None for failures)
        """
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if use_cache:
            for i, text in enumerate(texts):
                cache_key = f"embedding:{self.model_name}:{self._hash_text(text)}"
                cached = await redis_manager.get_json(cache_key)
                if cached:
                    results[i] = np.array(cached["embedding"], dtype=np.float32)
                    self._stats["cache_hits"] += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        for i in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[i : i + batch_size]
            batch_indices = uncached_indices[i : i + batch_size]

            try:
                embeddings = await self._call_embedding_api(batch_texts)

                for idx, text, embedding in zip(batch_indices, batch_texts, embeddings):
                    results[idx] = np.array(embedding, dtype=np.float32)

                    if use_cache:
                        cache_key = f"embedding:{self.model_name}:{self._hash_text(text)}"
                        await redis_manager.set_json(
                            cache_key,
                            {"embedding": embedding, "text_hash": self._hash_text(text)},
                            ttl=86400,  # 24 hours
                        )

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")

        self._stats["total_requests"] += len(texts)
        return results

    async def embed_conversation(
        self,
        messages: list[Message],
        use_cache: bool = True,
    ) -> np.ndarray | None:
        """
        Get embedding for a conversation.

        Args:
            messages: Conversation messages
            use_cache: Whether to use Redis cache

        Returns:
            Embedding vector or None if failed
        """
        text = self._prepare_conversation_text(messages)
        return await self.embed_text(text, use_cache)

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        cache_rate = (
            self._stats["cache_hits"] / self._stats["total_requests"] if self._stats["total_requests"] > 0 else 0
        )

        return {
            **self._stats,
            "cache_hit_rate": cache_rate,
            "model": self.model_name,
            "dimension": self.embedding_dimension,
        }

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear embedding cache."""
        try:
            count = 0
            async for key in redis_manager.scan_keys(f"embedding:{self.model_name}:{pattern}"):
                if await redis_manager.delete(key):
                    count += 1
            logger.info(f"Cleared {count} embedding cache entries")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0


embedding_service = EmbeddingService()
