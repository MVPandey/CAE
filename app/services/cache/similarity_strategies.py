"""Pluggable similarity search strategies for semantic cache."""

from abc import ABC, abstractmethod

import numpy as np

from ..embeddings.embedding_service import embedding_service


class SimilarityStrategy(ABC):
    """Abstract base class for similarity search strategies."""

    @abstractmethod
    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute similarity between two embeddings."""
        pass

    @abstractmethod
    async def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar embeddings from candidates."""
        pass

    @abstractmethod
    def preprocess_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Preprocess embedding before storage or comparison."""
        pass


class CosineSimilarityStrategy(SimilarityStrategy):
    """Cosine similarity search strategy."""

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        return embedding_service.cosine_similarity(embedding1, embedding2)

    async def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar embeddings using cosine similarity."""
        similarities = []

        for key, candidate_embedding in candidate_embeddings:
            similarity = await self.compute_similarity(query_embedding, candidate_embedding)
            if similarity >= threshold:
                similarities.append((key, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def preprocess_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding


class EuclideanDistanceStrategy(SimilarityStrategy):
    """Euclidean distance based similarity strategy."""

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute similarity based on Euclidean distance."""
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(np.exp(-distance))

    async def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar embeddings using Euclidean distance."""
        similarities = []

        for key, candidate_embedding in candidate_embeddings:
            similarity = await self.compute_similarity(query_embedding, candidate_embedding)
            if similarity >= threshold:
                similarities.append((key, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def preprocess_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """No preprocessing needed for Euclidean distance."""
        return embedding


class DotProductSimilarityStrategy(SimilarityStrategy):
    """Dot product similarity strategy (for normalized embeddings)."""

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute dot product similarity."""
        return float(np.dot(embedding1, embedding2))

    async def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar embeddings using dot product."""
        similarities = []

        for key, candidate_embedding in candidate_embeddings:
            similarity = await self.compute_similarity(query_embedding, candidate_embedding)
            if similarity >= threshold:
                similarities.append((key, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def preprocess_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for dot product similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding


class HybridSimilarityStrategy(SimilarityStrategy):
    """
    Hybrid strategy combining multiple similarity metrics.
    Useful for balancing different aspects of similarity.
    """

    def __init__(
        self,
        strategies: list[tuple[SimilarityStrategy, float]],
    ):
        """
        Initialize with weighted strategies.

        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
        total_weight = sum(weight for _, weight in strategies)
        self.strategies = [(strategy, weight / total_weight) for strategy, weight in strategies]

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute weighted average of similarities."""
        total_similarity = 0.0

        for strategy, weight in self.strategies:
            similarity = await strategy.compute_similarity(embedding1, embedding2)
            total_similarity += similarity * weight

        return total_similarity

    async def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar using hybrid approach."""
        similarities = []

        for key, candidate_embedding in candidate_embeddings:
            similarity = await self.compute_similarity(query_embedding, candidate_embedding)
            if similarity >= threshold:
                similarities.append((key, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def preprocess_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Use first strategy's preprocessing."""
        if self.strategies:
            return self.strategies[0][0].preprocess_embedding(embedding)
        return embedding


class SimilarityStrategyFactory:
    """Factory for creating similarity strategy instances."""

    _strategies = {
        "cosine": CosineSimilarityStrategy,
        "euclidean": EuclideanDistanceStrategy,
        "dot_product": DotProductSimilarityStrategy,
    }

    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> SimilarityStrategy:
        """Create a similarity strategy by name."""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown similarity strategy: {strategy_name}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)

    @classmethod
    def register(cls, name: str, strategy_class: type[SimilarityStrategy]) -> None:
        """Register a new similarity strategy."""
        cls._strategies[name] = strategy_class

    @classmethod
    def create_hybrid(
        cls,
        strategy_configs: list[tuple[str, float]],
    ) -> HybridSimilarityStrategy:
        """Create a hybrid strategy from configuration."""
        strategies = [(cls.create(name), weight) for name, weight in strategy_configs]
        return HybridSimilarityStrategy(strategies)
