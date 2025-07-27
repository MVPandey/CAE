"""Tests for similarity search strategies."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.cache.similarity_strategies import (
    CosineSimilarityStrategy,
    DotProductSimilarityStrategy,
    EuclideanDistanceStrategy,
    HybridSimilarityStrategy,
    SimilarityStrategyFactory,
)


class TestCosineSimilarityStrategy:
    """Test cosine similarity strategy."""

    @pytest.mark.asyncio
    @patch("app.services.cache.similarity_strategies.embedding_service")
    async def test_compute_similarity(self, mock_embedding_service):
        """Test computing cosine similarity."""
        strategy = CosineSimilarityStrategy()

        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])

        mock_embedding_service.cosine_similarity.return_value = 0.0

        result = await strategy.compute_similarity(embedding1, embedding2)

        assert result == 0.0
        mock_embedding_service.cosine_similarity.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_similar_above_threshold(self):
        """Test finding similar embeddings above threshold."""
        strategy = CosineSimilarityStrategy()

        query_embedding = np.array([1.0, 0.0, 0.0])
        candidates = [
            ("key1", np.array([0.9, 0.1, 0.0])),
            ("key2", np.array([0.0, 1.0, 0.0])),
            ("key3", np.array([0.95, 0.05, 0.0])),
        ]

        async def compute_sim_side_effect(*args):
            if not hasattr(compute_sim_side_effect, "call_count"):
                compute_sim_side_effect.call_count = 0
            results = [0.9, 0.0, 0.95]
            result = results[compute_sim_side_effect.call_count]
            compute_sim_side_effect.call_count += 1
            return result

        strategy.compute_similarity = compute_sim_side_effect

        results = await strategy.find_similar(query_embedding, candidates, threshold=0.8, max_results=2)

        assert len(results) == 2
        assert results[0] == ("key3", 0.95)  # Highest similarity first
        assert results[1] == ("key1", 0.9)

    @pytest.mark.asyncio
    async def test_find_similar_none_above_threshold(self):
        """Test finding similar when none meet threshold."""
        strategy = CosineSimilarityStrategy()

        query_embedding = np.array([1.0, 0.0, 0.0])
        candidates = [
            ("key1", np.array([0.0, 1.0, 0.0])),
            ("key2", np.array([0.0, 0.0, 1.0])),
        ]

        async def compute_sim_zero(*args):
            return 0.0

        strategy.compute_similarity = compute_sim_zero

        results = await strategy.find_similar(query_embedding, candidates, threshold=0.5)

        assert len(results) == 0

    def test_preprocess_embedding_normalization(self):
        """Test embedding normalization."""
        strategy = CosineSimilarityStrategy()

        embedding = np.array([3.0, 4.0])  # Norm = 5
        normalized = strategy.preprocess_embedding(embedding)

        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(normalized, expected)

        assert np.isclose(np.linalg.norm(normalized), 1.0)

    def test_preprocess_embedding_zero_vector(self):
        """Test preprocessing zero vector."""
        strategy = CosineSimilarityStrategy()

        embedding = np.array([0.0, 0.0, 0.0])
        result = strategy.preprocess_embedding(embedding)

        np.testing.assert_array_equal(result, embedding)


class TestEuclideanDistanceStrategy:
    """Test Euclidean distance strategy."""

    @pytest.mark.asyncio
    async def test_compute_similarity(self):
        """Test computing similarity based on Euclidean distance."""
        strategy = EuclideanDistanceStrategy()

        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([1.0, 2.0, 3.0])

        result = await strategy.compute_similarity(embedding1, embedding2)
        assert np.isclose(result, 1.0)

        embedding3 = np.array([4.0, 5.0, 6.0])
        result2 = await strategy.compute_similarity(embedding1, embedding3)
        assert result2 < 1.0
        assert result2 > 0.0

    @pytest.mark.asyncio
    async def test_find_similar(self):
        """Test finding similar embeddings."""
        strategy = EuclideanDistanceStrategy()

        query_embedding = np.array([0.0, 0.0, 0.0])
        candidates = [
            ("key1", np.array([1.0, 0.0, 0.0])),  # Distance = 1
            ("key2", np.array([0.0, 0.5, 0.0])),  # Distance = 0.5
            ("key3", np.array([3.0, 4.0, 0.0])),  # Distance = 5
        ]

        results = await strategy.find_similar(query_embedding, candidates, threshold=0.1, max_results=3)

        assert len(results) == 2  # key3 should be below threshold
        assert results[0][0] == "key2"  # Closest
        assert results[1][0] == "key1"

    def test_preprocess_embedding(self):
        """Test that Euclidean strategy doesn't preprocess embeddings."""
        strategy = EuclideanDistanceStrategy()

        embedding = np.array([1.0, 2.0, 3.0])
        result = strategy.preprocess_embedding(embedding)

        np.testing.assert_array_equal(result, embedding)


class TestDotProductSimilarityStrategy:
    """Test dot product similarity strategy."""

    @pytest.mark.asyncio
    async def test_compute_similarity(self):
        """Test computing dot product similarity."""
        strategy = DotProductSimilarityStrategy()

        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])

        result = await strategy.compute_similarity(embedding1, embedding2)

        assert result == 32.0

    @pytest.mark.asyncio
    async def test_find_similar(self):
        """Test finding similar embeddings."""
        strategy = DotProductSimilarityStrategy()

        query_embedding = np.array([1.0, 0.0, 0.0])
        candidates = [
            ("key1", np.array([0.5, 0.5, 0.0])),  # Dot product = 0.5
            ("key2", np.array([1.0, 0.0, 0.0])),  # Dot product = 1.0
            ("key3", np.array([0.0, 1.0, 0.0])),  # Dot product = 0.0
        ]

        results = await strategy.find_similar(query_embedding, candidates, threshold=0.4, max_results=2)

        assert len(results) == 2
        assert results[0] == ("key2", 1.0)
        assert results[1] == ("key1", 0.5)

    def test_preprocess_embedding(self):
        """Test embedding normalization for dot product."""
        strategy = DotProductSimilarityStrategy()

        embedding = np.array([3.0, 4.0])  # Norm = 5
        normalized = strategy.preprocess_embedding(embedding)

        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(normalized, expected)


class TestHybridSimilarityStrategy:
    """Test hybrid similarity strategy."""

    @pytest.mark.asyncio
    async def test_compute_similarity_weighted_average(self):
        """Test computing weighted average of similarities."""
        cosine_strategy = MagicMock()
        euclidean_strategy = MagicMock()

        async def cosine_compute(*args):
            return 0.8

        async def euclidean_compute(*args):
            return 0.6

        cosine_strategy.compute_similarity = cosine_compute
        euclidean_strategy.compute_similarity = euclidean_compute

        strategy = HybridSimilarityStrategy(
            [
                (cosine_strategy, 0.7),
                (euclidean_strategy, 0.3),
            ]
        )

        embedding1 = np.array([1.0, 0.0])
        embedding2 = np.array([0.0, 1.0])

        result = await strategy.compute_similarity(embedding1, embedding2)

        assert np.isclose(result, 0.74)

    @pytest.mark.asyncio
    async def test_compute_similarity_weight_normalization(self):
        """Test that weights are normalized."""
        strategy1 = MagicMock()
        strategy2 = MagicMock()

        async def strategy1_compute(*args):
            return 1.0

        async def strategy2_compute(*args):
            return 0.5

        strategy1.compute_similarity = strategy1_compute
        strategy2.compute_similarity = strategy2_compute

        strategy = HybridSimilarityStrategy(
            [
                (strategy1, 2.0),
                (strategy2, 1.0),
            ]
        )

        result = await strategy.compute_similarity(np.array([1.0]), np.array([1.0]))

        assert np.isclose(result, 0.834, atol=0.001)

    @pytest.mark.asyncio
    async def test_find_similar(self):
        """Test finding similar with hybrid approach."""
        strategy = HybridSimilarityStrategy(
            [
                (CosineSimilarityStrategy(), 0.5),
                (DotProductSimilarityStrategy(), 0.5),
            ]
        )

        async def compute_sim_hybrid(*args):
            if not hasattr(compute_sim_hybrid, "call_count"):
                compute_sim_hybrid.call_count = 0
            results = [0.9, 0.3, 0.7]
            result = results[compute_sim_hybrid.call_count]
            compute_sim_hybrid.call_count += 1
            return result

        strategy.compute_similarity = compute_sim_hybrid

        query_embedding = np.array([1.0, 0.0])
        candidates = [
            ("key1", np.array([0.9, 0.1])),
            ("key2", np.array([0.0, 1.0])),
            ("key3", np.array([0.7, 0.7])),
        ]

        results = await strategy.find_similar(query_embedding, candidates, threshold=0.5)

        assert len(results) == 2
        assert results[0] == ("key1", 0.9)
        assert results[1] == ("key3", 0.7)

    def test_preprocess_embedding_uses_first_strategy(self):
        """Test that preprocessing uses first strategy."""
        strategy1 = MagicMock()
        strategy2 = MagicMock()

        strategy1.preprocess_embedding.return_value = np.array([0.6, 0.8])

        hybrid = HybridSimilarityStrategy(
            [
                (strategy1, 0.7),
                (strategy2, 0.3),
            ]
        )

        embedding = np.array([3.0, 4.0])
        result = hybrid.preprocess_embedding(embedding)

        np.testing.assert_array_equal(result, np.array([0.6, 0.8]))
        strategy1.preprocess_embedding.assert_called_once_with(embedding)
        strategy2.preprocess_embedding.assert_not_called()

    def test_preprocess_embedding_empty_strategies(self):
        """Test preprocessing with no strategies."""
        hybrid = HybridSimilarityStrategy([])

        embedding = np.array([1.0, 2.0])
        result = hybrid.preprocess_embedding(embedding)

        np.testing.assert_array_equal(result, embedding)


class TestSimilarityStrategyFactory:
    """Test similarity strategy factory."""

    def test_create_cosine_strategy(self):
        """Test creating cosine strategy."""
        strategy = SimilarityStrategyFactory.create("cosine")
        assert isinstance(strategy, CosineSimilarityStrategy)

    def test_create_euclidean_strategy(self):
        """Test creating Euclidean strategy."""
        strategy = SimilarityStrategyFactory.create("euclidean")
        assert isinstance(strategy, EuclideanDistanceStrategy)

    def test_create_dot_product_strategy(self):
        """Test creating dot product strategy."""
        strategy = SimilarityStrategyFactory.create("dot_product")
        assert isinstance(strategy, DotProductSimilarityStrategy)

    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown similarity strategy: unknown"):
            SimilarityStrategyFactory.create("unknown")

    def test_register_custom_strategy(self):
        """Test registering custom strategy."""

        class CustomStrategy(CosineSimilarityStrategy):
            pass

        SimilarityStrategyFactory.register("custom", CustomStrategy)

        strategy = SimilarityStrategyFactory.create("custom")
        assert isinstance(strategy, CustomStrategy)

    def test_create_hybrid_strategy(self):
        """Test creating hybrid strategy from config."""
        strategy_configs = [
            ("cosine", 0.6),
            ("euclidean", 0.4),
        ]

        hybrid = SimilarityStrategyFactory.create_hybrid(strategy_configs)

        assert isinstance(hybrid, HybridSimilarityStrategy)
        assert len(hybrid.strategies) == 2

        assert isinstance(hybrid.strategies[0][0], CosineSimilarityStrategy)
        assert isinstance(hybrid.strategies[1][0], EuclideanDistanceStrategy)
        assert hybrid.strategies[0][1] == 0.6
        assert hybrid.strategies[1][1] == 0.4
