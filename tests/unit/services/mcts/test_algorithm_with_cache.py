"""Unit tests for MCTS algorithm with semantic cache integration."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.schema.llm.message import Message
from app.services.cache.semantic_cache import CacheEntry
from app.services.mcts.algorithm import MCTSAlgorithm
from app.services.mcts.node import MCTSNode


@pytest.fixture(autouse=True)
def mock_cache_modules():
    """Automatically mock cache modules for all tests to prevent real I/O."""
    with patch("app.services.mcts.algorithm.semantic_cache") as mock_semantic_cache:
        with patch("app.services.cache.semantic_cache.redis_manager") as mock_redis:
            with patch("app.services.cache.semantic_cache.embedding_service") as mock_embedding:
                mock_semantic_cache.get = AsyncMock(return_value=None)
                mock_semantic_cache.store = AsyncMock(return_value=True)

                mock_redis.get_json = AsyncMock(return_value=None)
                mock_redis.set_json = AsyncMock(return_value=True)
                mock_redis.exists = AsyncMock(return_value=False)
                mock_redis.scan_keys = AsyncMock(return_value=[])

                mock_embedding.get_embeddings = AsyncMock(return_value=[[0.1] * 1536])

                yield


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for MCTSAlgorithm."""
    response_generator = AsyncMock()
    simulator = AsyncMock()
    scorer = AsyncMock()

    response_generator.generate_expansion_response.return_value = "New response"
    simulator.simulate_conversation.return_value = {
        "simulation": [{"role": "user", "content": "Simulated user response"}],
        "user_reactions": ["Positive reaction"],
    }
    scorer.score_simulation.return_value = {
        "overall_score": 0.85,
        "general_metrics": {"clarity": 0.9, "relevance": 0.8},
        "goal_metrics": {"goal_achievement": 0.85},
    }

    return response_generator, simulator, scorer


@pytest.fixture
def mock_semantic_cache():
    """Create mock semantic cache."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)  # Default to cache miss
    cache.store = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def base_messages():
    """Create base conversation messages."""
    return [
        Message(role="user", content="I need help with my project"),
        Message(role="assistant", content="I'd be happy to help you"),
    ]


@pytest.fixture
def initial_responses():
    """Create initial response options."""
    return [
        "What specific aspect needs help?",
        "Can you tell me more about it?",
        "What challenges are you facing?",
    ]


@pytest.fixture
def mcts_config():
    """Create MCTS configuration."""
    return {
        "iterations": 1,  # Single iteration for fast tests
        "simulation_depth": 1,  # Minimal depth for fast tests
        "exploration_constant": 1.414,
        "goal": "Help the user effectively",
        "max_tokens": 250,
    }


@pytest.fixture
def sample_cache_entry():
    """Create sample cache entry."""
    return CacheEntry(
        key="cache_key",
        conversation_hash="hash123",
        embedding=None,
        response="Cached response",
        simulation_data={
            "simulation": [{"role": "user", "content": "Cached simulation"}],
            "user_reactions": ["Cached reaction"],
        },
        score_data={
            "overall_score": 0.9,
            "general_metrics": {"clarity": 0.95},
            "goal_metrics": {"goal_achievement": 0.9},
        },
        metadata={"similarity": 0.92},
        created_at=datetime.now(timezone.utc),
    )


class TestMCTSAlgorithmWithCache:
    """Test MCTS algorithm with cache integration."""

    async def test_mcts_with_cache_enabled(self, mock_dependencies, mock_semantic_cache):
        """Test MCTS algorithm with cache enabled."""
        response_generator, simulator, scorer = mock_dependencies

        with patch("app.services.mcts.algorithm.semantic_cache", mock_semantic_cache):
            mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=True)

            assert mcts.use_cache is True
            assert mcts._cache_stats["hits"] == 0
            assert mcts._cache_stats["misses"] == 0
            assert mcts._cache_stats["stores"] == 0

    async def test_mcts_with_cache_disabled(self, mock_dependencies):
        """Test MCTS algorithm with cache disabled."""
        response_generator, simulator, scorer = mock_dependencies

        mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=False)

        assert mcts.use_cache is False

    async def test_expand_and_simulate_cache_hit(
        self,
        mock_dependencies,
        mock_semantic_cache,
        base_messages,
        sample_cache_entry,
    ):
        """Test node expansion with cache hit."""
        response_generator, simulator, scorer = mock_dependencies
        mock_semantic_cache.get.return_value = sample_cache_entry

        with patch("app.services.mcts.algorithm.semantic_cache", mock_semantic_cache):
            mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=True)

            node = MCTSNode("Test response")
            config = {"max_tokens": 250, "goal": "Test goal"}

            score, new_children = await mcts._expand_and_simulate(base_messages, node, config)

        assert score == 0.9  # From cached data
        assert node.sub_history == sample_cache_entry.simulation_data["simulation"]
        assert node.simulated_reactions == sample_cache_entry.simulation_data["user_reactions"]
        assert node.general_metrics == sample_cache_entry.score_data["general_metrics"]
        assert mcts._cache_stats["hits"] == 1
        assert mcts._cache_stats["misses"] == 0

        response_generator.generate_expansion_response.assert_not_called()
        simulator.simulate_conversation.assert_not_called()
        scorer.score_simulation.assert_not_called()

    async def test_expand_and_simulate_cache_miss(
        self,
        mock_dependencies,
        mock_semantic_cache,
        base_messages,
    ):
        """Test node expansion with cache miss."""
        response_generator, simulator, scorer = mock_dependencies
        mock_semantic_cache.get.return_value = None  # Cache miss

        with patch("app.services.mcts.algorithm.semantic_cache", mock_semantic_cache):
            mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=True)

            node = MCTSNode("Test response")
            node.visits = 1  # Enable expansion
            config = {"max_tokens": 250, "goal": "Test goal", "simulation_depth": 3}

            score, new_children = await mcts._expand_and_simulate(base_messages, node, config)

        assert score == 0.85
        assert len(new_children) == 1
        assert mcts._cache_stats["hits"] == 0
        assert mcts._cache_stats["misses"] == 1
        assert mcts._cache_stats["stores"] == 1

        response_generator.generate_expansion_response.assert_called_once()
        simulator.simulate_conversation.assert_called_once()
        scorer.score_simulation.assert_called_once()

        mock_semantic_cache.store.assert_called_once()

    async def test_expand_and_simulate_no_cache(
        self,
        mock_dependencies,
        base_messages,
    ):
        """Test node expansion with cache disabled."""
        response_generator, simulator, scorer = mock_dependencies

        mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=False)

        node = MCTSNode("Test response")
        node.visits = 1  # Enable expansion
        config = {"max_tokens": 250, "simulation_depth": 3, "goal": "Test goal"}

        score, new_children = await mcts._expand_and_simulate(base_messages, node, config)

        assert score == 0.85
        assert len(new_children) == 1
        assert mcts._cache_stats["hits"] == 0
        assert mcts._cache_stats["misses"] == 0
        assert mcts._cache_stats["stores"] == 0

    async def test_run_with_cache_statistics(
        self,
        mock_dependencies,
        mock_semantic_cache,
        base_messages,
        initial_responses,
        mcts_config,
        sample_cache_entry,
    ):
        """Test full MCTS run with cache statistics."""
        response_generator, simulator, scorer = mock_dependencies

        mock_semantic_cache.get.side_effect = [
            sample_cache_entry,  # Hit
            None,  # Miss
            sample_cache_entry,  # Hit
        ]

        with patch("app.services.mcts.algorithm.semantic_cache", mock_semantic_cache):
            mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=True)

            root_nodes, stats = await mcts.run(base_messages, initial_responses, mcts_config)

        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == 2 / 3
        assert stats["nodes_evaluated"] > 0

    async def test_get_node_depth(self, mock_dependencies):
        """Test getting node depth."""
        response_generator, simulator, scorer = mock_dependencies
        mcts = MCTSAlgorithm(response_generator, simulator, scorer)

        root = MCTSNode("Root", parent=None)
        child = MCTSNode("Child", parent=root)
        grandchild = MCTSNode("Grandchild", parent=child)

        assert mcts._get_node_depth(root) == 0
        assert mcts._get_node_depth(child) == 1
        assert mcts._get_node_depth(grandchild) == 2

    async def test_build_conversation_path(self, mock_dependencies, base_messages):
        """Test building conversation path from node."""
        response_generator, simulator, scorer = mock_dependencies
        mcts = MCTSAlgorithm(response_generator, simulator, scorer)

        root = MCTSNode("", parent=None)  # Empty root
        child = MCTSNode("First response", parent=root)
        grandchild = MCTSNode("Second response", parent=child)

        path = mcts._build_conversation_path(base_messages, grandchild)

        assert len(path) == 4  # 2 base + 2 responses
        assert path[-2].content == "First response"
        assert path[-1].content == "Second response"

    def test_get_cache_stats(self, mock_dependencies):
        """Test getting cache statistics."""
        response_generator, simulator, scorer = mock_dependencies
        mcts = MCTSAlgorithm(response_generator, simulator, scorer)

        mcts._cache_stats = {
            "hits": 25,
            "misses": 75,
            "stores": 70,
        }

        stats = mcts.get_cache_stats()

        assert stats["hits"] == 25
        assert stats["misses"] == 75
        assert stats["stores"] == 70
        assert stats["total_lookups"] == 100
        assert stats["hit_rate"] == 0.25

    def test_get_cache_stats_no_lookups(self, mock_dependencies):
        """Test getting cache stats with no lookups."""
        response_generator, simulator, scorer = mock_dependencies
        mcts = MCTSAlgorithm(response_generator, simulator, scorer)

        stats = mcts.get_cache_stats()

        assert stats["total_lookups"] == 0
        assert stats["hit_rate"] == 0

    async def test_cache_store_only_for_non_root_nodes(
        self,
        mock_dependencies,
        mock_semantic_cache,
        base_messages,
    ):
        """Test that cache stores only happen for non-root nodes."""
        response_generator, simulator, scorer = mock_dependencies
        mock_semantic_cache.get.return_value = None  # Always miss

        with patch("app.services.mcts.algorithm.semantic_cache", mock_semantic_cache):
            mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=True)

            root_node = MCTSNode("")
            config = {"max_tokens": 250, "simulation_depth": 3}

            await mcts._expand_and_simulate(base_messages, root_node, config)

            mock_semantic_cache.store.assert_not_called()

            regular_node = MCTSNode("Regular response")
            await mcts._expand_and_simulate(base_messages, regular_node, config)

            mock_semantic_cache.store.assert_called_once()

    async def test_parallel_evaluations_with_cache(
        self,
        mock_dependencies,
        mock_semantic_cache,
        base_messages,
        initial_responses,
        mcts_config,
    ):
        """Test parallel node evaluations with cache."""
        response_generator, simulator, scorer = mock_dependencies

        mock_semantic_cache.get.return_value = None

        with patch("app.services.mcts.algorithm.semantic_cache", mock_semantic_cache):
            mcts = MCTSAlgorithm(response_generator, simulator, scorer, use_cache=True)

            original_gather = asyncio.gather
            gather_call_count = 0

            async def mock_gather(*tasks):
                nonlocal gather_call_count
                gather_call_count += 1
                return await original_gather(*tasks)

            with patch("asyncio.gather", side_effect=mock_gather):
                root_nodes, stats = await mcts.run(base_messages, initial_responses, mcts_config)

            assert gather_call_count > 0
            assert stats["parallel_evaluations"] > 0
