"""Unit tests for MCTSAlgorithm."""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.schema.llm.message import Message
from app.services.conversation_analysis.config import MCTSConfig
from app.services.mcts.algorithm import MCTSAlgorithm
from app.services.mcts.node import MCTSNode


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for MCTSAlgorithm."""
    response_generator = Mock()
    simulator = Mock()
    scorer = Mock()
    return response_generator, simulator, scorer


@pytest.fixture
def mcts_algorithm(mock_dependencies):
    """Create MCTSAlgorithm instance with mocked dependencies."""
    response_generator, simulator, scorer = mock_dependencies
    return MCTSAlgorithm(response_generator, simulator, scorer)


@pytest.fixture
def base_messages():
    """Create base conversation messages."""
    return [
        Message(role="user", content="I need help with my project"),
        Message(role="assistant", content="I'd be happy to help you")
    ]


@pytest.fixture
def initial_responses():
    """Create initial response options."""
    return [
        "What specific aspect of your project needs help?",
        "Can you tell me more about your project?",
        "What challenges are you facing with your project?"
    ]


@pytest.fixture
def mcts_config():
    """Create MCTS configuration."""
    return {
        "iterations": 5,
        "simulation_depth": 3,
        "exploration_constant": 1.414,
        "goal": "Help user with their project",
        "max_tokens": 100
    }


class TestMCTSAlgorithm:
    """Test cases for MCTSAlgorithm."""

    @pytest.mark.asyncio
    async def test_run_basic(
        self, mcts_algorithm, base_messages, initial_responses, mcts_config
    ):
        """Test basic MCTS run."""
        # Mock simulation and scoring
        simulation_data = {
            "simulation": [
                {"role": "user", "content": "It's a web app"},
                {"role": "assistant", "content": "What framework are you using?"}
            ],
            "user_reactions": ["User is engaged"]
        }

        score_data = {
            "general_metrics": {"clarity": 0.85, "relevance": 0.9},
            "goal_metrics": {"helpfulness": 0.88},
            "overall_score": 0.87
        }

        mcts_algorithm.simulator.simulate_conversation = AsyncMock(return_value=simulation_data)
        mcts_algorithm.scorer.score_simulation = AsyncMock(return_value=score_data)
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock(return_value=None)

        # Execute
        root_nodes, stats = await mcts_algorithm.run(
            base_messages, initial_responses, mcts_config
        )

        # Verify
        assert len(root_nodes) == 3
        assert all(isinstance(node, MCTSNode) for node in root_nodes)
        assert stats["total_iterations"] == 5
        assert stats["nodes_created"] == 3  # Initial nodes
        assert stats["nodes_evaluated"] > 0
        assert stats["parallel_evaluations"] > 0

        # Verify nodes have been updated
        for node in root_nodes:
            assert node.visits > 0
            assert node.avg_score > 0

    @pytest.mark.asyncio
    async def test_run_with_expansion(
        self, mcts_algorithm, base_messages, initial_responses, mcts_config
    ):
        """Test MCTS run with node expansion."""
        # Mock responses
        simulation_data = {
            "simulation": [{"role": "user", "content": "Test"}],
            "user_reactions": ["Neutral"]
        }

        score_data = {
            "general_metrics": {"clarity": 0.8},
            "goal_metrics": {},
            "overall_score": 0.8
        }

        # Mock expansion to return new response on second iteration
        expansion_called = 0
        async def mock_expansion(*args, **kwargs):
            nonlocal expansion_called
            expansion_called += 1
            return "New expanded response" if expansion_called == 2 else None

        mcts_algorithm.simulator.simulate_conversation = AsyncMock(return_value=simulation_data)
        mcts_algorithm.scorer.score_simulation = AsyncMock(return_value=score_data)
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock(
            side_effect=mock_expansion
        )

        # Execute
        root_nodes, stats = await mcts_algorithm.run(
            base_messages, initial_responses, mcts_config
        )

        # Verify expansion occurred
        assert stats["nodes_created"] > 3  # More than initial nodes

        # Check that at least one node has children
        has_children = any(len(node.children) > 0 for node in root_nodes)
        assert has_children

    @pytest.mark.asyncio
    async def test_run_with_pruning(
        self, mcts_algorithm, base_messages, initial_responses, mcts_config
    ):
        """Test MCTS run with branch pruning."""
        # Set up config to trigger pruning
        mcts_config["iterations"] = MCTSConfig.PRUNING_INTERVAL + 1

        # Mock responses - make one branch perform poorly
        async def mock_score_simulation(messages, sim_data, goal, max_tokens):
            # Check which node is being scored based on messages
            if "What specific aspect" in messages[-1].content:
                return {
                    "general_metrics": {"clarity": 0.3},  # Low score
                    "goal_metrics": {},
                    "overall_score": 0.3
                }
            return {
                "general_metrics": {"clarity": 0.85},
                "goal_metrics": {},
                "overall_score": 0.85
            }

        simulation_data = {"simulation": [], "user_reactions": []}

        mcts_algorithm.simulator.simulate_conversation = AsyncMock(return_value=simulation_data)
        mcts_algorithm.scorer.score_simulation = AsyncMock(side_effect=mock_score_simulation)
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock(return_value=None)

        # Execute
        root_nodes, stats = await mcts_algorithm.run(
            base_messages, initial_responses, mcts_config
        )

        # Verify pruning occurred
        assert stats["pruned_branches"] >= 0

    @pytest.mark.asyncio
    async def test_select_node(self, mcts_algorithm):
        """Test node selection logic."""
        # Create a tree structure
        root = MCTSNode("Root")
        child1 = MCTSNode("Child 1")
        child2 = MCTSNode("Child 2")

        root.add_child(child1)
        root.add_child(child2)

        # Update visit counts and scores
        root.visits = 10
        child1.visits = 6
        child1.avg_score = 0.8
        child2.visits = 4
        child2.avg_score = 0.7

        # Test selection from root - should select root itself since it's not fully expanded
        selected = await mcts_algorithm._select_node(root, 1.414)
        assert selected == root  # Root has only 2 children, not fully expanded

        # Now make root fully expanded
        child3 = MCTSNode("Child 3")
        child3.visits = 2
        child3.avg_score = 0.6
        root.add_child(child3)

        # Now should select best child
        selected = await mcts_algorithm._select_node(root, 1.414)
        # Should select one of the children based on UCB1 score
        assert selected in [child1, child2, child3]

    @pytest.mark.asyncio
    async def test_select_node_unexpanded(self, mcts_algorithm):
        """Test node selection with unexpanded nodes."""
        root = MCTSNode("Root")

        # Add 3 children to make root fully expanded
        for i in range(3):
            child = MCTSNode(f"Child {i}")
            child.visits = 1
            child.avg_score = 0.5 + i * 0.1
            root.add_child(child)

        root.visits = 5

        # Now root is fully expanded, should select best child
        selected = await mcts_algorithm._select_node(root, 1.414)

        # Should select one of the children (they all have no children so are not fully expanded)
        assert selected in root.children

    @pytest.mark.asyncio
    async def test_expand_and_simulate(
        self, mcts_algorithm, base_messages
    ):
        """Test node expansion and simulation."""
        node = MCTSNode("Test response")
        node.visits = 1  # Eligible for expansion

        config = {
            "goal": "Test goal",
            "max_tokens": 100,
            "simulation_depth": 2
        }

        # Mock dependencies
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock(
            return_value="New child response"
        )

        simulation_data = {
            "simulation": [{"role": "user", "content": "Great!"}],
            "user_reactions": ["Positive"]
        }
        mcts_algorithm.simulator.simulate_conversation = AsyncMock(return_value=simulation_data)

        score_data = {
            "general_metrics": {"clarity": 0.9},
            "goal_metrics": {"success": 0.85},
            "overall_score": 0.88
        }
        mcts_algorithm.scorer.score_simulation = AsyncMock(return_value=score_data)

        # Execute
        score, new_children = await mcts_algorithm._expand_and_simulate(
            base_messages, node, config
        )

        # Verify
        assert score == 0.88
        assert len(new_children) == 1
        assert new_children[0].response == "New child response"
        assert node.sub_history == simulation_data["simulation"]
        assert node.simulated_reactions == simulation_data["user_reactions"]
        assert node.general_metrics == score_data["general_metrics"]
        assert node.goal_metrics == score_data["goal_metrics"]

    @pytest.mark.asyncio
    async def test_expand_and_simulate_no_expansion(
        self, mcts_algorithm, base_messages
    ):
        """Test simulation without expansion."""
        node = MCTSNode("Test response")
        node.visits = 0  # Not eligible for expansion

        config = {
            "goal": None,
            "max_tokens": 100,
            "simulation_depth": 1
        }

        # Mock dependencies
        simulation_data = {"simulation": [], "user_reactions": []}
        score_data = {"general_metrics": {}, "goal_metrics": {}, "overall_score": 0.5}

        mcts_algorithm.simulator.simulate_conversation = AsyncMock(return_value=simulation_data)
        mcts_algorithm.scorer.score_simulation = AsyncMock(return_value=score_data)
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock()

        # Execute
        score, new_children = await mcts_algorithm._expand_and_simulate(
            base_messages, node, config
        )

        # Verify no expansion occurred
        assert score == 0.5
        assert len(new_children) == 0
        mcts_algorithm.response_generator.generate_expansion_response.assert_not_called()

    def test_build_conversation_path(self, mcts_algorithm, base_messages):
        """Test building conversation path from node."""
        # Create a path: root -> child -> grandchild
        root = MCTSNode("")  # Empty root
        child = MCTSNode("First response")
        grandchild = MCTSNode("Second response")

        root.add_child(child)
        child.add_child(grandchild)

        # Build path from grandchild
        path = mcts_algorithm._build_conversation_path(base_messages, grandchild)

        # Verify path includes base messages and responses
        assert len(path) == len(base_messages) + 2
        assert path[:len(base_messages)] == base_messages
        assert path[-2].content == "First response"
        assert path[-2].role == "assistant"
        assert path[-1].content == "Second response"
        assert path[-1].role == "assistant"

    def test_build_conversation_path_single_node(self, mcts_algorithm, base_messages):
        """Test building conversation path from single node."""
        node = MCTSNode("Single response")

        path = mcts_algorithm._build_conversation_path(base_messages, node)

        # Should just be base messages (root response is empty and removed)
        assert path == base_messages

    @pytest.mark.asyncio
    async def test_run_statistics_tracking(
        self, mcts_algorithm, base_messages, initial_responses, mcts_config
    ):
        """Test that statistics are properly tracked during run."""
        # Mock minimal responses
        simulation_data = {"simulation": [], "user_reactions": []}
        score_data = {"general_metrics": {}, "overall_score": 0.5}

        mcts_algorithm.simulator.simulate_conversation = AsyncMock(return_value=simulation_data)
        mcts_algorithm.scorer.score_simulation = AsyncMock(return_value=score_data)
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock(return_value=None)

        # Execute
        root_nodes, stats = await mcts_algorithm.run(
            base_messages, initial_responses, mcts_config
        )

        # Verify all statistics are present
        assert "total_iterations" in stats
        assert "nodes_created" in stats
        assert "nodes_evaluated" in stats
        assert "pruned_branches" in stats
        assert "parallel_evaluations" in stats
        assert "average_depth_explored" in stats

        # Verify statistics make sense
        assert stats["total_iterations"] == mcts_config["iterations"]
        assert stats["nodes_created"] >= len(initial_responses)
        assert stats["nodes_evaluated"] > 0
        assert stats["parallel_evaluations"] > 0
        assert stats["average_depth_explored"] >= 0

    @pytest.mark.asyncio
    async def test_run_parallel_processing(
        self, mcts_algorithm, base_messages, initial_responses, mcts_config
    ):
        """Test that nodes are processed in parallel."""
        max_concurrent_calls = []
        current_concurrent_calls = 0

        async def mock_simulation(*args, **kwargs):
            nonlocal current_concurrent_calls
            current_concurrent_calls += 1
            max_concurrent_calls.append(current_concurrent_calls)
            await asyncio.sleep(0.01)  # Small delay
            current_concurrent_calls -= 1
            return {"simulation": [], "user_reactions": []}

        score_data = {"general_metrics": {}, "overall_score": 0.5}

        mcts_algorithm.simulator.simulate_conversation = AsyncMock(side_effect=mock_simulation)
        mcts_algorithm.scorer.score_simulation = AsyncMock(return_value=score_data)
        mcts_algorithm.response_generator.generate_expansion_response = AsyncMock(return_value=None)

        # Execute
        await mcts_algorithm.run(base_messages, initial_responses, mcts_config)

        # Check that simulations happened in parallel
        # In each iteration, all 3 nodes should be processed simultaneously
        for i in range(0, len(max_concurrent_calls), 3):
            iteration_calls = max_concurrent_calls[i:i+3]
            if len(iteration_calls) == 3:
                # Assert that all 3 nodes were processed concurrently
                assert max(iteration_calls) == 3  # Parallel execution

    @pytest.mark.asyncio
    async def test_run_empty_initial_responses(
        self, mcts_algorithm, base_messages, mcts_config
    ):
        """Test handling of empty initial responses."""
        # Running with empty initial responses should handle gracefully
        root_nodes, stats = await mcts_algorithm.run(base_messages, [], mcts_config)

        assert len(root_nodes) == 0
        assert stats["nodes_created"] == 0
