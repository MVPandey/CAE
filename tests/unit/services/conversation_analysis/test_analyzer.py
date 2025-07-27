"""Unit tests for ConversationAnalyzer."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.schema.conversation_analysis import ConversationBranch
from app.schema.llm.message import Message
from app.services.conversation_analysis.analyzer import ConversationAnalyzer
from app.services.mcts.node import MCTSNode


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    return Mock()


@pytest.fixture
def analyzer(mock_llm_service):
    """Create ConversationAnalyzer instance."""
    return ConversationAnalyzer(mock_llm_service)


@pytest.fixture
def sample_nodes():
    """Create sample MCTS nodes for testing."""
    nodes = []
    for i in range(3):
        node = MCTSNode(f"Response {i}", index=i)
        node.avg_score = 0.9 - (i * 0.1)
        node.visits = 20 - (i * 5)
        node.simulated_reactions = [f"User reaction {i}"]
        node.sub_history = [
            {"role": "user", "content": f"User message {i}"},
            {"role": "assistant", "content": f"Assistant response {i}"},
        ]
        node.general_metrics = {"clarity": 0.9 - (i * 0.05), "relevance": 0.85 - (i * 0.05), "engagement": 0.8}
        node.goal_metrics = {"success": 0.85 - (i * 0.1)}

        for j in range(2):
            child = MCTSNode(f"Child response {i}-{j}", index=j)
            node.add_child(child)

        nodes.append(node)

    return nodes


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message(role="user", content="I need help with Python"),
        Message(role="assistant", content="I'd be happy to help you with Python"),
    ]


class TestConversationAnalyzer:
    """Test cases for ConversationAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_best_path_success(self, analyzer, sample_nodes, sample_messages, mock_llm_service):
        """Test successful best path analysis."""
        goal = "Help user learn Python"
        max_tokens = 100

        mock_response = Mock(content="This response is optimal because it clearly addresses the user's needs.")
        mock_llm_service.query_llm = AsyncMock(return_value=mock_response)

        best_node, best_idx, analysis = await analyzer.analyze_best_path(
            sample_nodes, sample_messages, goal, max_tokens
        )

        assert best_node == sample_nodes[0]
        assert best_idx == 0
        assert analysis == mock_response.content

        mock_llm_service.query_llm.assert_called_once()
        call_args = mock_llm_service.query_llm.call_args
        assert call_args.kwargs["json_response"] is False
        assert call_args.kwargs["max_tokens"] == max_tokens * 2

    @pytest.mark.asyncio
    async def test_analyze_best_path_llm_error(self, analyzer, sample_nodes, sample_messages, mock_llm_service):
        """Test best path analysis when LLM fails."""
        goal = "Help user"
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(side_effect=Exception("LLM error"))

        with patch("app.services.conversation_analysis.analyzer.logger") as mock_logger:
            best_node, best_idx, analysis = await analyzer.analyze_best_path(
                sample_nodes, sample_messages, goal, max_tokens
            )

            assert best_node == sample_nodes[0]
            assert best_idx == 0
            assert "Selected response 1" in analysis
            assert "0.90" in analysis

            mock_logger.error.assert_called_once()

    def test_convert_to_branches(self, analyzer, sample_nodes):
        """Test conversion of MCTS nodes to conversation branches."""
        branches = analyzer.convert_to_branches(sample_nodes)

        assert len(branches) == 3

        for i, branch in enumerate(branches):
            assert isinstance(branch, ConversationBranch)
            assert branch.response == f"Response {i}"
            assert branch.simulated_user_reactions == [f"User reaction {i}"]
            assert branch.score == sample_nodes[i].avg_score
            assert branch.sub_history == sample_nodes[i].sub_history
            assert branch.general_metrics == sample_nodes[i].general_metrics
            assert branch.goal_metrics == sample_nodes[i].goal_metrics
            assert branch.visits == sample_nodes[i].visits
            assert branch.parent_index is None
            assert branch.children_indices == [0, 1]

    def test_select_best_node_by_score_and_visits(self, analyzer, sample_nodes):
        """Test best node selection considering both score and visits."""
        sample_nodes[0].avg_score = 0.8
        sample_nodes[0].visits = 10

        sample_nodes[1].avg_score = 0.75
        sample_nodes[1].visits = 25

        sample_nodes[2].avg_score = 0.7
        sample_nodes[2].visits = 5

        best_node = analyzer._select_best_node(sample_nodes)

        assert best_node == sample_nodes[1]

    def test_select_best_node_zero_visits(self, analyzer):
        """Test best node selection when all nodes have zero visits."""
        nodes = [MCTSNode("Response 1"), MCTSNode("Response 2"), MCTSNode("Response 3")]

        for i, node in enumerate(nodes):
            node.avg_score = 0.9 - (i * 0.1)
            node.visits = 0

        best_node = analyzer._select_best_node(nodes)

        assert best_node == nodes[0]

    def test_build_analysis_prompt(self, analyzer, sample_nodes):
        """Test analysis prompt building."""
        goal = "Help user debug code"
        best_node = sample_nodes[0]

        prompt = analyzer._build_analysis_prompt(best_node, sample_nodes, goal)

        assert isinstance(prompt, Message)
        assert prompt.role == "system"
        assert goal in prompt.content
        assert best_node.response in prompt.content
        assert str(best_node.avg_score) in prompt.content
        assert str(best_node.visits) in prompt.content

        for node in sample_nodes:
            assert node.response[:100] in prompt.content

    def test_build_analysis_prompt_no_goal(self, analyzer, sample_nodes):
        """Test analysis prompt building without a goal."""
        best_node = sample_nodes[0]

        prompt = analyzer._build_analysis_prompt(best_node, sample_nodes, None)

        assert isinstance(prompt, Message)
        assert "<conversation_goal>" not in prompt.content

    def test_get_default_analysis(self, analyzer, sample_nodes):
        """Test default analysis generation."""
        best_node = sample_nodes[1]
        index = 1

        analysis = analyzer._get_default_analysis(best_node, index)

        assert "Selected response 2" in analysis
        assert "0.80" in analysis
        assert str(best_node.visits) in analysis

    def test_build_analysis_prompt_with_empty_metrics(self, analyzer):
        """Test analysis prompt building with nodes that have empty metrics."""
        nodes = [MCTSNode("Response 1")]
        nodes[0].avg_score = 0.5
        nodes[0].visits = 1
        nodes[0].general_metrics = {}

        prompt = analyzer._build_analysis_prompt(nodes[0], nodes, "Test goal")

        assert isinstance(prompt, Message)
        assert "key_strength" in prompt.content

    @pytest.mark.asyncio
    async def test_analyze_best_path_with_equal_scores(self, analyzer, mock_llm_service):
        """Test best path analysis when multiple nodes have equal scores."""
        nodes = []
        for i in range(3):
            node = MCTSNode(f"Response {i}", index=i)
            node.avg_score = 0.8
            node.visits = 10 + i
            node.general_metrics = {"clarity": 0.8}
            node.goal_metrics = {}
            nodes.append(node)

        mock_response = Mock(content="Analysis of best response")
        mock_llm_service.query_llm = AsyncMock(return_value=mock_response)

        best_node, best_idx, analysis = await analyzer.analyze_best_path(nodes, sample_messages, None, 100)

        assert best_node == nodes[2]
        assert best_idx == 2

    def test_convert_to_branches_with_no_children(self, analyzer):
        """Test conversion of nodes without children."""
        nodes = [MCTSNode("Response 1"), MCTSNode("Response 2")]
        for node in nodes:
            node.avg_score = 0.8
            node.visits = 10
            node.simulated_reactions = []
            node.sub_history = []
            node.general_metrics = {}
            node.goal_metrics = {}

        branches = analyzer.convert_to_branches(nodes)

        assert len(branches) == 2
        for branch in branches:
            assert branch.children_indices == []

    def test_select_best_node_single_node(self, analyzer):
        """Test best node selection with single node."""
        nodes = [MCTSNode("Only response")]
        nodes[0].avg_score = 0.5
        nodes[0].visits = 1

        best_node = analyzer._select_best_node(nodes)

        assert best_node == nodes[0]

    @pytest.mark.asyncio
    async def test_analyze_best_path_message_ordering(self, analyzer, sample_nodes, sample_messages, mock_llm_service):
        """Test that messages are passed to LLM in correct order."""
        mock_response = Mock(content="Analysis")
        mock_llm_service.query_llm = AsyncMock(return_value=mock_response)

        await analyzer.analyze_best_path(sample_nodes, sample_messages, "Goal", 100)

        call_args = mock_llm_service.query_llm.call_args
        messages = call_args.kwargs["messages"]

        assert messages[0].role == "system"
        assert messages[1:] == sample_messages
