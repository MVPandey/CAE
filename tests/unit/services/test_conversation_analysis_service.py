"""Unit tests for ConversationAnalysisService."""
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from app.schema.conversation_analysis import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
    ConversationBranch,
)
from app.services.conversation_analysis_service import ConversationAnalysisService
from app.services.mcts.node import MCTSNode
from app.utils.exceptions import ChatHistoryNotFoundError


@pytest.fixture
def mock_dependencies():
    """Create mocked dependencies for ConversationAnalysisService."""
    with patch("app.services.conversation_analysis_service.LLMService") as mock_llm_service, \
         patch("app.services.conversation_analysis_service.ResponseGenerator") as mock_response_gen, \
         patch("app.services.conversation_analysis_service.ConversationSimulator") as mock_simulator, \
         patch("app.services.conversation_analysis_service.ConversationScorer") as mock_scorer, \
         patch("app.services.conversation_analysis_service.ConversationAnalyzer") as mock_analyzer, \
         patch("app.services.conversation_analysis_service.MCTSAlgorithm") as mock_mcts:

        yield {
            "llm_service": mock_llm_service,
            "response_generator": mock_response_gen,
            "simulator": mock_simulator,
            "scorer": mock_scorer,
            "analyzer": mock_analyzer,
            "mcts": mock_mcts
        }


@pytest.fixture
def service(mock_dependencies):
    """Create ConversationAnalysisService instance with mocked dependencies."""
    return ConversationAnalysisService()


@pytest.fixture
def sample_request():
    """Create a sample conversation analysis request."""
    return ConversationAnalysisRequest(
        chat_id=uuid4(),
        conversation_goal="Help the user solve a technical problem",
        num_branches=3,
        mcts_iterations=10,
        simulation_depth=5,
        exploration_constant=1.414,
        max_tokens=100
    )


@pytest.fixture
def sample_chat_history():
    """Create sample chat history."""
    return [
        Mock(role=Mock(value="user"), content="I'm having trouble with my Python code"),
        Mock(role=Mock(value="assistant"), content="I'd be happy to help. What's the issue?"),
        Mock(role=Mock(value="user"), content="It's throwing a KeyError")
    ]


@pytest.fixture
def sample_mcts_nodes():
    """Create sample MCTS nodes."""
    nodes = []
    for i in range(3):
        node = MCTSNode(f"Response {i}")
        node.avg_score = 0.8 - (i * 0.1)
        node.visits = 10 - i
        node.simulated_reactions = ["User seems satisfied", "User is engaged"]
        node.sub_history = [
            {"role": "user", "content": "That makes sense"},
            {"role": "assistant", "content": "Great! Let me explain further"}
        ]
        node.general_metrics = {"clarity": 0.85, "relevance": 0.9}
        node.goal_metrics = {"problem_solving": 0.8}
        nodes.append(node)
    return nodes


class TestConversationAnalysisService:
    """Test cases for ConversationAnalysisService."""

    @pytest.mark.asyncio
    async def test_analyze_conversation_success(
        self, service, sample_request, sample_chat_history, sample_mcts_nodes, mock_dependencies
    ):
        """Test successful conversation analysis."""
        mock_get_history = AsyncMock(return_value=sample_chat_history)
        mock_create_analysis = AsyncMock(return_value={
            "id": str(uuid4()),
            "chat_id": str(sample_request.chat_id),
            "created_at": "2024-01-01T00:00:00Z"
        })

        with patch("app.services.conversation_analysis_service.get_chat_history", mock_get_history), \
             patch("app.services.conversation_analysis_service.create_conversation_analysis", mock_create_analysis):

            service.response_generator.generate_initial_branches = AsyncMock(
                return_value=["Response 1", "Response 2", "Response 3"]
            )

            mcts_stats = {
                "total_iterations": sample_request.mcts_iterations,
                "nodes_created": sample_request.num_branches,
                "nodes_evaluated": 3 * 10,  # Assuming each node is evaluated in every iteration
                "pruned_branches": 3,  # Example: one branch pruned per node
                "parallel_evaluations": 3,  # Example: evaluations performed in parallel for each node
                "average_depth_explored": 10 / 3  # Example: average depth based on iterations and nodes
            }
            service.mcts.run = AsyncMock(return_value=(sample_mcts_nodes, mcts_stats))

            best_node = sample_mcts_nodes[0]
            service.analyzer.analyze_best_path = AsyncMock(
                return_value=(best_node, 0, "This response effectively addresses the issue")
            )

            branches = []
            for i, node in enumerate(sample_mcts_nodes):
                branch = Mock(spec=ConversationBranch)
                branch.response = node.response
                branch.simulated_user_reactions = node.simulated_reactions
                branch.score = node.avg_score
                branch.sub_history = node.sub_history
                branch.general_metrics = node.general_metrics
                branch.goal_metrics = node.goal_metrics
                branch.visits = node.visits
                branches.append(branch)

            service.analyzer.convert_to_branches = Mock(return_value=branches)

            result = await service.analyze_conversation(sample_request)

            assert isinstance(result, ConversationAnalysisResponse)
            assert result.chat_id == sample_request.chat_id
            assert result.conversation_goal == sample_request.conversation_goal
            assert result.selected_branch_index == 0
            assert result.selected_response == best_node.response
            assert len(result.branches) == 3
            assert result.mcts_statistics == mcts_stats

            mock_get_history.assert_called_once_with(sample_request.chat_id)
            service.response_generator.generate_initial_branches.assert_called_once()
            service.mcts.run.assert_called_once()
            service.analyzer.analyze_best_path.assert_called_once()
            service.analyzer.convert_to_branches.assert_called_once_with(sample_mcts_nodes)

    @pytest.mark.asyncio
    async def test_analyze_conversation_no_history(self, service, sample_request):
        """Test conversation analysis with no chat history."""
        mock_get_history = AsyncMock(return_value=None)

        with patch("app.services.conversation_analysis_service.get_chat_history", mock_get_history):
            with pytest.raises(ChatHistoryNotFoundError) as exc_info:
                await service.analyze_conversation(sample_request)

            assert str(sample_request.chat_id) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_conversation_empty_history(self, service, sample_request):
        """Test conversation analysis with empty chat history."""
        mock_get_history = AsyncMock(return_value=[])

        with patch("app.services.conversation_analysis_service.get_chat_history", mock_get_history):
            with pytest.raises(ChatHistoryNotFoundError):
                await service.analyze_conversation(sample_request)

    def test_calculate_variance(self, service):
        """Test variance calculation."""
        assert service._calculate_variance([]) == 0.0

        assert service._calculate_variance([5.0]) == 0.0

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        variance = service._calculate_variance(values)
        assert variance == 2.0  # Expected variance

        assert service._calculate_variance([3.0, 3.0, 3.0]) == 0.0

    def test_branch_to_dict(self, service):
        """Test branch to dictionary conversion."""
        branch = Mock(
            response="Test response",
            simulated_user_reactions=["Reaction 1", "Reaction 2"],
            score=0.85,
            sub_history=[{"role": "user", "content": "Test"}],
            general_metrics={"clarity": 0.9},
            goal_metrics={"success": 0.8},
            visits=10
        )

        result = service._branch_to_dict(branch)

        assert result["response"] == "Test response"
        assert result["simulated_user_reactions"] == ["Reaction 1", "Reaction 2"]
        assert result["score"] == 0.85
        assert result["sub_history"] == [{"role": "user", "content": "Test"}]
        assert result["general_metrics"] == {"clarity": 0.9}
        assert result["goal_metrics"] == {"success": 0.8}
        assert result["visits"] == 10

    @pytest.mark.asyncio
    async def test_analyze_conversation_with_logger(
        self, service, sample_request, sample_chat_history, sample_mcts_nodes
    ):
        """Test that appropriate logging occurs during conversation analysis."""
        mock_get_history = AsyncMock(return_value=sample_chat_history)
        mock_create_analysis = AsyncMock(return_value={
            "id": str(uuid4()),
            "chat_id": str(sample_request.chat_id),
            "created_at": "2024-01-01T00:00:00Z"
        })

        with patch("app.services.conversation_analysis_service.get_chat_history", mock_get_history), \
             patch("app.services.conversation_analysis_service.create_conversation_analysis", mock_create_analysis), \
             patch("app.services.conversation_analysis_service.logger") as mock_logger:

            service.response_generator.generate_initial_branches = AsyncMock(
                return_value=["Response 1", "Response 2", "Response 3"]
            )
            service.mcts.run = AsyncMock(return_value=(sample_mcts_nodes, {}))
            service.analyzer.analyze_best_path = AsyncMock(
                return_value=(sample_mcts_nodes[0], 0, "Analysis")
            )
            service.analyzer.convert_to_branches = Mock(return_value=[])

            await service.analyze_conversation(sample_request)

            assert mock_logger.info.call_count == 2
            start_log = mock_logger.info.call_args_list[0]
            assert "Starting conversation analysis" in start_log[0][0]

            end_log = mock_logger.info.call_args_list[1]
            assert "Analysis completed" in end_log[0][0]

    def test_service_initialization(self):
        """Test service initialization with correct dependencies."""
        with patch("app.services.conversation_analysis_service.app_settings") as mock_settings:
            mock_settings.LLM_API_BASE_URL = "http://test.com"
            mock_settings.LLM_API_KEY = "test-key"
            mock_settings.LLM_MODEL_NAME = "test-model"

            service = ConversationAnalysisService()

            assert service.llm_service is not None
            assert service.response_generator is not None
            assert service.simulator is not None
            assert service.scorer is not None
            assert service.analyzer is not None
            assert service.mcts is not None

    @pytest.mark.asyncio
    async def test_analyze_conversation_performance(
        self, service, sample_request, sample_chat_history, sample_mcts_nodes
    ):
        """Test that conversation analysis completes in reasonable time."""
        mock_get_history = AsyncMock(return_value=sample_chat_history)
        mock_create_analysis = AsyncMock(return_value={
            "id": str(uuid4()),
            "chat_id": str(sample_request.chat_id),
            "created_at": "2024-01-01T00:00:00Z"
        })

        with patch("app.services.conversation_analysis_service.get_chat_history", mock_get_history), \
             patch("app.services.conversation_analysis_service.create_conversation_analysis", mock_create_analysis):

            service.response_generator.generate_initial_branches = AsyncMock(
                return_value=["Response 1", "Response 2", "Response 3"]
            )
            service.mcts.run = AsyncMock(return_value=(sample_mcts_nodes, {}))
            service.analyzer.analyze_best_path = AsyncMock(
                return_value=(sample_mcts_nodes[0], 0, "Analysis")
            )
            service.analyzer.convert_to_branches = Mock(return_value=[])

            start_time = time.time()
            await service.analyze_conversation(sample_request)
            elapsed_time = time.time() - start_time

            assert elapsed_time < 1.0
