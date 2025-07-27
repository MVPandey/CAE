"""Unit tests for ConversationSimulator."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.schema.llm.message import Message
from app.services.conversation_analysis.config import ResponseConfig
from app.services.conversation_analysis.simulator import ConversationSimulator


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    return Mock()


@pytest.fixture
def simulator(mock_llm_service):
    """Create ConversationSimulator instance."""
    return ConversationSimulator(mock_llm_service)


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message(role="user", content="I'm struggling with Python async/await"),
        Message(role="assistant", content="I'd be happy to help you understand async/await"),
        Message(role="user", content="When should I use it?"),
    ]


class TestConversationSimulator:
    """Test cases for ConversationSimulator."""

    @pytest.mark.asyncio
    async def test_simulate_conversation_success(self, simulator, sample_messages, mock_llm_service):
        """Test successful conversation simulation."""
        depth = 3
        goal = "Help user understand async programming"
        max_tokens = 100

        expected_simulation = {
            "simulation": [
                {"role": "assistant", "content": "Async/await is useful for I/O-bound operations"},
                {"role": "user", "content": "Like database queries?"},
                {"role": "assistant", "content": "Exactly! Database queries are a perfect example"},
                {"role": "user", "content": "How do I handle errors?"},
                {"role": "assistant", "content": "Use try/except blocks with async functions"},
                {"role": "user", "content": "That makes sense, thanks!"},
            ],
            "user_reactions": [
                "User is learning and engaged",
                "User shows understanding",
                "User is satisfied with explanation",
            ],
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_simulation)

        result = await simulator.simulate_conversation(sample_messages, depth, goal, max_tokens)

        assert result == expected_simulation
        assert len(result["simulation"]) == 6
        assert len(result["user_reactions"]) == 3

        mock_llm_service.query_llm.assert_called_once()
        call_args = mock_llm_service.query_llm.call_args
        assert call_args.kwargs["json_response"] is True
        assert call_args.kwargs["max_tokens"] == max_tokens * ResponseConfig.TOKEN_MULTIPLIER_SIMULATION

    @pytest.mark.asyncio
    async def test_simulate_conversation_no_goal(self, simulator, sample_messages, mock_llm_service):
        """Test conversation simulation without a specific goal."""
        depth = 2
        max_tokens = 100

        expected_simulation = {
            "simulation": [
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "User response 1"},
                {"role": "assistant", "content": "Response 2"},
                {"role": "user", "content": "User response 2"},
            ],
            "user_reactions": ["Neutral", "Engaged"],
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_simulation)

        result = await simulator.simulate_conversation(sample_messages, depth, None, max_tokens)

        assert result == expected_simulation

        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        system_prompt = messages[0]
        assert "<conversation_goal>" not in system_prompt.content

    @pytest.mark.asyncio
    async def test_simulate_conversation_llm_exception(self, simulator, sample_messages, mock_llm_service):
        """Test handling of LLM exceptions during simulation."""
        depth = 3
        goal = "Test goal"
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(side_effect=Exception("LLM error"))

        with patch("app.services.conversation_analysis.simulator.logger") as mock_logger:
            result = await simulator.simulate_conversation(sample_messages, depth, goal, max_tokens)

            assert result == {"simulation": [], "user_reactions": []}

            mock_logger.error.assert_called_once_with("Failed to simulate conversation", exc_info=True)

    @pytest.mark.asyncio
    async def test_simulate_conversation_partial_response(self, simulator, sample_messages, mock_llm_service):
        """Test handling of partial response from LLM."""
        depth = 2
        max_tokens = 100

        partial_response = {
            "simulation": [{"role": "assistant", "content": "Response"}, {"role": "user", "content": "Reply"}]
        }

        mock_llm_service.query_llm = AsyncMock(return_value=partial_response)

        result = await simulator.simulate_conversation(sample_messages, depth, None, max_tokens)

        assert result["simulation"] == partial_response["simulation"]
        assert result["user_reactions"] == []

    @pytest.mark.asyncio
    async def test_simulate_conversation_missing_simulation_key(self, simulator, sample_messages, mock_llm_service):
        """Test handling when simulation key is missing."""
        depth = 2
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(return_value={"user_reactions": ["Reaction"]})

        result = await simulator.simulate_conversation(sample_messages, depth, None, max_tokens)

        assert result["simulation"] == []
        assert result["user_reactions"] == ["Reaction"]

    def test_build_simulation_prompt_with_goal(self, simulator):
        """Test building simulation prompt with a goal."""
        depth = 3
        goal = "Help user debug their code"

        prompt = simulator._build_simulation_prompt(depth, goal)

        assert isinstance(prompt, Message)
        assert prompt.role == "system"
        assert goal in prompt.content
        assert "<conversation_goal>" in prompt.content
        assert str(depth) in prompt.content
        assert "Return JSON:" in prompt.content

    def test_build_simulation_prompt_without_goal(self, simulator):
        """Test building simulation prompt without a goal."""
        depth = 5

        prompt = simulator._build_simulation_prompt(depth, None)

        assert isinstance(prompt, Message)
        assert "<conversation_goal>" not in prompt.content
        assert str(depth) in prompt.content

    @pytest.mark.asyncio
    async def test_simulate_conversation_depth_variations(self, simulator, sample_messages, mock_llm_service):
        """Test simulation with different depth values."""
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(
            return_value={
                "simulation": [
                    {"role": "assistant", "content": "Single response"},
                    {"role": "user", "content": "Single reply"},
                ],
                "user_reactions": ["One reaction"],
            }
        )

        result = await simulator.simulate_conversation(sample_messages, 1, None, max_tokens)
        assert len(result["simulation"]) == 2
        assert len(result["user_reactions"]) == 1

        prompt = mock_llm_service.query_llm.call_args.kwargs["messages"][0]
        assert "1" in prompt.content

    @pytest.mark.asyncio
    async def test_simulate_conversation_empty_messages(self, simulator, mock_llm_service):
        """Test simulation with empty message history."""
        empty_messages = []
        depth = 2
        max_tokens = 100

        expected_simulation = {
            "simulation": [
                {"role": "assistant", "content": "Starting conversation"},
                {"role": "user", "content": "Hello"},
            ],
            "user_reactions": ["User initiated"],
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_simulation)

        result = await simulator.simulate_conversation(empty_messages, depth, None, max_tokens)

        assert result == expected_simulation

    @pytest.mark.asyncio
    async def test_simulate_conversation_prompt_structure(self, simulator, sample_messages, mock_llm_service):
        """Test the structure of the simulation prompt."""
        depth = 3
        goal = "Test goal"
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(return_value={"simulation": [], "user_reactions": []})

        await simulator.simulate_conversation(sample_messages, depth, goal, max_tokens)

        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        assert len(messages) == len(sample_messages) + 1

        assert messages[0].role == "system"
        assert "Simulate realistic conversation continuation" in messages[0].content

        assert messages[1:] == sample_messages

    @pytest.mark.asyncio
    async def test_simulate_conversation_token_multiplier(self, simulator, sample_messages, mock_llm_service):
        """Test that token multiplier is applied correctly."""
        depth = 2
        max_tokens = 50

        mock_llm_service.query_llm = AsyncMock(return_value={"simulation": [], "user_reactions": []})

        await simulator.simulate_conversation(sample_messages, depth, None, max_tokens)

        expected_tokens = max_tokens * ResponseConfig.TOKEN_MULTIPLIER_SIMULATION
        actual_tokens = mock_llm_service.query_llm.call_args.kwargs["max_tokens"]
        assert actual_tokens == expected_tokens

    @pytest.mark.asyncio
    async def test_simulate_conversation_complex_goal(self, simulator, sample_messages, mock_llm_service):
        """Test simulation with complex multi-line goal."""
        depth = 2
        complex_goal = """Help the user understand:
        1. Basic async/await syntax
        2. When to use async programming
        3. Common pitfalls and how to avoid them"""
        max_tokens = 100

        expected_simulation = {
            "simulation": [
                {"role": "assistant", "content": "Let's start with basic syntax"},
                {"role": "user", "content": "Show me an example"},
            ],
            "user_reactions": ["User is ready to learn"],
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_simulation)

        result = await simulator.simulate_conversation(sample_messages, depth, complex_goal, max_tokens)

        assert result == expected_simulation

        prompt = mock_llm_service.query_llm.call_args.kwargs["messages"][0]
        assert complex_goal in prompt.content
