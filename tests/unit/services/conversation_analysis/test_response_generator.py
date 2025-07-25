"""Unit tests for ResponseGenerator."""
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.schema.llm.message import Message
from app.services.conversation_analysis.config import ResponseConfig
from app.services.conversation_analysis.response_generator import ResponseGenerator


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    return Mock()


@pytest.fixture
def response_generator(mock_llm_service):
    """Create ResponseGenerator instance."""
    return ResponseGenerator(mock_llm_service)


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message(role="user", content="Hello, I need help"),
        Message(role="assistant", content="I'm here to help you")
    ]


class TestResponseGenerator:
    """Test cases for ResponseGenerator."""

    @pytest.mark.asyncio
    async def test_generate_initial_branches_success(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test successful generation of initial response branches."""
        num_branches = 3
        goal = "Help user with their problem"
        max_tokens = 100

        # Mock LLM response
        expected_responses = [
            "I understand you need help. Can you provide more details?",
            "Let me assist you with that. What specific issue are you facing?",
            "I'm here to help. Could you elaborate on what you need?"
        ]
        mock_llm_service.query_llm = AsyncMock(return_value={"responses": expected_responses})

        # Execute
        responses = await response_generator.generate_initial_branches(
            sample_messages, num_branches, goal, max_tokens
        )

        # Verify
        assert responses == expected_responses
        assert len(responses) == num_branches

        # Verify LLM call
        mock_llm_service.query_llm.assert_called_once()
        call_args = mock_llm_service.query_llm.call_args
        assert call_args.kwargs["json_response"] is True
        assert call_args.kwargs["max_tokens"] == max_tokens * ResponseConfig.TOKEN_MULTIPLIER_INITIAL

    @pytest.mark.asyncio
    async def test_generate_initial_branches_no_goal(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test initial branches generation without a goal."""
        num_branches = 2
        max_tokens = 100

        expected_responses = ["Response 1", "Response 2"]
        mock_llm_service.query_llm = AsyncMock(return_value={"responses": expected_responses})

        responses = await response_generator.generate_initial_branches(
            sample_messages, num_branches, None, max_tokens
        )

        assert responses == expected_responses

        # Check that goal section is not included in prompt
        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        system_prompt = messages[0]
        assert "<conversation_goal>" not in system_prompt.content

    @pytest.mark.asyncio
    async def test_generate_initial_branches_invalid_response_format(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test handling of invalid response format from LLM."""
        num_branches = 3
        max_tokens = 100

        # Mock invalid response (not a dict)
        mock_llm_service.query_llm = AsyncMock(return_value="Invalid response")

        with patch("app.services.conversation_analysis.response_generator.logger") as mock_logger:
            responses = await response_generator.generate_initial_branches(
                sample_messages, num_branches, None, max_tokens
            )

            # Should return default responses
            assert responses == ResponseConfig.DEFAULT_RESPONSES[:num_branches]
            assert len(responses) == num_branches

            # Verify error was logged
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_initial_branches_missing_responses_key(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test handling when response dict is missing 'responses' key."""
        num_branches = 2
        max_tokens = 100

        # Mock response without 'responses' key
        mock_llm_service.query_llm = AsyncMock(return_value={"data": ["Response 1"]})

        with patch("app.services.conversation_analysis.response_generator.logger") as mock_logger:
            responses = await response_generator.generate_initial_branches(
                sample_messages, num_branches, None, max_tokens
            )

            assert responses == ResponseConfig.DEFAULT_RESPONSES[:num_branches]
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_initial_branches_llm_exception(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test handling of LLM exceptions."""
        num_branches = 3
        max_tokens = 100

        # Mock LLM to raise exception
        mock_llm_service.query_llm = AsyncMock(side_effect=Exception("LLM error"))

        with patch("app.services.conversation_analysis.response_generator.logger") as mock_logger:
            responses = await response_generator.generate_initial_branches(
                sample_messages, num_branches, "Goal", max_tokens
            )

            assert responses == ResponseConfig.DEFAULT_RESPONSES[:num_branches]
            mock_logger.error.assert_called_once_with(
                "Failed to generate initial branches", exc_info=True
            )

    @pytest.mark.asyncio
    async def test_generate_expansion_response_success(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test successful generation of expansion response."""
        existing_responses = ["Response 1", "Response 2"]
        goal = "Help user solve problem"
        max_tokens = 100

        expected_response = "Here's a different approach to help you"
        mock_llm_service.query_llm = AsyncMock(return_value={"response": expected_response})

        # Execute
        response = await response_generator.generate_expansion_response(
            sample_messages, existing_responses, goal, max_tokens
        )

        # Verify
        assert response == expected_response

        # Verify LLM call
        mock_llm_service.query_llm.assert_called_once()
        call_args = mock_llm_service.query_llm.call_args
        assert call_args.kwargs["json_response"] is True
        assert call_args.kwargs["max_tokens"] == max_tokens

    @pytest.mark.asyncio
    async def test_generate_expansion_response_no_goal(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test expansion response generation without a goal."""
        existing_responses = ["Response 1"]
        max_tokens = 100

        expected_response = "New response"
        mock_llm_service.query_llm = AsyncMock(return_value={"response": expected_response})

        response = await response_generator.generate_expansion_response(
            sample_messages, existing_responses, None, max_tokens
        )

        assert response == expected_response

        # Check that goal section is not included
        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        system_prompt = messages[0]
        assert "<goal>" not in system_prompt.content

    @pytest.mark.asyncio
    async def test_generate_expansion_response_invalid_format(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test handling of invalid response format for expansion."""
        existing_responses = ["Response 1"]
        max_tokens = 100

        # Mock invalid response
        mock_llm_service.query_llm = AsyncMock(return_value={"data": "Invalid"})

        with patch("app.services.conversation_analysis.response_generator.logger") as mock_logger:
            response = await response_generator.generate_expansion_response(
                sample_messages, existing_responses, None, max_tokens
            )

            assert response is None
            mock_logger.error.assert_called_once_with("Invalid expansion response format")

    @pytest.mark.asyncio
    async def test_generate_expansion_response_exception(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test handling of exceptions during expansion generation."""
        existing_responses = ["Response 1"]
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(side_effect=Exception("LLM error"))

        with patch("app.services.conversation_analysis.response_generator.logger") as mock_logger:
            response = await response_generator.generate_expansion_response(
                sample_messages, existing_responses, "Goal", max_tokens
            )

            assert response is None
            mock_logger.error.assert_called_once_with(
                "Failed to generate expansion response", exc_info=True
            )

    def test_build_initial_branches_prompt_with_goal(self, response_generator):
        """Test building initial branches prompt with a goal."""
        num_branches = 3
        goal = "Help user debug their code"

        prompt = response_generator._build_initial_branches_prompt(num_branches, goal)

        assert isinstance(prompt, Message)
        assert prompt.role == "system"
        assert str(num_branches) in prompt.content
        assert goal in prompt.content
        assert "<conversation_goal>" in prompt.content
        assert "Return JSON" in prompt.content

    def test_build_initial_branches_prompt_without_goal(self, response_generator):
        """Test building initial branches prompt without a goal."""
        num_branches = 2

        prompt = response_generator._build_initial_branches_prompt(num_branches, None)

        assert isinstance(prompt, Message)
        assert prompt.role == "system"
        assert str(num_branches) in prompt.content
        assert "<conversation_goal>" not in prompt.content

    def test_build_expansion_prompt_with_goal(self, response_generator):
        """Test building expansion prompt with a goal."""
        existing = ["Response 1", "Response 2"]
        goal = "Help achieve user's objective"

        prompt = response_generator._build_expansion_prompt(existing, goal)

        assert isinstance(prompt, Message)
        assert prompt.role == "system"
        assert goal in prompt.content
        assert "<goal>" in prompt.content
        assert json.dumps(existing) in prompt.content
        assert "Generate ONE new response" in prompt.content

    def test_build_expansion_prompt_without_goal(self, response_generator):
        """Test building expansion prompt without a goal."""
        existing = ["Response 1"]

        prompt = response_generator._build_expansion_prompt(existing, None)

        assert isinstance(prompt, Message)
        assert "<goal>" not in prompt.content
        assert json.dumps(existing) in prompt.content

    @pytest.mark.asyncio
    async def test_generate_initial_branches_prompt_format(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test that the prompt format is correct for initial branches."""
        num_branches = 3
        goal = "Test goal"
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(return_value={"responses": ["R1", "R2", "R3"]})

        await response_generator.generate_initial_branches(
            sample_messages, num_branches, goal, max_tokens
        )

        # Check the structure of messages sent to LLM
        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        assert len(messages) == len(sample_messages) + 1  # System prompt + conversation
        assert messages[0].role == "system"  # First is system prompt
        assert messages[1:] == sample_messages  # Rest is conversation history

    @pytest.mark.asyncio
    async def test_generate_expansion_response_prompt_format(
        self, response_generator, sample_messages, mock_llm_service
    ):
        """Test that the prompt format is correct for expansion."""
        existing = ["Existing 1", "Existing 2"]
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(return_value={"response": "New response"})

        await response_generator.generate_expansion_response(
            sample_messages, existing, None, max_tokens
        )

        # Check the structure of messages sent to LLM
        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        assert messages[0].role == "system"
        assert json.dumps(existing) in messages[0].content

    @pytest.mark.asyncio
    async def test_generate_initial_branches_with_empty_messages(
        self, response_generator, mock_llm_service
    ):
        """Test initial branches generation with empty message history."""
        empty_messages = []
        num_branches = 2
        max_tokens = 100

        expected_responses = ["Response 1", "Response 2"]
        mock_llm_service.query_llm = AsyncMock(return_value={"responses": expected_responses})

        responses = await response_generator.generate_initial_branches(
            empty_messages, num_branches, None, max_tokens
        )

        assert responses == expected_responses
