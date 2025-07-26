"""Unit tests for ConversationScorer."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.schema.llm.message import Message
from app.services.conversation_analysis.config import ScoringConfig
from app.services.conversation_analysis.scorer import ConversationScorer


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    return Mock()


@pytest.fixture
def scorer(mock_llm_service):
    """Create ConversationScorer instance."""
    return ConversationScorer(mock_llm_service)


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message(role="user", content="I need help with my code"),
        Message(role="assistant", content="I'd be happy to help"),
        Message(role="user", content="It's not working properly"),
    ]


@pytest.fixture
def sample_simulation_data():
    """Create sample simulation data."""
    return {
        "simulation": [
            {"role": "assistant", "content": "Can you show me the error?"},
            {"role": "user", "content": "Here's the error message..."},
        ],
        "user_reactions": ["User is engaged", "User seems satisfied"],
    }


class TestConversationScorer:
    """Test cases for ConversationScorer."""

    @pytest.mark.asyncio
    async def test_score_simulation_success(self, scorer, sample_messages, sample_simulation_data, mock_llm_service):
        """Test successful simulation scoring."""
        goal = "Help user debug code"
        max_tokens = 100

        expected_scores = {
            "general_metrics": {
                "clarity": 0.85,
                "relevance": 0.90,
                "engagement": 0.80,
                "authenticity": 0.85,
                "coherence": 0.88,
                "respectfulness": 0.95,
            },
            "goal_metrics": {"problem_solving": 0.82, "technical_accuracy": 0.88, "user_satisfaction": 0.85},
            "overall_score": 0.86,
            "reasoning": "The conversation effectively addresses the user's debugging needs",
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_scores)

        result = await scorer.score_simulation(sample_messages, sample_simulation_data, goal, max_tokens)

        assert result == expected_scores
        assert result["general_metrics"]["clarity"] == 0.85
        assert result["goal_metrics"]["problem_solving"] == 0.82
        assert result["overall_score"] == 0.86

        mock_llm_service.query_llm.assert_called_once()
        call_args = mock_llm_service.query_llm.call_args
        assert call_args.kwargs["json_response"] is True
        assert call_args.kwargs["max_tokens"] == max_tokens

    @pytest.mark.asyncio
    async def test_score_simulation_no_goal(self, scorer, sample_messages, sample_simulation_data, mock_llm_service):
        """Test simulation scoring without a specific goal."""
        max_tokens = 100

        expected_scores = {
            "general_metrics": {metric: 0.8 for metric in ScoringConfig.GENERAL_METRICS},
            "goal_metrics": {},
            "overall_score": 0.8,
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_scores)

        result = await scorer.score_simulation(sample_messages, sample_simulation_data, None, max_tokens)

        assert result == expected_scores
        assert result["goal_metrics"] == {}

        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        system_prompt = messages[0]
        assert "<goal_specific_scoring>" not in system_prompt.content

    @pytest.mark.asyncio
    async def test_score_simulation_llm_exception(
        self, scorer, sample_messages, sample_simulation_data, mock_llm_service
    ):
        """Test handling of LLM exceptions during scoring."""
        goal = "Test goal"
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(side_effect=Exception("LLM error"))

        with patch("app.services.conversation_analysis.scorer.logger") as mock_logger:
            result = await scorer.score_simulation(sample_messages, sample_simulation_data, goal, max_tokens)

            assert result == scorer._get_default_scores()
            assert all(score == 0.5 for score in result["general_metrics"].values())
            assert result["overall_score"] == 0.5

            mock_logger.error.assert_called_once_with("Failed to score simulation", exc_info=True)

    def test_build_scoring_prompt_with_goal(self, scorer):
        """Test building scoring prompt with a goal."""
        simulation_data = {"test": "data"}
        goal = "Help user learn Python"

        prompt = scorer._build_scoring_prompt(simulation_data, goal)

        assert isinstance(prompt, Message)
        assert prompt.role == "system"
        assert goal in prompt.content
        assert "<goal_specific_scoring>" in prompt.content
        assert json.dumps(simulation_data) in prompt.content
        assert "Return JSON:" in prompt.content

    def test_build_scoring_prompt_without_goal(self, scorer):
        """Test building scoring prompt without a goal."""
        simulation_data = {"test": "data"}

        prompt = scorer._build_scoring_prompt(simulation_data, None)

        assert isinstance(prompt, Message)
        assert "<goal_specific_scoring>" not in prompt.content
        assert json.dumps(simulation_data) in prompt.content

    def test_validate_scoring_result_complete(self, scorer):
        """Test validation of complete scoring result."""
        result = {
            "general_metrics": {
                "clarity": 0.9,
                "relevance": 0.85,
                "engagement": 0.8,
                "authenticity": 0.85,
                "coherence": 0.9,
                "respectfulness": 0.95,
            },
            "goal_metrics": {"success": 0.88},
            "overall_score": 0.87,
        }

        validated = scorer._validate_scoring_result(result)

        assert validated == result  # Should not modify complete result

    def test_validate_scoring_result_missing_general_metrics(self, scorer):
        """Test validation when general_metrics is missing."""
        result = {"goal_metrics": {"success": 0.8}, "overall_score": 0.8}

        validated = scorer._validate_scoring_result(result)

        assert "general_metrics" in validated
        assert all(metric in validated["general_metrics"] for metric in ScoringConfig.GENERAL_METRICS)
        assert all(score == 0.0 for score in validated["general_metrics"].values())

    def test_validate_scoring_result_partial_general_metrics(self, scorer):
        """Test validation when some general metrics are missing."""
        result = {"general_metrics": {"clarity": 0.9, "relevance": 0.85}, "goal_metrics": {}, "overall_score": 0.87}

        validated = scorer._validate_scoring_result(result)

        assert len(validated["general_metrics"]) == len(ScoringConfig.GENERAL_METRICS)
        assert validated["general_metrics"]["clarity"] == 0.9
        assert validated["general_metrics"]["relevance"] == 0.85
        assert validated["general_metrics"]["engagement"] == 0.0
        assert validated["general_metrics"]["authenticity"] == 0.0

    def test_validate_scoring_result_missing_goal_metrics(self, scorer):
        """Test validation when goal_metrics is missing."""
        result = {"general_metrics": {"clarity": 0.9}, "overall_score": 0.9}

        validated = scorer._validate_scoring_result(result)

        assert "goal_metrics" in validated
        assert validated["goal_metrics"] == {}

    def test_validate_scoring_result_missing_overall_score(self, scorer):
        """Test validation when overall_score is missing."""
        result = {"general_metrics": {"clarity": 0.8, "relevance": 0.9, "engagement": 0.7}, "goal_metrics": {}}

        validated = scorer._validate_scoring_result(result)

        assert "overall_score" in validated
        total = 0.8 + 0.9 + 0.7 + 0.0 + 0.0 + 0.0
        expected_score = total / len(ScoringConfig.GENERAL_METRICS)
        assert validated["overall_score"] == pytest.approx(expected_score, rel=1e-6)

    def test_validate_scoring_result_empty_general_metrics(self, scorer):
        """Test validation with empty general_metrics."""
        result = {"general_metrics": {}, "goal_metrics": {}}

        validated = scorer._validate_scoring_result(result)

        assert len(validated["general_metrics"]) == len(ScoringConfig.GENERAL_METRICS)
        assert all(score == 0.0 for score in validated["general_metrics"].values())
        assert validated["overall_score"] == 0.0

    def test_get_default_scores(self, scorer):
        """Test default scores generation."""
        default_scores = scorer._get_default_scores()

        assert "general_metrics" in default_scores
        assert "goal_metrics" in default_scores
        assert "overall_score" in default_scores

        assert len(default_scores["general_metrics"]) == len(ScoringConfig.GENERAL_METRICS)
        assert all(score == 0.5 for score in default_scores["general_metrics"].values())

        assert default_scores["goal_metrics"] == {}
        assert default_scores["overall_score"] == 0.5

    @pytest.mark.asyncio
    async def test_score_simulation_prompt_format(
        self, scorer, sample_messages, sample_simulation_data, mock_llm_service
    ):
        """Test that the scoring prompt is properly formatted."""
        goal = "Test goal"
        max_tokens = 100

        mock_llm_service.query_llm = AsyncMock(return_value={"general_metrics": {}, "overall_score": 0.5})

        await scorer.score_simulation(sample_messages, sample_simulation_data, goal, max_tokens)

        messages = mock_llm_service.query_llm.call_args.kwargs["messages"]
        assert len(messages) == len(sample_messages) + 1

        system_prompt = messages[0]
        assert system_prompt.role == "system"
        assert "Score this conversation" in system_prompt.content
        assert all(metric in system_prompt.content for metric in ScoringConfig.GENERAL_METRICS)

    @pytest.mark.asyncio
    async def test_score_simulation_complex_data(self, scorer, sample_messages, mock_llm_service):
        """Test scoring with complex simulation data."""
        complex_simulation_data = {
            "simulation": [
                {"role": "assistant", "content": "Let me help you step by step"},
                {"role": "user", "content": "That would be great"},
                {"role": "assistant", "content": "First, check your imports"},
                {"role": "user", "content": "I found the issue, thanks!"},
            ],
            "user_reactions": ["User is receptive", "User is engaged", "User is following along", "User is satisfied"],
            "metadata": {"duration": 120, "turns": 4},
        }

        expected_scores = {
            "general_metrics": {metric: 0.9 for metric in ScoringConfig.GENERAL_METRICS},
            "goal_metrics": {"success": 0.95},
            "overall_score": 0.91,
        }

        mock_llm_service.query_llm = AsyncMock(return_value=expected_scores)

        result = await scorer.score_simulation(sample_messages, complex_simulation_data, "Help debug", 100)

        assert result == expected_scores

        prompt = mock_llm_service.query_llm.call_args.kwargs["messages"][0]
        assert json.dumps(complex_simulation_data) in prompt.content
