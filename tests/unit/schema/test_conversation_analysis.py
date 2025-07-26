"""Unit tests for conversation_analysis schema models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schema.conversation_analysis import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
    ConversationBranch,
)


class TestConversationBranch:
    """Tests for ConversationBranch model."""

    def test_conversation_branch_creation(self):
        """Test creating a valid ConversationBranch."""
        branch = ConversationBranch(
            response="Hello, how can I help you?",
            simulated_user_reactions=["I need help with Python", "Just browsing"],
            score=0.85,
            sub_history=[
                {"user": "I need help with Python"},
                {"assistant": "I'd be happy to help with Python!"}
            ],
            general_metrics={"coherence": 0.9, "relevance": 0.8},
            goal_metrics={"helpfulness": 0.85, "engagement": 0.75},
        )

        assert branch.response == "Hello, how can I help you?"
        assert len(branch.simulated_user_reactions) == 2
        assert branch.score == 0.85
        assert branch.visits == 0  # Default value
        assert branch.parent_index is None  # Default value
        assert branch.children_indices == []  # Default value

    def test_conversation_branch_with_tree_structure(self):
        """Test ConversationBranch with tree structure fields."""
        branch = ConversationBranch(
            response="Test response",
            simulated_user_reactions=["Reaction 1"],
            score=0.5,
            sub_history=[],
            general_metrics={},
            goal_metrics={},
            visits=10,
            parent_index=0,
            children_indices=[2, 3, 4]
        )

        assert branch.visits == 10
        assert branch.parent_index == 0
        assert branch.children_indices == [2, 3, 4]

    def test_conversation_branch_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationBranch(
                response="Test response",
            )

        errors = exc_info.value.errors()
        field_names = {error['loc'][0] for error in errors}
        assert 'simulated_user_reactions' in field_names
        assert 'score' in field_names
        assert 'sub_history' in field_names
        assert 'general_metrics' in field_names
        assert 'goal_metrics' in field_names


class TestConversationAnalysisRequest:
    """Tests for ConversationAnalysisRequest model."""

    def test_request_creation_minimal(self):
        """Test creating a request with minimal required fields."""
        chat_id = uuid4()
        request = ConversationAnalysisRequest(chat_id=chat_id)

        assert request.chat_id == chat_id
        assert request.conversation_goal is None
        assert request.num_branches == 5
        assert request.simulation_depth == 3
        assert request.max_tokens == 250
        assert request.mcts_iterations == 10
        assert request.exploration_constant == 1.414

    def test_request_creation_full(self):
        """Test creating a request with all fields."""
        chat_id = uuid4()
        request = ConversationAnalysisRequest(
            chat_id=chat_id,
            conversation_goal="Help user feel better",
            num_branches=10,
            simulation_depth=5,
            max_tokens=500,
            mcts_iterations=20,
            exploration_constant=2.0
        )

        assert request.chat_id == chat_id
        assert request.conversation_goal == "Help user feel better"
        assert request.num_branches == 10
        assert request.simulation_depth == 5
        assert request.max_tokens == 500
        assert request.mcts_iterations == 20
        assert request.exploration_constant == 2.0

    def test_request_invalid_chat_id(self):
        """Test that invalid chat_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationAnalysisRequest(chat_id="not-a-uuid")

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'chat_id' for error in errors)


class TestConversationAnalysisResponse:
    """Tests for ConversationAnalysisResponse model."""

    def test_response_creation(self):
        """Test creating a valid ConversationAnalysisResponse."""
        analysis_id = uuid4()
        chat_id = uuid4()
        created_at = datetime.now(UTC)

        branches = [
            ConversationBranch(
                response=f"Response {i}",
                simulated_user_reactions=[f"Reaction {i}"],
                score=0.5 + i * 0.1,
                sub_history=[],
                general_metrics={"metric": i},
                goal_metrics={"goal": i},
            )
            for i in range(3)
        ]

        response = ConversationAnalysisResponse(
            id=analysis_id,
            chat_id=chat_id,
            created_at=created_at,
            conversation_goal="Test goal",
            branches=branches,
            selected_branch_index=1,
            selected_response="Response 1",
            analysis="Selected based on highest score",
            overall_scores={"average_score": 0.6, "best_score": 0.7},
            mcts_statistics={"iterations": 10, "time_elapsed": 1.5}
        )

        assert response.id == analysis_id
        assert response.chat_id == chat_id
        assert response.created_at == created_at
        assert response.conversation_goal == "Test goal"
        assert len(response.branches) == 3
        assert response.selected_branch_index == 1
        assert response.selected_response == "Response 1"
        assert response.analysis == "Selected based on highest score"
        assert response.overall_scores["average_score"] == 0.6
        assert response.mcts_statistics["iterations"] == 10

    def test_response_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationAnalysisResponse(
                id=uuid4(),
                chat_id=uuid4(),
            )

        errors = exc_info.value.errors()
        field_names = {error['loc'][0] for error in errors}
        assert 'created_at' in field_names
        assert 'branches' in field_names
        assert 'selected_branch_index' in field_names
        assert 'selected_response' in field_names
        assert 'analysis' in field_names
        assert 'overall_scores' in field_names
        assert 'mcts_statistics' in field_names

    def test_response_with_no_goal(self):
        """Test response with no conversation goal."""
        branches = [
            ConversationBranch(
                response="Response",
                simulated_user_reactions=["Reaction"],
                score=0.5,
                sub_history=[],
                general_metrics={},
                goal_metrics={},
            )
        ]

        response = ConversationAnalysisResponse(
            id=uuid4(),
            chat_id=uuid4(),
            created_at=datetime.now(UTC),
            conversation_goal=None,
            branches=branches,
            selected_branch_index=0,
            selected_response="Response",
            analysis="Analysis",
            overall_scores={},
            mcts_statistics={}
        )

        assert response.conversation_goal is None
