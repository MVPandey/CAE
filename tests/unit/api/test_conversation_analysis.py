"""Unit tests for the conversation analysis API endpoints - simplified version."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status

from app.main import app
from app.schema.conversation_analysis import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
    ConversationBranch,
)
from app.services.conversation_analysis_service import ConversationAnalysisService


@pytest.fixture
def mock_conversation_branches():
    """Create mock conversation branches."""
    return [
        ConversationBranch(
            response="I understand how you're feeling. Would you like to talk about what's bothering you?",
            simulated_user_reactions=["Yes, I'd like that", "Not really", "Maybe later"],
            score=0.85,
            sub_history=[
                {"role": "assistant", "content": "I understand how you're feeling..."},
                {"role": "user", "content": "Yes, I'd like that"},
                {"role": "assistant", "content": "I'm here to listen..."}
            ],
            general_metrics={"empathy": 0.9, "clarity": 0.8, "relevance": 0.85},
            goal_metrics={"emotional_support": 0.9, "comfort": 0.85},
            visits=15,
            parent_index=None,
            children_indices=[1, 2]
        ),
        ConversationBranch(
            response="That sounds challenging. Have you considered trying a different approach?",
            simulated_user_reactions=["What kind of approach?", "I've tried everything", "Tell me more"],
            score=0.75,
            sub_history=[
                {"role": "assistant", "content": "That sounds challenging..."},
                {"role": "user", "content": "What kind of approach?"},
                {"role": "assistant", "content": "Well, sometimes..."}
            ],
            general_metrics={"empathy": 0.7, "clarity": 0.8, "relevance": 0.75},
            goal_metrics={"emotional_support": 0.7, "comfort": 0.75},
            visits=10,
            parent_index=0,
            children_indices=[]
        )
    ]


@pytest.fixture
def mock_analysis_response(mock_conversation_branches):
    """Create a mock conversation analysis response."""
    chat_id = uuid4()
    return ConversationAnalysisResponse(
        id=uuid4(),
        chat_id=chat_id,
        created_at=datetime.utcnow(),
        conversation_goal="feel better",
        branches=mock_conversation_branches,
        selected_branch_index=0,
        selected_response=mock_conversation_branches[0].response,
        analysis="Branch 0 was selected due to its higher empathy score and better alignment with the emotional support goal.",
        overall_scores={
            "avg_score": 0.8,
            "best_score": 0.85,
            "worst_score": 0.75,
            "variance": 0.05
        },
        mcts_statistics={
            "total_iterations": 10,
            "total_simulations": 25,
            "avg_simulation_depth": 2.8,
            "time_taken_seconds": 5.2
        }
    )


class TestConversationAnalysisEndpoints:
    """Test cases for conversation analysis endpoints."""

    @pytest.mark.asyncio
    async def test_analyze_conversation_success(self, async_client, mock_analysis_response):
        """Test successful conversation analysis."""
        chat_id = uuid4()
        mock_analysis_response.chat_id = chat_id

        # Create a mock service
        mock_service = AsyncMock(spec=ConversationAnalysisService)
        mock_service.analyze_conversation = AsyncMock(return_value=mock_analysis_response)

        # Override the dependency
        app.dependency_overrides[ConversationAnalysisService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/analysis/",
                json={
                    "chat_id": str(chat_id),
                    "conversation_goal": "feel better",
                    "num_branches": 5,
                    "simulation_depth": 3
                }
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["chat_id"] == str(chat_id)
            assert data["conversation_goal"] == "feel better"
            assert len(data["branches"]) == 2
            assert data["selected_branch_index"] == 0
            assert "analysis" in data
            assert "overall_scores" in data
            assert "mcts_statistics" in data

            # Verify the service was called
            mock_service.analyze_conversation.assert_called_once()
            call_args = mock_service.analyze_conversation.call_args[0][0]
            assert isinstance(call_args, ConversationAnalysisRequest)
            assert call_args.chat_id == chat_id
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_analyze_conversation_minimal_params(self, async_client, mock_analysis_response):
        """Test conversation analysis with minimal parameters."""
        chat_id = uuid4()
        mock_analysis_response.chat_id = chat_id

        # Create a mock service
        mock_service = AsyncMock(spec=ConversationAnalysisService)
        mock_service.analyze_conversation = AsyncMock(return_value=mock_analysis_response)

        # Override the dependency
        app.dependency_overrides[ConversationAnalysisService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/analysis/",
                json={"chat_id": str(chat_id)}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["chat_id"] == str(chat_id)

            # Verify defaults were used
            call_args = mock_service.analyze_conversation.call_args[0][0]
            assert call_args.num_branches == 5
            assert call_args.simulation_depth == 3
            assert call_args.max_tokens == 250
            assert call_args.mcts_iterations == 10
            assert call_args.exploration_constant == 1.414
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_analyze_conversation_value_error(self, async_client):
        """Test conversation analysis when service raises ValueError."""
        # Create a mock service that raises ValueError
        mock_service = AsyncMock(spec=ConversationAnalysisService)
        mock_service.analyze_conversation = AsyncMock(side_effect=ValueError("Chat not found"))

        # Override the dependency
        app.dependency_overrides[ConversationAnalysisService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/analysis/",
                json={"chat_id": str(uuid4())}
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Chat not found" in response.json()["detail"]
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_analyze_conversation_timeout(self, async_client):
        """Test conversation analysis when operation times out."""
        # Create a mock service that raises TimeoutError
        mock_service = AsyncMock(spec=ConversationAnalysisService)
        mock_service.analyze_conversation = AsyncMock(side_effect=asyncio.TimeoutError())

        # Override the dependency
        app.dependency_overrides[ConversationAnalysisService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/analysis/",
                json={"chat_id": str(uuid4())}
            )

            assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
            assert "timed out" in response.json()["detail"]
            assert "reducing the simulation depth" in response.json()["detail"]
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_analyze_conversation_general_error(self, async_client):
        """Test conversation analysis when unexpected error occurs."""
        # Create a mock service that raises a general exception
        mock_service = AsyncMock(spec=ConversationAnalysisService)
        mock_service.analyze_conversation = AsyncMock(side_effect=Exception("Unexpected error"))

        # Override the dependency
        app.dependency_overrides[ConversationAnalysisService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/analysis/",
                json={"chat_id": str(uuid4())}
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Internal server error" in response.json()["detail"]
            assert "Unexpected error" in response.json()["detail"]
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_analyze_conversation_validation_errors(self, async_client):
        """Test request validation."""
        # Missing chat_id
        response = await async_client.post(
            "/analysis/",
            json={"conversation_goal": "feel better"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid chat_id format
        response = await async_client.post(
            "/analysis/",
            json={"chat_id": "not-a-uuid"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_analyses_success(self, async_client, mock_analysis_response):
        """Test successful retrieval of chat analyses."""
        chat_id = uuid4()
        analyses = [mock_analysis_response, mock_analysis_response]

        with patch("app.api.conversation_analysis.get_chat_analyses", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = analyses

            response = await async_client.get(f"/analysis/{chat_id}")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert all("id" in analysis for analysis in data)
            assert all("branches" in analysis for analysis in data)
            assert all("selected_branch_index" in analysis for analysis in data)
            mock_get.assert_called_once_with(chat_id)

    @pytest.mark.asyncio
    async def test_get_analyses_empty(self, async_client):
        """Test getting analyses when chat has none."""
        chat_id = uuid4()

        with patch("app.api.conversation_analysis.get_chat_analyses", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []

            response = await async_client.get(f"/analysis/{chat_id}")

            assert response.status_code == status.HTTP_200_OK
            assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_analyses_database_error(self, async_client):
        """Test getting analyses when database error occurs."""
        chat_id = uuid4()

        with patch("app.api.conversation_analysis.get_chat_analyses", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database connection lost")

            response = await async_client.get(f"/analysis/{chat_id}")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Database connection lost" in response.json()["detail"]
