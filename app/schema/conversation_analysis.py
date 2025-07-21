from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ConversationBranch(BaseModel):
    """Represents a single conversation branch explored by MCTS"""

    response: str = Field(..., description="The potential response for this branch")
    simulated_user_reactions: list[str] = Field(..., description="Possible user reactions to this response")
    score: float = Field(..., description="Overall score for this branch based on goal optimization")
    sub_history: list[dict[str, str]] = Field(..., description="Simulated conversation continuation")
    general_metrics: dict[str, float] = Field(..., description="General conversation quality metrics")
    goal_metrics: dict[str, float] = Field(..., description="Goal-specific scoring metrics")
    visits: int = Field(default=0, description="Number of times this node was visited in MCTS")
    parent_index: int | None = Field(default=None, description="Index of parent branch in tree")
    children_indices: list[int] = Field(default_factory=list, description="Indices of child branches")


class ConversationAnalysisRequest(BaseModel):
    """Request model for conversation analysis"""

    chat_id: UUID = Field(..., description="The chat session to analyze")
    conversation_goal: str | None = Field(
        default=None,
        description="The goal of the conversation (e.g., 'feel better', 'get constructive criticism')",
    )
    num_branches: int = Field(default=5, description="Number of initial branches to explore")
    simulation_depth: int = Field(default=3, description="How many turns to simulate ahead")
    max_tokens: int = Field(default=250, description="Maximum tokens per LLM response")
    mcts_iterations: int = Field(default=10, description="Number of MCTS iterations to perform")
    exploration_constant: float = Field(default=1.414, description="UCB1 exploration constant (sqrt(2) by default)")


class ConversationAnalysisResponse(BaseModel):
    """Response model for conversation analysis"""

    id: UUID = Field(..., description="Analysis ID")
    chat_id: UUID = Field(..., description="Chat session ID")
    created_at: datetime = Field(..., description="When the analysis was created")
    conversation_goal: str | None = Field(..., description="The conversation goal used for optimization")
    branches: list[ConversationBranch] = Field(..., description="All explored branches")
    selected_branch_index: int = Field(..., description="Index of the selected branch")
    selected_response: str = Field(..., description="The chosen response")
    analysis: str = Field(..., description="Detailed analysis of why this path was chosen")
    overall_scores: dict[str, Any] = Field(..., description="Aggregated scoring metrics and statistics")
    mcts_statistics: dict[str, Any] = Field(..., description="MCTS algorithm performance statistics")
