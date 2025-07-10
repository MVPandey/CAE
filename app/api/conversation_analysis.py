from uuid import UUID
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from ..schema.conversation_analysis import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
)
from ..services.conversation_analysis_service import ConversationAnalysisService
from ..db.chat import get_chat_analyses
from ..utils.logger import logger


router = APIRouter(prefix="/analysis", tags=["Conversation Analysis"])


@router.post("/", response_model=ConversationAnalysisResponse)
async def analyze_conversation(
    request: ConversationAnalysisRequest,
    service: ConversationAnalysisService = Depends(ConversationAnalysisService),
):
    """
    Analyzes a conversation using MCTS to find the highest EQ response path.

    This endpoint:
    1. Generates 5 different response branches
    2. Simulates conversation continuations for each branch
    3. Scores each path based on emotional intelligence factors
    4. Returns all branches with analysis of why one was selected
    """
    logger.info(
        "Received conversation analysis request",
        extra={
            "chat_id": str(request.chat_id),
            "num_branches": request.num_branches,
            "simulation_depth": request.simulation_depth,
        },
    )

    try:
        result = await service.analyze_conversation(request)

        logger.info(
            "Conversation analysis completed successfully",
            extra={
                "chat_id": str(request.chat_id),
                "selected_branch": result.selected_branch_index,
                "best_score": result.branches[result.selected_branch_index].score,
                "analysis_id": str(result.id),
            },
        )

        return result

    except ValueError as e:
        logger.error(
            "Validation error in conversation analysis",
            extra={
                "chat_id": str(request.chat_id),
                "error": str(e),
            },
        )
        raise HTTPException(status_code=400, detail=str(e))

    except asyncio.TimeoutError:
        logger.error(
            "Timeout error in conversation analysis",
            extra={
                "chat_id": str(request.chat_id),
                "error": "Analysis timed out after 10 minutes",
            },
        )
        raise HTTPException(
            status_code=504,
            detail="Analysis timed out. The conversation may be too complex. Try reducing the simulation depth or number of branches.",
        )

    except Exception as e:
        logger.error(
            "Error analyzing conversation",
            extra={
                "chat_id": str(request.chat_id),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))


@router.get("/{chat_id}", response_model=list[ConversationAnalysisResponse])
async def get_conversation_analyses(chat_id: UUID):
    """
    Retrieves all analyses performed for a specific chat session.
    """
    logger.info(
        "Retrieving analyses for chat",
        extra={
            "chat_id": str(chat_id),
        },
    )

    try:
        analyses = await get_chat_analyses(chat_id)

        logger.info(
            "Retrieved chat analyses successfully",
            extra={
                "chat_id": str(chat_id),
                "analysis_count": len(analyses),
            },
        )

        return analyses

    except Exception as e:
        logger.error(
            "Error retrieving chat analyses",
            extra={
                "chat_id": str(chat_id),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
