from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from ..db import chat as db
from ..schema.llm.chat import ChatMessage
from ..services.chat_service import ChatService
from ..utils.logger import logger


class ChatRequest(BaseModel):
    user_id: str
    message: str
    chat_id: str | None = None


router = APIRouter(prefix="/chats", tags=["Chat"])


@router.post("/", response_model=list[ChatMessage])
async def send_message(request: ChatRequest, service: ChatService = Depends(ChatService)):
    """
    Sends a message to a chat, creating a new session if no chat_id is provided.
    """
    logger.info(
        "Received chat message request",
        extra={
            "user_id": str(request.user_id),
            "chat_id": str(request.chat_id) if request.chat_id else None,
            "message_length": len(request.message),
            "message_preview": request.message[:100] + "..." if len(request.message) > 100 else request.message,
        },
    )

    try:
        result = await service.process_message(
            chat_id=request.chat_id,
            user_id=request.user_id,
            user_message=request.message,
        )
        logger.info(
            "Chat message processed successfully",
            extra={
                "user_id": str(request.user_id),
                "chat_id": str(request.chat_id) if request.chat_id else None,
                "response_messages": len(result),
            },
        )
        return result
    except Exception as e:
        logger.error(
            "Error processing chat message",
            extra={
                "user_id": str(request.user_id),
                "chat_id": str(request.chat_id) if request.chat_id else None,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chat_id}", response_model=list[ChatMessage])
async def get_chat_history(chat_id: str):
    """
    Retrieves the message history for a specific chat session.
    """
    history = await db.get_chat_history(chat_id)
    if not history:
        raise HTTPException(status_code=404, detail="Chat not found")
    return history


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str):
    """
    Deletes a chat session and its entire message history.
    """
    await db.delete_chat_session(chat_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
