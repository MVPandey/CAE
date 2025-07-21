from uuid import UUID

from app.db.chat import (
    create_chat_message,
    create_chat_session,
    get_chat_history,
)

from ..schema.llm.chat import ChatMessage, ChatRole
from ..schema.llm.message import Message as LLMMessage
from ..services.llm_service import LLMService


class ChatService:
    """
    Core service for handling chat logic, including interacting with the LLM
    and persisting the conversation.
    """

    def __init__(self):
        self.llm_service = LLMService()

    async def process_message(self, chat_id: UUID | None, user_id: UUID, user_message: str) -> list[ChatMessage]:
        """
        Processes a user message, gets a response from the LLM, and persists
        the conversation.

        Args:
            chat_id: The ID of the chat session, or None to create a new one.
            user_id: The ID of the user.
            user_message: The message from the user.

        Returns:
            The full, updated chat history for the session.
        """
        if not chat_id:
            chat = await create_chat_session(user_id)
            chat_id = chat.id

        user_chat_message = ChatMessage(chat_id=chat_id, role=ChatRole.USER, content=user_message)
        await create_chat_message(user_chat_message)

        history = await get_chat_history(chat_id)

        llm_messages = [LLMMessage(role=h.role.value, content=h.content) for h in history]

        # The LLM service handles the tool-calling loop internally. We get back
        # the final assistant message. This means we cannot save intermediate
        # tool requests and responses as separate messages.
        llm_response = await self.llm_service.query_llm(llm_messages, tools=list(self.llm_service.tools.keys()))

        tool_calls_for_db = None
        if llm_response.tool_calls:
            tool_calls_for_db = {"tool_calls": [tc.model_dump() for tc in llm_response.tool_calls]}

        assistant_message = ChatMessage(
            chat_id=chat_id,
            role=ChatRole.ASSISTANT,
            content=llm_response.content or "",
            tool_calls=tool_calls_for_db,
        )
        await create_chat_message(assistant_message)

        return await get_chat_history(chat_id)
