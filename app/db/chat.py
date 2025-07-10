from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy import String, DateTime, ForeignKey, JSON, select
from uuid import UUID, uuid4
from datetime import datetime
import sqlalchemy as sa
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from ..schema.llm.chat import Chat, ChatMessage, ChatRole
from ..schema.user import User
from ..utils.config import app_settings

Base = declarative_base()


class UserModel(Base):
    __tablename__ = "user"
    id: Mapped[UUID] = mapped_column(sa.UUID, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chats: Mapped[list["ChatModel"]] = relationship(
        "ChatModel", back_populates="user", cascade="all, delete-orphan"
    )


class ChatModel(Base):
    __tablename__ = "chat"
    id: Mapped[UUID] = mapped_column(sa.UUID, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[UserModel] = relationship("UserModel", back_populates="chats")

    messages: Mapped[list["ChatMessageModel"]] = relationship(
        "ChatMessageModel", back_populates="chat", cascade="all, delete-orphan"
    )


class ChatMessageModel(Base):
    __tablename__ = "chat_message"
    id: Mapped[UUID] = mapped_column(sa.UUID, primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[ChatRole] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    tool_calls: Mapped[dict | None] = mapped_column(JSON)
    tool_call_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chat: Mapped[ChatModel] = relationship("ChatModel", back_populates="messages")


class ConversationAnalysisModel(Base):
    __tablename__ = "conversation_analysis"
    id: Mapped[UUID] = mapped_column(sa.UUID, primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation_goal: Mapped[str | None] = mapped_column(String, nullable=True)

    branches: Mapped[list[dict]] = mapped_column(JSON, nullable=False)

    selected_branch_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    selected_response: Mapped[str] = mapped_column(String, nullable=False)

    analysis: Mapped[str] = mapped_column(String, nullable=False)
    scores: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)

    mcts_statistics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    chat: Mapped[ChatModel] = relationship("ChatModel")


class Database:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(
            db_url, echo=False, pool_size=20, max_overflow=10
        )
        self.async_session_maker = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_db_and_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.async_session_maker() as session:
            yield session


DATABASE_URL = f"postgresql+asyncpg://{app_settings.DB_USER}:{app_settings.DB_SECRET}@{app_settings.DB_HOST}:{app_settings.DB_PORT}/{app_settings.DB_NAME}"
db = Database(DATABASE_URL)


async def create_user(name: str) -> User:
    """Creates a new user and returns the User object."""
    async with db.get_session() as session:
        new_user = UserModel(name=name)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return User.model_validate(new_user)


async def get_user(user_id: UUID) -> User | None:
    """Retrieves a user by their ID."""
    async with db.get_session() as session:
        result = await session.get(UserModel, user_id)
        return User.model_validate(result) if result else None


async def delete_user(user_id: UUID) -> bool:
    """
    Deletes a user and all their associated chats and messages.
    Returns True if user was deleted, False if user was not found.
    """
    async with db.get_session() as session:
        user = await session.get(UserModel, user_id)
        if not user:
            return False

        await session.delete(user)
        await session.commit()
        return True


async def list_users() -> list[User]:
    """Retrieves all users in the system."""
    async with db.get_session() as session:
        result = await session.execute(
            select(UserModel).order_by(UserModel.created_at.desc())
        )
        users = result.scalars().all()
        return [User.model_validate(u) for u in users]


async def get_user_chats(user_id: UUID) -> list[Chat]:
    """Retrieves all chat sessions for a specific user."""
    async with db.get_session() as session:
        result = await session.execute(
            select(ChatModel)
            .where(ChatModel.user_id == user_id)
            .order_by(ChatModel.created_at.desc())
        )
        chats = result.scalars().all()
        return [Chat.model_validate(c) for c in chats]


async def create_chat_session(user_id: UUID) -> Chat:
    """Creates a new chat session for a given user and returns the Chat object."""
    async with db.get_session() as session:
        new_chat = ChatModel(user_id=user_id)
        session.add(new_chat)
        await session.commit()
        await session.refresh(new_chat)
        return Chat.model_validate(new_chat)


async def get_chat_session(chat_id: UUID) -> Chat | None:
    """Retrieves a chat session by its ID."""
    async with db.get_session() as session:
        result = await session.get(ChatModel, chat_id)
        return Chat.model_validate(result) if result else None


async def create_chat_message(message: ChatMessage) -> ChatMessage:
    """Adds a new message to the database."""
    async with db.get_session() as session:
        new_message = ChatMessageModel(**message.model_dump())
        session.add(new_message)
        await session.commit()
        await session.refresh(new_message)
        return ChatMessage.model_validate(new_message)


async def get_chat_history(chat_id: UUID) -> list[ChatMessage]:
    """Retrieves all messages for a given chat session, ordered by created_at."""
    async with db.get_session() as session:
        result = await session.execute(
            select(ChatMessageModel)
            .where(ChatMessageModel.chat_id == chat_id)
            .order_by(ChatMessageModel.created_at)
        )
        messages = result.scalars().all()
        return [ChatMessage.model_validate(m) for m in messages]


async def delete_chat_session(chat_id: UUID) -> None:
    """Deletes a chat session and all associated messages from the database."""
    async with db.get_session() as session:
        await session.execute(
            sa.delete(ChatMessageModel).where(ChatMessageModel.chat_id == chat_id)
        )

        chat_to_delete = await session.get(ChatModel, chat_id)
        if chat_to_delete:
            await session.delete(chat_to_delete)

        await session.commit()


async def create_conversation_analysis(
    chat_id: UUID,
    conversation_goal: str | None,
    branches: list[dict],
    selected_branch_index: int,
    selected_response: str,
    analysis: str,
    scores: dict[str, float],
    mcts_statistics: dict[str, Any],
) -> dict:
    """Creates a new conversation analysis record."""
    async with db.get_session() as session:
        new_analysis = ConversationAnalysisModel(
            chat_id=chat_id,
            conversation_goal=conversation_goal,
            branches=branches,
            selected_branch_index=selected_branch_index,
            selected_response=selected_response,
            analysis=analysis,
            scores=scores,
            mcts_statistics=mcts_statistics,
        )
        session.add(new_analysis)
        await session.commit()
        await session.refresh(new_analysis)
        return {
            "id": new_analysis.id,
            "chat_id": new_analysis.chat_id,
            "created_at": new_analysis.created_at,
            "conversation_goal": new_analysis.conversation_goal,
            "branches": new_analysis.branches,
            "selected_branch_index": new_analysis.selected_branch_index,
            "selected_response": new_analysis.selected_response,
            "analysis": new_analysis.analysis,
            "scores": new_analysis.scores,
            "mcts_statistics": new_analysis.mcts_statistics,
        }


async def get_chat_analyses(chat_id: UUID) -> list[dict]:
    """Retrieves all analyses for a given chat session."""
    async with db.get_session() as session:
        result = await session.execute(
            select(ConversationAnalysisModel)
            .where(ConversationAnalysisModel.chat_id == chat_id)
            .order_by(ConversationAnalysisModel.created_at.desc())
        )
        analyses = result.scalars().all()
        return [
            {
                "id": a.id,
                "chat_id": a.chat_id,
                "created_at": a.created_at,
                "conversation_goal": a.conversation_goal,
                "branches": a.branches,
                "selected_branch_index": a.selected_branch_index,
                "selected_response": a.selected_response,
                "analysis": a.analysis,
                "scores": a.scores,
                "mcts_statistics": a.mcts_statistics,
            }
            for a in analyses
        ]
