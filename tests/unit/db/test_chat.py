"""Unit tests for app.db.chat module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.chat import (
    Base,
    ChatMessageModel,
    ChatModel,
    ConversationAnalysisModel,
    Database,
    UserModel,
    create_chat_message,
    create_chat_session,
    create_conversation_analysis,
    create_user,
    delete_chat_session,
    delete_user,
    get_chat_analyses,
    get_chat_history,
    get_chat_session,
    get_user,
    get_user_chats,
    list_users,
)
from app.schema.llm.chat import Chat, ChatMessage, ChatRole
from app.schema.user import User


class TestDatabaseModels:
    """Test SQLAlchemy model definitions."""

    def test_user_model_structure(self):
        """Test UserModel has correct structure."""
        assert UserModel.__tablename__ == "user"

        assert hasattr(UserModel, "id")
        assert hasattr(UserModel, "name")
        assert hasattr(UserModel, "created_at")
        assert hasattr(UserModel, "chats")

        assert UserModel.chats.property.back_populates == "user"

    def test_chat_model_structure(self):
        """Test ChatModel has correct structure."""
        assert ChatModel.__tablename__ == "chat"

        assert hasattr(ChatModel, "id")
        assert hasattr(ChatModel, "user_id")
        assert hasattr(ChatModel, "created_at")
        assert hasattr(ChatModel, "user")
        assert hasattr(ChatModel, "messages")

        assert ChatModel.user.property.back_populates == "chats"
        assert ChatModel.messages.property.back_populates == "chat"

    def test_chat_message_model_structure(self):
        """Test ChatMessageModel has correct structure."""
        assert ChatMessageModel.__tablename__ == "chat_message"

        assert hasattr(ChatMessageModel, "id")
        assert hasattr(ChatMessageModel, "chat_id")
        assert hasattr(ChatMessageModel, "role")
        assert hasattr(ChatMessageModel, "content")
        assert hasattr(ChatMessageModel, "tool_calls")
        assert hasattr(ChatMessageModel, "tool_call_id")
        assert hasattr(ChatMessageModel, "created_at")
        assert hasattr(ChatMessageModel, "chat")

    def test_conversation_analysis_model_structure(self):
        """Test ConversationAnalysisModel has correct structure."""
        assert ConversationAnalysisModel.__tablename__ == "conversation_analysis"

        assert hasattr(ConversationAnalysisModel, "id")
        assert hasattr(ConversationAnalysisModel, "chat_id")
        assert hasattr(ConversationAnalysisModel, "created_at")
        assert hasattr(ConversationAnalysisModel, "conversation_goal")
        assert hasattr(ConversationAnalysisModel, "branches")
        assert hasattr(ConversationAnalysisModel, "selected_branch_index")
        assert hasattr(ConversationAnalysisModel, "selected_response")
        assert hasattr(ConversationAnalysisModel, "analysis")
        assert hasattr(ConversationAnalysisModel, "scores")
        assert hasattr(ConversationAnalysisModel, "mcts_statistics")
        assert hasattr(ConversationAnalysisModel, "chat")


class TestDatabase:
    """Test Database class."""

    @patch("app.db.chat.create_async_engine")
    @patch("app.db.chat.sessionmaker")
    def test_database_initialization(self, mock_sessionmaker, mock_create_engine):
        """Test Database initialization."""
        db_url = "postgresql+asyncpg://test:test@localhost:5432/testdb"
        db = Database(db_url)

        mock_create_engine.assert_called_once_with(db_url, echo=False, pool_size=20, max_overflow=10)

        mock_sessionmaker.assert_called_once_with(
            mock_create_engine.return_value, class_=AsyncSession, expire_on_commit=False
        )

        assert db.engine == mock_create_engine.return_value
        assert db.async_session_maker == mock_sessionmaker.return_value

    @pytest.mark.asyncio
    @patch("app.db.chat.create_async_engine")
    async def test_create_db_and_tables(self, mock_create_engine):
        """Test create_db_and_tables method."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_begin = AsyncMock()
        mock_begin.__aenter__.return_value = mock_conn
        mock_begin.__aexit__.return_value = None
        mock_engine.begin = MagicMock(return_value=mock_begin)
        mock_create_engine.return_value = mock_engine

        db = Database("postgresql+asyncpg://test:test@localhost:5432/testdb")
        await db.create_db_and_tables()

        mock_conn.run_sync.assert_called_once()
        args = mock_conn.run_sync.call_args[0]
        assert args[0] == Base.metadata.create_all

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test get_session context manager."""
        db = Database("postgresql+asyncpg://test:test@localhost:5432/testdb")

        mock_session = AsyncMock()
        mock_session_maker = AsyncMock()
        mock_session_maker.__aenter__.return_value = mock_session
        mock_session_maker.__aexit__.return_value = None
        db.async_session_maker = MagicMock(return_value=mock_session_maker)

        async with db.get_session() as session:
            assert session == mock_session


class TestUserFunctions:
    """Test user-related database functions."""

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_create_user(self, mock_db):
        """Test create_user function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        user_id = uuid4()
        mock_user = Mock(spec=UserModel)
        mock_user.id = user_id
        mock_user.name = "Test User"
        mock_user.created_at = datetime.now(UTC)

        with patch("app.db.chat.UserModel") as MockUserModel:
            MockUserModel.return_value = mock_user
            result = await create_user("Test User")

        MockUserModel.assert_called_once_with(name="Test User")
        mock_session.add.assert_called_once_with(mock_user)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_user)

        assert isinstance(result, User)

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_get_user_found(self, mock_db):
        """Test get_user when user exists."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        user_id = uuid4()
        mock_user = Mock(spec=UserModel)
        mock_user.id = user_id
        mock_user.name = "Test User"
        mock_user.created_at = datetime.now(UTC)
        mock_session.get.return_value = mock_user

        result = await get_user(user_id)

        mock_session.get.assert_called_once_with(UserModel, user_id)
        assert isinstance(result, User)

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_get_user_not_found(self, mock_db):
        """Test get_user when user doesn't exist."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context
        mock_session.get.return_value = None

        result = await get_user(uuid4())

        assert result is None

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_delete_user_exists(self, mock_db):
        """Test delete_user when user exists."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        mock_user = Mock(spec=UserModel)
        mock_session.get.return_value = mock_user

        result = await delete_user(uuid4())

        mock_session.delete.assert_called_once_with(mock_user)
        mock_session.commit.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_delete_user_not_found(self, mock_db):
        """Test delete_user when user doesn't exist."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context
        mock_session.get.return_value = None

        result = await delete_user(uuid4())

        mock_session.delete.assert_not_called()
        assert result is False

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_list_users(self, mock_db):
        """Test list_users function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        mock_users = []
        for i in range(3):
            user = Mock(spec=UserModel)
            user.id = uuid4()
            user.name = f"User {i}"
            user.created_at = datetime.now(UTC)
            mock_users.append(user)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_users
        mock_session.execute.return_value = mock_result

        result = await list_users()

        assert len(result) == 3
        assert all(isinstance(u, User) for u in result)


class TestChatFunctions:
    """Test chat-related database functions."""

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_get_user_chats(self, mock_db):
        """Test get_user_chats function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        user_id = uuid4()
        mock_chats = []
        for i in range(2):
            chat = Mock(spec=ChatModel)
            chat.id = uuid4()
            chat.user_id = user_id
            chat.created_at = datetime.now(UTC)
            mock_chats.append(chat)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_chats
        mock_session.execute.return_value = mock_result

        result = await get_user_chats(user_id)

        assert len(result) == 2
        assert all(isinstance(c, Chat) for c in result)

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_create_chat_session(self, mock_db):
        """Test create_chat_session function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        user_id = uuid4()
        chat_id = uuid4()
        mock_chat = Mock(spec=ChatModel)
        mock_chat.id = chat_id
        mock_chat.user_id = user_id
        mock_chat.created_at = datetime.now(UTC)

        with patch("app.db.chat.ChatModel") as MockChatModel:
            MockChatModel.return_value = mock_chat
            result = await create_chat_session(user_id)

        MockChatModel.assert_called_once_with(user_id=user_id)
        mock_session.add.assert_called_once_with(mock_chat)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_chat)

        assert isinstance(result, Chat)

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_get_chat_session(self, mock_db):
        """Test get_chat_session function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        chat_id = uuid4()
        mock_chat = Mock(spec=ChatModel)
        mock_chat.id = chat_id
        mock_chat.user_id = uuid4()
        mock_chat.created_at = datetime.now(UTC)
        mock_session.get.return_value = mock_chat

        result = await get_chat_session(chat_id)

        mock_session.get.assert_called_once_with(ChatModel, chat_id)
        assert isinstance(result, Chat)

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_delete_chat_session(self, mock_db):
        """Test delete_chat_session function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        chat_id = uuid4()
        mock_chat = Mock(spec=ChatModel)
        mock_session.get.return_value = mock_chat

        await delete_chat_session(chat_id)

        mock_session.execute.assert_called_once()  # For deleting messages
        mock_session.get.assert_called_once_with(ChatModel, chat_id)
        mock_session.delete.assert_called_once_with(mock_chat)
        mock_session.commit.assert_called_once()


class TestMessageFunctions:
    """Test message-related database functions."""

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_create_chat_message(self, mock_db):
        """Test create_chat_message function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        message = ChatMessage(chat_id=uuid4(), role=ChatRole.USER, content="Test message")

        mock_message = Mock(spec=ChatMessageModel)
        mock_message.id = uuid4()
        mock_message.chat_id = message.chat_id
        mock_message.role = message.role
        mock_message.content = message.content
        mock_message.tool_calls = None
        mock_message.tool_call_id = None
        mock_message.created_at = datetime.now(UTC)

        with patch("app.db.chat.ChatMessageModel") as MockMessageModel:
            MockMessageModel.return_value = mock_message
            result = await create_chat_message(message)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

        assert isinstance(result, ChatMessage)

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_get_chat_history(self, mock_db):
        """Test get_chat_history function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        chat_id = uuid4()
        mock_messages = []
        for i in range(3):
            msg = Mock(spec=ChatMessageModel)
            msg.id = uuid4()
            msg.chat_id = chat_id
            msg.role = ChatRole.USER if i % 2 == 0 else ChatRole.ASSISTANT
            msg.content = f"Message {i}"
            msg.tool_calls = None
            msg.tool_call_id = None
            msg.created_at = datetime.now(UTC)
            mock_messages.append(msg)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_messages
        mock_session.execute.return_value = mock_result

        result = await get_chat_history(chat_id)

        assert len(result) == 3
        assert all(isinstance(m, ChatMessage) for m in result)


class TestConversationAnalysisFunctions:
    """Test conversation analysis functions."""

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_create_conversation_analysis(self, mock_db):
        """Test create_conversation_analysis function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        chat_id = uuid4()
        analysis_id = uuid4()
        test_data = {
            "chat_id": chat_id,
            "conversation_goal": "Test goal",
            "branches": [{"response": "Branch 1"}],
            "selected_branch_index": 0,
            "selected_response": "Selected response",
            "analysis": "Test analysis",
            "scores": {"score1": 0.8},
            "mcts_statistics": {"stat1": "value1"},
        }

        mock_analysis = Mock(spec=ConversationAnalysisModel)
        mock_analysis.id = analysis_id
        mock_analysis.chat_id = chat_id
        mock_analysis.created_at = datetime.now(UTC)
        mock_analysis.conversation_goal = test_data["conversation_goal"]
        mock_analysis.branches = test_data["branches"]
        mock_analysis.selected_branch_index = test_data["selected_branch_index"]
        mock_analysis.selected_response = test_data["selected_response"]
        mock_analysis.analysis = test_data["analysis"]
        mock_analysis.scores = test_data["scores"]
        mock_analysis.mcts_statistics = test_data["mcts_statistics"]

        with patch("app.db.chat.ConversationAnalysisModel") as MockAnalysisModel:
            MockAnalysisModel.return_value = mock_analysis
            result = await create_conversation_analysis(**test_data)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

        assert isinstance(result, dict)
        assert result["id"] == analysis_id
        assert result["chat_id"] == chat_id

    @pytest.mark.asyncio
    @patch("app.db.chat.db")
    async def test_get_chat_analyses(self, mock_db):
        """Test get_chat_analyses function."""
        mock_session = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None
        mock_db.get_session.return_value = mock_context

        chat_id = uuid4()
        mock_analyses = []
        for i in range(2):
            analysis = Mock(spec=ConversationAnalysisModel)
            analysis.id = uuid4()
            analysis.chat_id = chat_id
            analysis.created_at = datetime.now(UTC)
            analysis.conversation_goal = f"Goal {i}"
            analysis.branches = [{"response": f"Branch {i}"}]
            analysis.selected_branch_index = 0
            analysis.selected_response = f"Response {i}"
            analysis.analysis = f"Analysis {i}"
            analysis.scores = {"score": float(i)}
            analysis.mcts_statistics = {"stat": i}
            mock_analyses.append(analysis)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_analyses
        mock_session.execute.return_value = mock_result

        result = await get_chat_analyses(chat_id)

        assert len(result) == 2
        assert all(isinstance(a, dict) for a in result)
        assert all("id" in a for a in result)
        assert all("chat_id" in a for a in result)
