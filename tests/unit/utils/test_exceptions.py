"""Unit tests for app.utils.exceptions module."""

import pytest

from app.utils.exceptions import (
    ChatHistoryNotFoundError,
    ConversationAnalysisException,
    LLMException,
    MCTSException,
)


class TestLLMException:
    """Test LLMException class."""

    def test_llm_exception_with_message_only(self):
        """Test LLMException with only a message."""
        message = "LLM API error occurred"
        exception = LLMException(message)

        assert str(exception) == message
        assert exception.message == message
        assert exception.details is None

    def test_llm_exception_with_details(self):
        """Test LLMException with message and details."""
        message = "LLM API error occurred"
        details = {"error_code": "rate_limit", "retry_after": 60}
        exception = LLMException(message, details)

        assert exception.message == message
        assert exception.details == details
        assert str(exception) == f"{message} - Details: {details}"

    def test_llm_exception_with_string_details(self):
        """Test LLMException with string details."""
        message = "Connection failed"
        details = "Timeout after 30 seconds"
        exception = LLMException(message, details)

        assert exception.message == message
        assert exception.details == details
        assert str(exception) == f"{message} - Details: {details}"

    def test_llm_exception_inheritance(self):
        """Test that LLMException inherits from Exception."""
        exception = LLMException("test")
        assert isinstance(exception, Exception)

    def test_llm_exception_raise_and_catch(self):
        """Test raising and catching LLMException."""
        with pytest.raises(LLMException) as exc_info:
            raise LLMException("Test error", {"code": 500})

        assert exc_info.value.message == "Test error"
        assert exc_info.value.details == {"code": 500}


class TestConversationAnalysisException:
    """Test ConversationAnalysisException class."""

    def test_conversation_analysis_exception(self):
        """Test basic ConversationAnalysisException."""
        exception = ConversationAnalysisException("Analysis failed")
        assert str(exception) == "Analysis failed"

    def test_conversation_analysis_exception_inheritance(self):
        """Test that ConversationAnalysisException inherits from Exception."""
        exception = ConversationAnalysisException()
        assert isinstance(exception, Exception)

    def test_conversation_analysis_exception_raise_and_catch(self):
        """Test raising and catching ConversationAnalysisException."""
        with pytest.raises(ConversationAnalysisException) as exc_info:
            raise ConversationAnalysisException("Test analysis error")

        assert str(exc_info.value) == "Test analysis error"


class TestChatHistoryNotFoundError:
    """Test ChatHistoryNotFoundError class."""

    def test_chat_history_not_found_error(self):
        """Test ChatHistoryNotFoundError with chat_id."""
        chat_id = "test-chat-123"
        exception = ChatHistoryNotFoundError(chat_id)

        assert exception.chat_id == chat_id
        assert str(exception) == f"No chat history found for chat_id {chat_id}"

    def test_chat_history_not_found_inheritance(self):
        """Test inheritance chain for ChatHistoryNotFoundError."""
        exception = ChatHistoryNotFoundError("test-id")
        assert isinstance(exception, ConversationAnalysisException)
        assert isinstance(exception, Exception)

    def test_chat_history_not_found_raise_and_catch(self):
        """Test raising and catching ChatHistoryNotFoundError."""
        chat_id = "missing-chat-456"
        with pytest.raises(ChatHistoryNotFoundError) as exc_info:
            raise ChatHistoryNotFoundError(chat_id)

        assert exc_info.value.chat_id == chat_id
        assert chat_id in str(exc_info.value)

    def test_catch_as_parent_exception(self):
        """Test catching ChatHistoryNotFoundError as ConversationAnalysisException."""
        with pytest.raises(ConversationAnalysisException):
            raise ChatHistoryNotFoundError("test-id")


class TestMCTSException:
    """Test MCTSException class."""

    def test_mcts_exception(self):
        """Test basic MCTSException."""
        exception = MCTSException("MCTS algorithm failed")
        assert str(exception) == "MCTS algorithm failed"

    def test_mcts_exception_inheritance(self):
        """Test inheritance chain for MCTSException."""
        exception = MCTSException()
        assert isinstance(exception, ConversationAnalysisException)
        assert isinstance(exception, Exception)

    def test_mcts_exception_raise_and_catch(self):
        """Test raising and catching MCTSException."""
        with pytest.raises(MCTSException) as exc_info:
            raise MCTSException("Tree expansion failed")

        assert str(exc_info.value) == "Tree expansion failed"

    def test_mcts_catch_as_parent_exception(self):
        """Test catching MCTSException as ConversationAnalysisException."""
        with pytest.raises(ConversationAnalysisException):
            raise MCTSException("MCTS error")


class TestExceptionHierarchy:
    """Test the overall exception hierarchy."""

    def test_all_custom_exceptions_inherit_from_exception(self):
        """Test that all custom exceptions inherit from Exception."""
        exceptions = [
            LLMException("test"),
            ConversationAnalysisException("test"),
            ChatHistoryNotFoundError("test-id"),
            MCTSException("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)

    def test_conversation_analysis_subclasses(self):
        """Test ConversationAnalysisException subclasses."""
        chat_error = ChatHistoryNotFoundError("test-id")
        mcts_error = MCTSException("test")

        assert isinstance(chat_error, ConversationAnalysisException)
        assert isinstance(mcts_error, ConversationAnalysisException)

        assert not isinstance(chat_error, MCTSException)
        assert not isinstance(mcts_error, ChatHistoryNotFoundError)
