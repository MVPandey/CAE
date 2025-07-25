# API Unit Tests

This directory contains comprehensive unit tests for all API endpoints in the application.

## Test Coverage

All API modules have 100% test coverage:
- `app.api.user` - 100% coverage
- `app.api.chat` - 100% coverage  
- `app.api.conversation_analysis` - 100% coverage

## Test Structure

### User API Tests (`test_user.py`)
Tests for user management endpoints:
- **POST /users/** - Create user
  - Success case
  - Empty name validation
  - Missing name validation
  - Long name validation
  - Database error handling
- **GET /users/** - List all users
  - Success case
  - Empty list case
  - Database error handling
- **GET /users/{user_id}** - Get specific user
  - Success case
  - User not found
  - Invalid UUID format
  - Database error handling
- **DELETE /users/{user_id}** - Delete user
  - Success case
  - User not found
  - Invalid UUID format
  - Database error handling
- **GET /users/{user_id}/chats** - Get user's chats
  - Success case
  - User not found
  - Empty chats list
  - Invalid UUID format
  - Database error handling

### Chat API Tests (`test_chat.py`)
Tests for chat messaging endpoints:
- **POST /chats/** - Send message
  - New chat creation
  - Existing chat message
  - Service error handling
  - Request validation
- **GET /chats/{chat_id}** - Get chat history
  - Success case
  - Chat not found
- **DELETE /chats/{chat_id}** - Delete chat
  - Success case

### Conversation Analysis API Tests (`test_conversation_analysis.py`)
Tests for MCTS-based conversation analysis:
- **POST /analysis/** - Analyze conversation
  - Success with full parameters
  - Success with minimal parameters
  - ValueError handling (bad request)
  - Timeout handling (gateway timeout)
  - General error handling
  - Request validation
- **GET /analysis/{chat_id}** - Get analyses for chat
  - Success case
  - Empty analyses list
  - Database error handling

## Testing Approach

1. **Mocking Strategy**: 
   - Uses `unittest.mock` for mocking dependencies
   - FastAPI dependency injection overrides for service mocking
   - AsyncMock for async functions

2. **Fixtures**:
   - `async_client` - HTTPX async test client with ASGI transport
   - Mock data fixtures for consistent test data
   - Auto-cleanup of dependency overrides

3. **Test Organization**:
   - Grouped by endpoint functionality
   - Clear test names describing the scenario
   - Comprehensive error case coverage

## Running Tests

```bash
# Run all API tests
python -m pytest tests/unit/api/ -v

# Run specific test file
python -m pytest tests/unit/api/test_user.py -v

# Run with coverage report
python -m pytest tests/unit/api/ -v --cov=app.api --cov-report=term-missing

# Run without coverage check
python -m pytest tests/unit/api/ -v --no-cov
```

## Key Testing Patterns

1. **Dependency Injection Override**:
```python
mock_service = AsyncMock(spec=ChatService)
app.dependency_overrides[ChatService] = lambda: mock_service
try:
    # test code
finally:
    app.dependency_overrides.clear()
```

2. **Direct Mocking**:
```python
with patch("app.api.chat.db.get_chat_history", new_callable=AsyncMock) as mock:
    mock.return_value = test_data
    # test code
```

3. **Validation Testing**:
- Missing required fields
- Invalid data types
- Constraint violations

4. **Error Handling**:
- Database errors (500)
- Not found errors (404)
- Validation errors (422)
- Business logic errors (400)
- Timeout errors (504)