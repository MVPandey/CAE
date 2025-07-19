# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Conversational Analysis Engine (CAE) is an advanced system that combines Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS) algorithms to optimize conversational paths. It serves dual purposes:
- Standard chat API for production use
- Analysis engine for exploring optimal response paths using MCTS

## Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your configuration
```

### Running the Application
```bash
# Local development with hot-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

# Using startup script
./scripts/start.sh

# Docker development
./scripts/docker-start.sh

# Docker production
docker-compose -f docker-compose.prod.yml up -d
```

### Code Quality
```bash
# Format code
black app/

# Lint code
flake8 app/
```

### Database
```bash
# Run migrations
python migrations/add_users_table.py
python migrations/add_conversation_goal_and_mcts_stats.py

# Access PostgreSQL
docker exec -it cae-postgres psql -U cae_user -d conversation_analysis
```

## Architecture

### Layer Structure
```
API Layer (FastAPI routers) → Service Layer (business logic) → Database Layer (SQLAlchemy)
```

### Key Components

**API Endpoints** (`/app/api/`)
- `/users/` - User management
- `/chats/` - Standard chat functionality  
- `/analysis/` - MCTS-based conversation analysis

**Services** (`/app/services/`)
- `chat_service.py` - Chat session management
- `llm_service.py` - LLM interaction layer
- `conversation_analysis_service.py` - MCTS orchestration

**MCTS Implementation** (`/app/services/mcts/`)
- `algorithm.py` - Core MCTS loop with parallel evaluation
- `node.py` - Tree node with UCB1 selection
- `tree_operations.py` - Tree manipulation utilities

**Analysis Components** (`/app/services/conversation_analysis/`)
- `response_generator.py` - Generate response branches
- `conversation_simulator.py` - Simulate future turns
- `conversation_scorer.py` - Multi-metric evaluation
- `conversation_analyzer.py` - Path analysis and explanations

### Key Design Patterns
- **Async Throughout**: All I/O operations use async/await
- **Dependency Injection**: FastAPI's `Depends` for service injection
- **Strategy Pattern**: Separate strategies for generation, simulation, scoring
- **Tool System**: Extensible framework with abstract base classes

### Data Flow

**Standard Chat**:
Client → POST `/chats/` → ChatService → LLMService → OpenAI API → Store in DB → Return response

**MCTS Analysis**:
1. Load conversation history
2. Generate initial branches (default: 5)
3. Run MCTS iterations:
   - Select (UCB1) → Expand → Simulate → Backpropagate
   - Parallel branch evaluation
   - Periodic pruning
4. Select best branch by score
5. Store analysis and return results

### Database Schema
- `UserModel` → Many `ChatModel` (cascade delete)
- `ChatModel` → Many `ChatMessageModel` (cascade delete)
- `ChatModel` → Many `ConversationAnalysisModel`

## Development Guidelines

### Code Standards
- Python 3.10+ with type hints
- Async/await for all I/O operations
- Pydantic for data validation
- Black formatting (88 char limit)
- Comprehensive error handling

### When Working on MCTS
- Maintain separation between algorithm and LLM calls
- Preserve parallel processing capabilities
- Test with various conversation goals
- Consider memory usage for deep trees

### Adding Features
1. Define schemas in `/app/schema/`
2. Implement service logic in `/app/services/`
3. Add API endpoints in `/app/api/`
4. Update tests (when they exist)

### Performance Considerations
- Aggressive parallelization in MCTS
- Connection pooling for database
- Configurable timeouts for long analyses
- Stateless services for scalability