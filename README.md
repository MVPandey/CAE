# Conversational Analysis Engine (CAE)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Test Coverage](https://img.shields.io/badge/coverage-95%25+-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)

> **Optimize LLM responses using search algorithms** - A production-ready FastAPI backend and MCP server that leverages Monte Carlo Tree Search (MCTS) to intelligently explore and evaluate conversation paths for enhanced response quality.

## Table of Contents

- [Quick Start (Docker Compose)](#quick-start-docker-compose)
- [Overview](#overview)
- [Architecture](#architecture)
- [Service Modes & Usage](#service-modes--usage)
- [Claude Desktop Integration](#claude-desktop-integration)
- [Manual Setup](#manual-setup)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

## Quick Start (Docker Compose)

**Recommended Installation** - Automatically provisions Redis, PostgreSQL/PGVector, and Prometheus:

### Prerequisites
- **Docker ≥ 24.0**
- **Docker Compose v2**

### One-Command Setup

```bash
git clone https://github.com/yourusername/ConversationalAnalysisEngine
cd ConversationalAnalysisEngine
docker compose up --build
```

### What You Get
- **API Backend**: `http://localhost:8000` (health checks, conditional metrics)
- **MCP Server**: `http://localhost:8001/mcp/v1` (conversation analysis)
- **Redis Cache**: `localhost:6379`
- **PostgreSQL/PGVector**: `localhost:5432`

### Optional Monitoring (Add `--profile monitoring`)
```bash
docker compose --profile monitoring up --build
```
- **Prometheus Metrics**: `http://localhost:9090`
- **Grafana Dashboard**: `http://localhost:3000` (admin/admin)

### Shutdown
```bash
docker compose down -v
```

## Overview

The **Conversational Analysis Engine (CAE)** enhances LLM response optimization by applying advanced search algorithms to conversation paths. Instead of generating single responses, CAE uses **Monte Carlo Tree Search (MCTS)** to:

1. **Generate multiple response branches** for any conversation context
2. **Simulate conversation continuations** to predict outcomes
3. **Score paths based on goal-specific metrics** (emotional intelligence, persuasiveness, helpfulness)
4. **Select the optimal response** through intelligent exploration

**Dual Architecture:**
- **FastAPI Backend**: Production-ready API with health checks, metrics, and monitoring
- **MCP Server**: Monte Carlo Tree Search optimization via Model Context Protocol

## Architecture

```
[Claude Desktop/Code] → [MCP Server :8001] ↔ [MCTS Algorithm] ↔ [Redis Cache]
[MCP Clients]         →                                      ↓
                                                    [Response Generator]
[Health Checks]       → [API Backend :8000] ← metrics ← [Prometheus :9090]
[Monitoring]          →                              ↓
                                            [PostgreSQL/PGVector :5432]
```

**Service Architecture:**
- **MCP Server** (Port 8001): Conversation analysis via MCTS algorithm
- **API Backend** (Port 8000): Health checks, conditional metrics endpoints
- **Redis** (Port 6379): **Required** - Conversation storage and caching
- **PostgreSQL/PGVector** (Port 5432): **Required** - Conversation storage
- **Prometheus** (Port 9090): **Optional** - Metrics collection (monitoring profile)
- **Grafana** (Port 3000): **Optional** - Metrics dashboard (monitoring profile)

### Core Components

- **MCTS Algorithm** (`app/services/mcts/`): Monte Carlo Tree Search implementation with UCB1 exploration
- **Response Generator**: Creates diverse response branches using LLM variations
- **Conversation Simulator**: Predicts user reactions and conversation continuations
- **Conversation Scorer**: Evaluates path quality based on customizable metrics
- **Semantic Cache**: Redis-based caching with embedding similarity for performance optimization
- **Metrics Collection**: Prometheus metrics for production monitoring

## Service Modes & Usage

### A. MCP Server (Recommended)

**Docker (Recommended):**
```bash
docker compose up mcp
```

**Local Development:**
```bash
poetry run python servers/mcp/mcts_analysis_server.py --transport http --port 8001
```

**Features:**
- ✅ **MCTS-Powered Conversation Analysis**: Multi-branch exploration with intelligent search
- ✅ **Goal-Oriented Optimization**: Customize for empathy, persuasion, problem-solving
- ✅ **Configurable Parameters**: Branch count, simulation depth, exploration constants
- ✅ **Real-time Processing**: Efficient async processing with resource management

### B. API Backend (Health & Monitoring)

**Docker:**
```bash
docker compose up api
```

**Local Development:**
```bash
poetry run python -m app.main
```

**Features:**
- ✅ Health checks at `GET /health`
- ✅ Conditional Prometheus metrics at `GET /metrics` (when enabled)
- ✅ Service monitoring and logging

### ⚠️ Deprecated: REST Analysis Endpoint

**WARNING:** The `POST /api/v1/analyze` endpoint is **deprecated** and returns HTTP 410.

**Migration Path:** Use the MCP server for all conversation analysis:

```python
# ❌ Deprecated - DO NOT USE
response = httpx.post("http://localhost:8000/api/v1/analyze", ...)

# ✅ Use MCP Server instead
from mcp import Client
client = Client("http://localhost:8001/mcp/v1")
result = await client.call_tool("analyze_conversation", ...)
```

### MCP Tool Usage

The server exposes the `analyze_conversation` tool with the following signature:

```typescript
// TypeScript/JavaScript MCP Client Example
import { Client } from '@modelcontextprotocol/sdk/client/index.js';

const client = new Client({
    name: "cae-client",
    version: "1.0.0"
});

const result = await client.callTool("analyze_conversation", {
    conversation_goal: "help user feel better about their situation", 
    messages: [
        {role: "user", content: "I failed my exam and feel terrible"},
        {role: "assistant", content: "I'm sorry to hear about your exam."}
    ],
    num_branches: 3,
    simulation_depth: 2,
    mcts_iterations: 10
});

console.log("Optimized response:", result.selected_response);
console.log("Analysis:", result.analysis);
```

## Claude Desktop Integration

To use CAE with Claude Desktop, add the MCP server to your configuration:

1. **Open Claude Desktop configuration**:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add CAE MCP server configuration**:

```json
{
  "mcpServers": {
    "conversational-analysis-engine": {
      "command": "docker",
      "args": [
        "compose", "-f", "/path/to/ConversationalAnalysisEngine/docker-compose.yml", 
        "up", "mcp", "--build"
      ],
      "env": {
        "LLM_API_KEY": "your_openai_api_key"
      }
    }
  }
}
```

3. **Restart Claude Desktop** to load the MCP server

4. **Use in conversations**:
```
I need to respond to a difficult customer complaint. Can you use the MCTS analysis to help me find the best response?

Goal: Maintain customer relationship while addressing concerns
Current conversation: [customer complaint details]
```

### Claude Code Integration

For Claude Code users, configure the MCP server in your settings:

```json
{
  "mcp": {
    "servers": {
      "cae": {
        "command": "docker",
        "args": [
          "compose", "-f", "/path/to/ConversationalAnalysisEngine/docker-compose.yml",
          "up", "mcp", "--build"
        ],
        "env": {
          "LLM_API_KEY": "your_openai_api_key"
        }
      }
    }
  }
}
```

## Manual Setup

For advanced users who prefer manual installation:

### Prerequisites

- **Python 3.12+**
- **Poetry** (package manager)
- **Redis** (required for caching)
- **PostgreSQL with PGVector** (required for conversation storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ConversationalAnalysisEngine
cd ConversationalAnalysisEngine

# Install dependencies with Poetry
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Environment Configuration

**Minimal Setup** - Only one environment variable required:

```env
# REQUIRED: LLM Configuration
LLM_API_KEY=your_openai_api_key
```

**Full Setup** - All optional configuration with smart defaults:

```env
# REQUIRED: LLM Configuration
LLM_API_KEY=your_openai_api_key

# LLM Configuration (optional - smart defaults)
LLM_API_BASE_URL=https://api.openai.com/v1  # Default
LLM_MODEL_NAME=o3-mini                      # Default

# OPTIONAL: Embedding Configuration (enables semantic caching when present)
EMBEDDING_MODEL_API_KEY=your_openai_api_key      # Optional
EMBEDDING_MODEL_BASE_URL=https://api.openai.com/v1  # Default
EMBEDDING_MODEL_NAME=text-embedding-3-large      # Default

# Feature Toggles (optional)
DISABLE_PROMETHEUS_METRICS=false  # Default: metrics enabled

# Database Configuration (Docker Compose defaults)
DB_HOST=postgres           # Default for Docker
DB_PORT=5432              # Default
DB_NAME=conversation_analysis  # Default
DB_USER=cae_user          # Default
DB_SECRET=cae_password    # Default

# Redis Configuration (Docker Compose defaults)
REDIS_HOST=redis          # Default for Docker
REDIS_PORT=6379           # Default

# Application Settings (optional)
LOG_LEVEL=INFO            # Default
LLM_TIMEOUT_SECONDS=600   # Default
```

**Alternative Providers** (e.g., OpenRouter, Groq):

```env
# OpenRouter Example
LLM_API_KEY=your_openrouter_api_key
LLM_API_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=anthropic/claude-3-sonnet

# Groq Example  
LLM_API_KEY=your_groq_api_key
LLM_API_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL_NAME=llama-3.1-8b-instant

# For semantic caching with different embedding provider
EMBEDDING_MODEL_API_KEY=your_embedding_provider_key
EMBEDDING_MODEL_BASE_URL=https://api.your-provider.com/v1
EMBEDDING_MODEL_NAME=your-embedding-model
```

### Infrastructure Setup

```bash
# Start only infrastructure with Docker Compose
docker compose -f compose.infrastructure.yml up

# Or start services manually:
redis-server
# Configure PostgreSQL with PGVector extension
```

### MCTS Configuration

Customize search behavior through configuration:

```python
# High-quality, slower analysis
config = {
    "num_branches": 8,           # More initial branches
    "mcts_iterations": 20,       # More iterations
    "simulation_depth": 4,       # Deeper simulations
    "exploration_constant": 1.0  # Balanced exploration
}

# Fast, real-time analysis  
config = {
    "num_branches": 3,
    "mcts_iterations": 5,
    "simulation_depth": 2,
    "exploration_constant": 2.0  # More exploration
}
```

## Development

### Code Quality

This project maintains high code quality standards:

```bash
# Linting and formatting
poetry run ruff format .
poetry run ruff check .

# Type checking
poetry run mypy app/

# Run all quality checks
make quality-check
```

### Development Servers

```bash
# Start API backend with hot reload
poetry run uvicorn app.main:app --reload --port 8000

# Start MCP server with debug logging
poetry run python servers/mcp/mcts_analysis_server.py --log-level DEBUG

# Start both with Docker Compose
docker compose up --build
```

## Testing

Comprehensive test suite with **95%+ coverage requirement**:

```bash
# Run all tests
poetry run pytest

# With coverage report
poetry run pytest --cov=app --cov-report=html --cov-report=term

# Run specific test categories
poetry run pytest tests/unit/           # Unit tests
poetry run pytest tests/integration/   # Integration tests
poetry run pytest tests/e2e/          # End-to-end tests

# Performance tests
poetry run pytest tests/performance/ -v
```

**Test Structure:**
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: Service interaction testing
- **E2E Tests**: Full workflow testing via API/MCP
- **Performance Tests**: Load and latency testing

## Contributing

Open-source contributions are welcome! Please follow these guidelines:

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/yourusername/ConversationalAnalysisEngine`
3. **Install dependencies**: `poetry install`
4. **Create a feature branch** using the naming convention:

```bash
# Branch naming format: feature/<feature-abbrev>-<issue-num>-<tag-line>
git checkout -b feature/CRITICAL-1-CORS-fix
git checkout -b feature/PERF-23-redis-optimization
git checkout -b feature/MCTS-45-branching-strategy
```

### Code Standards

- **Linter**: Use `ruff` for code formatting and linting
- **Test Coverage**: Maintain ≥95% test coverage for all new code
- **Type Hints**: All functions must have proper type annotations
- **Documentation**: Update docstrings and README for new features

### Pull Request Process

1. **Status Checks**: Ensure all CI checks pass (tests, linting, coverage)
2. **PR Approval**: At least one approval required from maintainer
3. **Branch Protection**: Feature branches must be up-to-date with main
4. **Documentation**: Update relevant documentation for new features

### Commit Guidelines

```bash
# Good commit messages
git commit -m "feat: add semantic caching for MCTS nodes"
git commit -m "fix: handle timeout errors in conversation simulation"
git commit -m "docs: update API examples in README"
```

## Roadmap

### Current Focus: Performance & Scalability

- 🚧 **Unified LLM Evaluation**: Combine multiple LLM calls into single requests (66% cost reduction)
- 🚧 **Advanced Semantic Caching**: Embedding-based cache with similarity detection
- 🚧 **Resource-Aware Scheduling**: Dynamic resource allocation and request prioritization

### Near-Term: Domain Abstraction (Coming Soon)

- 🔮 **Domain-Agnostic MCTS**: Generalized search framework for various optimization tasks beyond conversation
- 🔮 **Alternative Search Algorithms**: Beam search, A* with heuristics, hybrid approaches
- 🔮 **Multi-Objective Optimization**: Simultaneous optimization for multiple conversation goals

### Future Enhancements: Advanced Search Mechanisms

- 🔮 **Reinforcement Learning Integration**: Learning-based path selection improvements
- 🔮 **Distributed Processing**: Horizontal scaling with work queues
- 🔮 **Model Cascading**: Use smaller models for simulation, larger for final generation
- 🔮 **Advanced Analytics**: Conversation pattern analysis and success prediction

## License

This project is licensed under the **MIT License**.

---

## Acknowledgments

- **FastMCP** for excellent MCP server framework
- **OpenAI** for OpenAI API SPec
- **Anthropic** for MCP specification and Claude integration

## Support

- **Documentation**: Detailed guides in `/docs`
- **Issues**: Report bugs and feature requests via GitHub Issues

---

*Built by [Manav Pandey](mailto:manavpandey1999@gmail.com) for the AI community - enabling smarter conversations through algorithmic optimization.*