# MCTS Analysis MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for Monte Carlo Tree Search (MCTS) conversation analysis.

## Overview

The MCTS MCP server replaces the FastAPI endpoint for conversation analysis. It provides the same functionality but through the MCP protocol, allowing for better integration with AI assistants and tools that support MCP.

## Key Changes from FastAPI

1. **No Chat ID Required**: The MCP server accepts messages directly instead of requiring a chat_id
2. **Stateless Operation**: Results are not persisted to the database - implement your own storage if needed
3. **Streamable HTTP Transport**: Runs as a separate process with HTTP interface
4. **Direct Parameters**: All MCTS parameters can be passed directly in the request

## Running the Server

### Direct execution:
```bash
python servers/mcp/mcts_analysis_server.py --host 0.0.0.0 --port 8001
```

### Using the standalone script:
```bash
python servers/mcp/run_mcts_mcp_server.py --host 0.0.0.0 --port 8001
```

### Command Line Arguments:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8001)
- `--log-level`: Log level (default: INFO)

### Environment Variables:
You can also configure the server using environment variables:
- `MCP_HOST`: Host to bind to
- `MCP_PORT`: Port to bind to
- `MCP_LOG_LEVEL`: Log level

## API Endpoint

The server exposes its MCP interface at: `http://<host>:<port>/mcp/v1`

## Tool: analyze_conversation

Analyzes a conversation using MCTS to find the highest EQ response path.

### Parameters:
- `conversation_goal` (str, required): The goal of the conversation (e.g., 'feel better', 'get constructive criticism')
- `messages` (list[dict], required): Array of messages with 'role' and 'content' fields
- `num_branches` (int, default: 5): Number of initial branches to explore
- `simulation_depth` (int, default: 3): How many turns to simulate ahead
- `max_tokens` (int, default: 250): Maximum tokens per LLM response
- `mcts_iterations` (int, default: 10): Number of MCTS iterations to perform
- `exploration_constant` (float, default: 1.414): UCB1 exploration constant (sqrt(2))

### Example Request:
```json
{
  "conversation_goal": "feel better",
  "messages": [
    {"role": "user", "content": "I'm feeling sad today"},
    {"role": "assistant", "content": "I understand you're feeling sad"}
  ],
  "num_branches": 5,
  "simulation_depth": 3
}
```

### Response:
Returns a complete analysis including:
- `id`: Unique analysis ID
- `created_at`: Timestamp
- `conversation_goal`: The goal used for optimization
- `branches`: All explored conversation branches
- `selected_branch_index`: Index of the recommended branch
- `selected_response`: The chosen response text
- `analysis`: Detailed reasoning for the selection
- `overall_scores`: Aggregated scoring metrics
- `mcts_statistics`: Algorithm performance stats
- `processing_time`: Time taken for analysis

## Testing

Run the tests with:
```bash
pytest tests/test_mcp_mcts_server.py
```

## Migration from FastAPI

The FastAPI endpoints at `/api/analysis` have been marked as deprecated. To migrate:

1. Instead of sending a chat_id, send the messages array directly
2. Use an MCP client library to connect to the server
3. Call the `analyze_conversation` tool with the required parameters
4. Handle the response (note: it's not persisted to the database)

## Dependencies

The server requires `fastmcp` which is included in `pyproject.toml`. Install with:
```bash
poetry install
```

or

```bash
pip install fastmcp
```