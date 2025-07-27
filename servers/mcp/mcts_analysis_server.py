import asyncio
import signal
import sys
import time
from typing import Any
from uuid import uuid4

from fastmcp import Context, FastMCP
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schema.conversation_analysis import ConversationBranch
from app.schema.llm.message import Message
from app.services.conversation_analysis import (
    ConversationAnalyzer,
    ConversationScorer,
    ConversationSimulator,
    ResponseGenerator,
)
from app.services.llm_service import LLMService
from app.services.mcts import MCTSAlgorithm
from app.utils.config import app_settings
from app.utils.constants import RETRY_MAX_ATTEMPTS, RETRY_MAX_WAIT, RETRY_MIN_WAIT, RETRY_MULTIPLIER
from app.utils.logger import logger
from app.utils.metrics import metrics_collector, track_mcp_tool

llm_service: LLMService | None = None
response_generator: ResponseGenerator | None = None
simulator: ConversationSimulator | None = None
scorer: ConversationScorer | None = None
analyzer: ConversationAnalyzer | None = None
mcts_algorithm: MCTSAlgorithm | None = None
_services_initialized = False
_initialization_lock = asyncio.Lock()

shutdown_event = asyncio.Event()


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    shutdown_event.set()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    reraise=True,
)
async def initialize_services():
    """Initialize all required services with retry logic."""
    global llm_service, response_generator, simulator, scorer, analyzer, mcts_algorithm
    global _services_initialized

    async with _initialization_lock:
        if _services_initialized:
            return

        logger.info("Initializing MCTS Analysis Server services")

        try:
            llm_service = LLMService(
                base_url=app_settings.LLM_API_BASE_URL,
                api_key=app_settings.LLM_API_KEY,
                model_name=app_settings.LLM_MODEL_NAME,
            )

            response_generator = ResponseGenerator(llm_service)
            simulator = ConversationSimulator(llm_service)
            scorer = ConversationScorer(llm_service)
            analyzer = ConversationAnalyzer(llm_service)

            mcts_algorithm = MCTSAlgorithm(response_generator, simulator, scorer)

            _services_initialized = True
            logger.info("MCTS Analysis Server services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}", exc_info=True)
            raise


def create_mcp_server(initialize_on_startup: bool = True) -> FastMCP:
    """Create the MCP server with optional startup initialization."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(server):
        """Manage server lifecycle."""
        logger.info(
            f"Starting MCTS MCP Server - Version {app_settings.VERSION if hasattr(app_settings, 'VERSION') else 'unknown'}"
        )

        if initialize_on_startup:
            await initialize_services()

        metrics_collector.initialize()
        metrics_collector.update_mcp_sessions(1)

        yield

        metrics_collector.update_mcp_sessions(0)
        logger.info("Shutting down MCTS MCP Server")

    return FastMCP(
        name="MCTS Analysis Server",
        instructions="""
        This server provides Monte Carlo Tree Search (MCTS) analysis for conversations.
        It generates multiple response branches and simulates conversation continuations
        to find the highest EQ response path based on the specified goal.
        """,
        lifespan=lifespan,
    )


mcp = create_mcp_server(
    initialize_on_startup=False
)  # Initialization is deferred to allow transport-specific handling. Ensure transport mechanisms trigger initialization when required.


@mcp.tool
@track_mcp_tool("analyze_conversation")
async def analyze_conversation(
    ctx: Context,
    conversation_goal: str,
    messages: list[dict[str, str]],
    num_branches: int = 5,
    simulation_depth: int = 3,
    max_tokens: int = 250,
    mcts_iterations: int = 10,
    exploration_constant: float = 1.414,
) -> dict[str, Any]:
    """
    Analyzes a conversation using MCTS to find the highest EQ response path.

    This tool:
    1. Generates multiple response branches (default: 5)
    2. Simulates conversation continuations for each branch
    3. Scores each path based on emotional intelligence factors
    4. Returns all branches with analysis of why one was selected

    Args:
        conversation_goal: The goal of the conversation (e.g., 'feel better', 'get constructive criticism')
        messages: Array of messages with 'role' and 'content' fields
        num_branches: Number of initial branches to explore (default: 5)
        simulation_depth: How many turns to simulate ahead (default: 3)
        max_tokens: Maximum tokens per LLM response (default: 250)
        mcts_iterations: Number of MCTS iterations to perform (default: 10)
        exploration_constant: UCB1 exploration constant (default: 1.414, sqrt(2))

    Returns:
        Complete analysis including all branches, selected response, and reasoning
    """
    start_time = time.time()
    analysis_id = uuid4()

    if not _services_initialized:
        await ctx.info("Initializing services for first request...")
        await initialize_services()

    await ctx.info(f"Starting MCTS analysis with goal: {conversation_goal}")

    try:
        message_objects = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

        await ctx.info(f"Generating {num_branches} initial response branches")
        initial_responses = await response_generator.generate_initial_branches(
            message_objects,
            num_branches,
            conversation_goal,
            max_tokens,
        )

        mcts_config = {
            "iterations": mcts_iterations,
            "simulation_depth": simulation_depth,
            "exploration_constant": exploration_constant,
            "goal": conversation_goal,
            "max_tokens": max_tokens,
        }

        await ctx.info(f"Running MCTS with {mcts_iterations} iterations")
        root_nodes, mcts_stats = await mcts_algorithm.run(message_objects, initial_responses, mcts_config)

        await ctx.info("Analyzing best path and generating recommendations")
        best_node, best_idx, analysis = await analyzer.analyze_best_path(
            root_nodes, message_objects, conversation_goal, max_tokens
        )

        branches = analyzer.convert_to_branches(root_nodes)

        scores = {
            "best_score": best_node.avg_score,
            "average_score": sum(node.avg_score for node in root_nodes) / len(root_nodes),
            "score_variance": _calculate_variance([node.avg_score for node in root_nodes]),
        }

        elapsed_time = time.time() - start_time
        await ctx.info(f"Analysis completed in {elapsed_time:.2f}s")

        return {
            "id": str(analysis_id),
            "created_at": time.time(),
            "conversation_goal": conversation_goal,
            "branches": [_branch_to_dict(b) for b in branches],
            "selected_branch_index": best_idx,
            "selected_response": best_node.response,
            "analysis": analysis,
            "overall_scores": scores,
            "mcts_statistics": mcts_stats,
            "processing_time": elapsed_time,
        }

    except asyncio.TimeoutError:
        error_msg = (
            "Analysis timed out. The conversation may be too complex. "
            "Try reducing the simulation depth or number of branches."
        )
        await ctx.error(error_msg)
        raise Exception(error_msg)

    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        await ctx.error(error_msg)
        logger.error(
            "Error analyzing conversation",
            extra={
                "goal": conversation_goal,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


def _calculate_variance(values: list[float]) -> float:
    """Calculate variance of a list of values."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def _branch_to_dict(branch: ConversationBranch) -> dict[str, Any]:
    """Convert a ConversationBranch to a dictionary."""
    return {
        "response": branch.response,
        "simulated_user_reactions": branch.simulated_user_reactions,
        "score": branch.score,
        "sub_history": branch.sub_history,
        "general_metrics": branch.general_metrics,
        "goal_metrics": branch.goal_metrics,
        "visits": branch.visits,
        "parent_index": branch.parent_index,
        "children_indices": branch.children_indices,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCTS Analysis MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="http", help="Transport to use (default: http)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to for HTTP transport (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to for HTTP transport (default: 8001)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args()

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    server = create_mcp_server(initialize_on_startup=(args.transport == "http"))

    try:
        if args.transport == "stdio":
            logger.info("Starting MCTS MCP Server with STDIO transport")
            server.run(
                transport="stdio",
                log_level=args.log_level,
            )
        else:
            logger.info(f"Starting MCTS MCP Server with HTTP transport on {args.host}:{args.port}")
            server.run(
                transport="http",
                host=args.host,
                port=args.port,
                path="/mcp/v1",
                log_level=args.log_level,
            )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)
