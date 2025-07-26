#!/usr/bin/env python3
"""
Standalone script to run the MCTS MCP server.
This allows running the server as a separate process.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCTS Analysis MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.getenv("MCP_TRANSPORT", "http"),
        help="Transport to use (default: http)"
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"), help="Host to bind to for HTTP transport (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8001")), help="Port to bind to for HTTP transport (default: 8001)")
    parser.add_argument("--log-level", default=os.getenv("MCP_LOG_LEVEL", "INFO"), help="Log level (default: INFO)")

    args = parser.parse_args()

    from app.utils.logger import logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    try:
        from servers.mcp.mcts_analysis_server import create_mcp_server
        server = create_mcp_server(initialize_on_startup=(args.transport == "http"))

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
