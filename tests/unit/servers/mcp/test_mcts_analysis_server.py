from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Client

from app.schema.llm.message import Message
from servers.mcp.mcts_analysis_server import mcp


@pytest.fixture
def mock_services():
    """Mock all the MCTS services."""
    with (
        patch("servers.mcp.mcts_analysis_server.llm_service") as mock_llm,
        patch("servers.mcp.mcts_analysis_server.response_generator") as mock_gen,
        patch("servers.mcp.mcts_analysis_server.simulator") as mock_sim,
        patch("servers.mcp.mcts_analysis_server.scorer") as mock_scorer,
        patch("servers.mcp.mcts_analysis_server.analyzer") as mock_analyzer,
        patch("servers.mcp.mcts_analysis_server.mcts_algorithm") as mock_mcts,
    ):
        mock_gen.generate_initial_branches = AsyncMock(return_value=["response1", "response2"])

        mock_node1 = Mock(response="response1", avg_score=0.8)
        mock_node2 = Mock(response="response2", avg_score=0.6)
        mock_nodes = [mock_node1, mock_node2]

        mock_mcts.run = AsyncMock(return_value=(mock_nodes, {"iterations": 10}))

        mock_analyzer.analyze_best_path = AsyncMock(return_value=(mock_node1, 0, "Best path analysis"))

        mock_branch1 = Mock(
            response="response1",
            simulated_user_reactions=["reaction1"],
            score=0.8,
            sub_history=[],
            general_metrics={},
            goal_metrics={},
            visits=5,
            parent_index=None,
            children_indices=[],
        )
        mock_branch2 = Mock(
            response="response2",
            simulated_user_reactions=["reaction2"],
            score=0.6,
            sub_history=[],
            general_metrics={},
            goal_metrics={},
            visits=3,
            parent_index=None,
            children_indices=[],
        )

        mock_analyzer.convert_to_branches = Mock(return_value=[mock_branch1, mock_branch2])

        yield {
            "llm_service": mock_llm,
            "response_generator": mock_gen,
            "simulator": mock_sim,
            "scorer": mock_scorer,
            "analyzer": mock_analyzer,
            "mcts_algorithm": mock_mcts,
            "nodes": mock_nodes,
            "branches": [mock_branch1, mock_branch2],
        }


@pytest.mark.asyncio
async def test_mcp_server_tool_registration():
    """Test that the MCP server has the analyze_conversation tool registered."""
    async with Client(mcp) as client:
        tools = await client.list_tools()

        tool_names = [tool.name for tool in tools]
        assert "analyze_conversation" in tool_names

        analyze_tool = next(tool for tool in tools if tool.name == "analyze_conversation")
        assert analyze_tool.name == "analyze_conversation"
        assert "Analyzes a conversation using MCTS" in analyze_tool.description


@pytest.mark.asyncio
async def test_analyze_conversation_basic(mock_services):
    """Test basic conversation analysis functionality."""
    async def fast_initialize_services():
        """Fast initialization without retry."""
        import servers.mcp.mcts_analysis_server as mcts_server
        mcts_server._services_initialized = True
        mcts_server.llm_service = mock_services["llm_service"]
        mcts_server.response_generator = mock_services["response_generator"]
        mcts_server.simulator = mock_services["simulator"]
        mcts_server.scorer = mock_services["scorer"]
        mcts_server.analyzer = mock_services["analyzer"]
        mcts_server.mcts_algorithm = mock_services["mcts_algorithm"]
    
    with patch("servers.mcp.mcts_analysis_server.initialize_services", side_effect=fast_initialize_services):
        from servers.mcp.mcts_analysis_server import initialize_services
        await initialize_services()

    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_conversation",
            {
                "conversation_goal": "feel better",
                "messages": [
                    {"role": "user", "content": "I'm feeling sad today"},
                    {"role": "assistant", "content": "I understand you're feeling sad"},
                ],
            },
        )

        assert result.data["conversation_goal"] == "feel better"
        assert "selected_branch_index" in result.data
        assert isinstance(result.data["selected_branch_index"], int)
        assert "selected_response" in result.data
        assert isinstance(result.data["selected_response"], str)
        assert "analysis" in result.data
        assert "branches" in result.data
        assert isinstance(result.data["branches"], list)
        assert "overall_scores" in result.data


@pytest.mark.asyncio
async def test_analyze_conversation_with_custom_params(mock_services):
    """Test conversation analysis with custom parameters."""
    async def fast_initialize_services():
        """Fast initialization without retry."""
        import servers.mcp.mcts_analysis_server as mcts_server
        mcts_server._services_initialized = True
        mcts_server.llm_service = mock_services["llm_service"]
        mcts_server.response_generator = mock_services["response_generator"]
        mcts_server.simulator = mock_services["simulator"]
        mcts_server.scorer = mock_services["scorer"]
        mcts_server.analyzer = mock_services["analyzer"]
        mcts_server.mcts_algorithm = mock_services["mcts_algorithm"]
    
    with patch("servers.mcp.mcts_analysis_server.initialize_services", side_effect=fast_initialize_services):
        from servers.mcp.mcts_analysis_server import initialize_services
        await initialize_services()

    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_conversation",
            {
                "conversation_goal": "get constructive criticism",
                "messages": [
                    {"role": "user", "content": "Review my code"},
                    {"role": "assistant", "content": "I'll review your code"},
                ],
                "num_branches": 3,
                "simulation_depth": 5,
                "mcts_iterations": 20,
                "max_tokens": 300,
            },
        )

        mock_services["response_generator"].generate_initial_branches.assert_called_with(
            [
                Message(role="user", content="Review my code"),
                Message(role="assistant", content="I'll review your code"),
            ],
            3,  # num_branches
            "get constructive criticism",
            300,  # max_tokens
        )

        assert result.data["conversation_goal"] == "get constructive criticism"


@pytest.mark.asyncio
async def test_analyze_conversation_error_handling():
    """Test error handling in conversation analysis."""
    with patch("servers.mcp.mcts_analysis_server.response_generator") as mock_gen:
        mock_gen.generate_initial_branches = AsyncMock(side_effect=Exception("Test error"))

        async def fast_initialize_services():
            """Fast initialization without retry."""
            import servers.mcp.mcts_analysis_server as mcts_server
            mcts_server._services_initialized = True
            mcts_server.response_generator = mock_gen
        
        with patch("servers.mcp.mcts_analysis_server.initialize_services", side_effect=fast_initialize_services):
            from servers.mcp.mcts_analysis_server import initialize_services
            await initialize_services()

        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "analyze_conversation",
                    {"conversation_goal": "test", "messages": [{"role": "user", "content": "test"}]},
                )

            assert "Test error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """Test that the MCP server initializes correctly."""
    assert mcp.name == "MCTS Analysis Server"
    assert "Monte Carlo Tree Search" in mcp.instructions

    tools = await mcp.get_tools()
    assert "analyze_conversation" in tools


@pytest.mark.asyncio
async def test_variance_calculation():
    """Test the variance calculation function."""
    from servers.mcp.mcts_analysis_server import _calculate_variance

    assert _calculate_variance([]) == 0.0

    assert _calculate_variance([5.0]) == 0.0

    assert abs(_calculate_variance([1.0, 2.0, 3.0, 4.0, 5.0]) - 2.0) < 0.01

    assert _calculate_variance([3.0, 3.0, 3.0]) == 0.0
