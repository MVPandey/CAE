"""Edge case tests for MCTS algorithm."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.mcts.algorithm import MCTSAlgorithm
from app.services.mcts.node import MCTSNode


class TestMCTSAlgorithmEdgeCases:
    """Test edge cases for MCTS algorithm."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        response_generator = AsyncMock()
        simulator = AsyncMock()
        scorer = AsyncMock()
        return response_generator, simulator, scorer

    @pytest.mark.asyncio
    async def test_select_node_no_children(self, mock_services):
        """Test _select_node when node has no children."""
        algo = MCTSAlgorithm(*mock_services)

        root = MCTSNode("response", 0.5)
        root.children = []

        selected = await algo._select_node(root, 1.414)
        assert selected == root

    @pytest.mark.asyncio
    async def test_select_node_not_fully_expanded(self, mock_services):
        """Test _select_node when node is not fully expanded."""
        algo = MCTSAlgorithm(*mock_services)

        root = MCTSNode("response", 0.5)
        child1 = MCTSNode("child1", 0.6)
        root.children = [child1]
        root._is_fully_expanded = False

        root.is_fully_expanded = MagicMock(return_value=False)

        selected = await algo._select_node(root, 1.414)
        assert selected == root
