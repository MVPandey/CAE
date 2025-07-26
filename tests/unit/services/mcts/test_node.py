"""Unit tests for MCTSNode."""
import math

import pytest

from app.services.mcts.node import MCTSNode


class TestMCTSNode:
    """Test cases for MCTSNode."""

    def test_node_initialization(self):
        """Test basic node initialization."""
        response = "Test response"
        node = MCTSNode(response)

        assert node.response == response
        assert node.parent is None
        assert node.children == []
        assert node.index == 0
        assert node.visits == 0
        assert node.total_score == 0.0
        assert node.avg_score == 0.0
        assert node.simulated_reactions == []
        assert node.sub_history == []
        assert node.general_metrics == {}
        assert node.goal_metrics == {}

    def test_node_initialization_with_parent(self):
        """Test node initialization with parent."""
        parent = MCTSNode("Parent response")
        child = MCTSNode("Child response", parent=parent, index=5)

        assert child.response == "Child response"
        assert child.parent == parent
        assert child.index == 5

    def test_add_child(self):
        """Test adding child nodes."""
        parent = MCTSNode("Parent")
        child1 = MCTSNode("Child 1")
        child2 = MCTSNode("Child 2")

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert parent.children[0] == child1
        assert parent.children[1] == child2
        assert child1.parent == parent
        assert child2.parent == parent
        assert child1.index == 0
        assert child2.index == 1

    def test_add_child_updates_existing_parent(self):
        """Test that add_child updates child's existing parent."""
        parent1 = MCTSNode("Parent 1")
        parent2 = MCTSNode("Parent 2")
        child = MCTSNode("Child", parent=parent1)

        parent2.add_child(child)

        assert child.parent == parent2
        assert child in parent2.children
        assert child.index == 0

    def test_is_fully_expanded_default(self):
        """Test is_fully_expanded with default max_children."""
        node = MCTSNode("Node")

        assert not node.is_fully_expanded()

        for i in range(3):
            node.add_child(MCTSNode(f"Child {i}"))

        assert node.is_fully_expanded()

    def test_is_fully_expanded_custom_limit(self):
        """Test is_fully_expanded with custom max_children."""
        node = MCTSNode("Node")

        node.add_child(MCTSNode("Child 1"))
        node.add_child(MCTSNode("Child 2"))

        assert not node.is_fully_expanded(max_children=5)
        assert node.is_fully_expanded(max_children=2)
        assert node.is_fully_expanded(max_children=1)

    def test_best_child_single_child(self):
        """Test best_child with single child."""
        parent = MCTSNode("Parent")
        child = MCTSNode("Child")
        parent.add_child(child)

        parent.visits = 10

        best = parent.best_child()
        assert best == child

    def test_best_child_no_children(self):
        """Test best_child raises error when no children."""
        node = MCTSNode("Node")

        with pytest.raises(ValueError, match="No children to select from"):
            node.best_child()

    def test_best_child_unvisited_children(self):
        """Test best_child prioritizes unvisited children."""
        parent = MCTSNode("Parent")
        parent.visits = 10

        visited_child = MCTSNode("Visited")
        visited_child.visits = 5
        visited_child.avg_score = 0.8

        unvisited_child = MCTSNode("Unvisited")
        unvisited_child.visits = 0

        parent.add_child(visited_child)
        parent.add_child(unvisited_child)

        best = parent.best_child()
        assert best == unvisited_child

    def test_best_child_ucb1_calculation(self):
        """Test best_child uses correct UCB1 calculation."""
        parent = MCTSNode("Parent")
        parent.visits = 100

        child1 = MCTSNode("Child 1")
        child1.visits = 20
        child1.avg_score = 0.8

        child2 = MCTSNode("Child 2")
        child2.visits = 10
        child2.avg_score = 0.85  # Higher score but fewer visits

        child3 = MCTSNode("Child 3")
        child3.visits = 30
        child3.avg_score = 0.75

        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)

        c = 1.414  # Default exploration constant

        0.8 + c * math.sqrt(2 * math.log(100) / 20)
        0.85 + c * math.sqrt(2 * math.log(100) / 10)
        0.75 + c * math.sqrt(2 * math.log(100) / 30)

        best = parent.best_child()
        assert best == child2

    def test_best_child_custom_exploration_constant(self):
        """Test best_child with custom exploration constant."""
        parent = MCTSNode("Parent")
        parent.visits = 50

        child1 = MCTSNode("Child 1")
        child1.visits = 25
        child1.avg_score = 0.8

        child2 = MCTSNode("Child 2")
        child2.visits = 25
        child2.avg_score = 0.7

        parent.add_child(child1)
        parent.add_child(child2)

        best_low_exploration = parent.best_child(exploration_constant=0.1)
        assert best_low_exploration == child1

        best_high_exploration = parent.best_child(exploration_constant=10.0)
        assert best_high_exploration in [child1, child2]

    def test_update(self):
        """Test node update method."""
        node = MCTSNode("Node")

        assert node.visits == 0
        assert node.total_score == 0.0
        assert node.avg_score == 0.0

        node.update(0.8)
        assert node.visits == 1
        assert node.total_score == 0.8
        assert node.avg_score == 0.8

        node.update(0.6)
        assert node.visits == 2
        assert node.total_score == 1.4
        assert node.avg_score == 0.7

        node.update(0.9)
        assert node.visits == 3
        assert node.total_score == 2.3
        assert node.avg_score == pytest.approx(0.7667, rel=1e-3)

    def test_ucb1_score_calculation(self):
        """Test internal UCB1 score calculation."""
        parent = MCTSNode("Parent")
        parent.visits = 100

        child = MCTSNode("Child")
        parent.add_child(child)

        ucb1_score = parent._ucb1_score(child, 1.414)
        assert ucb1_score == float("inf")

        child.visits = 10
        child.avg_score = 0.75

        expected_exploration = 1.414 * math.sqrt(2 * math.log(100) / 10)
        expected_ucb1 = 0.75 + expected_exploration

        ucb1_score = parent._ucb1_score(child, 1.414)
        assert ucb1_score == pytest.approx(expected_ucb1, rel=1e-6)

    def test_node_data_attributes(self):
        """Test node data attributes can be set and retrieved."""
        node = MCTSNode("Response")

        node.simulated_reactions = ["User is happy", "User understood"]
        node.sub_history = [
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome!"}
        ]

        node.general_metrics = {
            "clarity": 0.9,
            "relevance": 0.85,
            "engagement": 0.8
        }
        node.goal_metrics = {
            "task_completion": 0.95,
            "user_satisfaction": 0.9
        }

        assert len(node.simulated_reactions) == 2
        assert node.simulated_reactions[0] == "User is happy"
        assert len(node.sub_history) == 2
        assert node.sub_history[0]["role"] == "user"
        assert node.general_metrics["clarity"] == 0.9
        assert node.goal_metrics["task_completion"] == 0.95

    def test_multiple_children_indexing(self):
        """Test that children maintain correct indices."""
        parent = MCTSNode("Parent")

        children = []
        for i in range(5):
            child = MCTSNode(f"Child {i}")
            parent.add_child(child)
            children.append(child)

        for i, child in enumerate(children):
            assert child.index == i

    def test_node_tree_structure(self):
        """Test building a multi-level tree structure."""
        root = MCTSNode("Root")

        level1_a = MCTSNode("Level 1 A")
        level1_b = MCTSNode("Level 1 B")
        root.add_child(level1_a)
        root.add_child(level1_b)

        level2_a1 = MCTSNode("Level 2 A1")
        level2_a2 = MCTSNode("Level 2 A2")
        level1_a.add_child(level2_a1)
        level1_a.add_child(level2_a2)

        level2_b1 = MCTSNode("Level 2 B1")
        level1_b.add_child(level2_b1)

        assert len(root.children) == 2
        assert len(level1_a.children) == 2
        assert len(level1_b.children) == 1
        assert level2_a1.parent == level1_a
        assert level2_b1.parent == level1_b

    def test_update_with_zero_score(self):
        """Test updating node with zero score."""
        node = MCTSNode("Node")

        node.update(0.0)
        assert node.visits == 1
        assert node.total_score == 0.0
        assert node.avg_score == 0.0

        node.update(1.0)
        assert node.visits == 2
        assert node.total_score == 1.0
        assert node.avg_score == 0.5

    def test_update_with_negative_score(self):
        """Test updating node with negative score (edge case)."""
        node = MCTSNode("Node")

        node.update(-0.5)
        assert node.visits == 1
        assert node.total_score == -0.5
        assert node.avg_score == -0.5
