"""Unit tests for TreeOperations."""

from app.services.mcts.node import MCTSNode
from app.services.mcts.tree_operations import TreeOperations


class TestTreeOperations:
    """Test cases for TreeOperations."""

    def test_backpropagate_single_node(self):
        """Test backpropagation with single node."""
        node = MCTSNode("Node")
        score = 0.85

        TreeOperations.backpropagate(node, score)

        assert node.visits == 1
        assert node.total_score == 0.85
        assert node.avg_score == 0.85

    def test_backpropagate_path(self):
        """Test backpropagation along a path."""
        root = MCTSNode("Root")
        child = MCTSNode("Child")
        grandchild = MCTSNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        score = 0.9
        TreeOperations.backpropagate(grandchild, score)

        assert grandchild.visits == 1
        assert grandchild.avg_score == 0.9

        assert child.visits == 1
        assert child.avg_score == 0.9

        assert root.visits == 1
        assert root.avg_score == 0.9

    def test_backpropagate_multiple_scores(self):
        """Test backpropagation with multiple updates."""
        root = MCTSNode("Root")
        child = MCTSNode("Child")
        root.add_child(child)

        TreeOperations.backpropagate(child, 0.8)
        assert child.avg_score == 0.8
        assert root.avg_score == 0.8

        TreeOperations.backpropagate(child, 0.6)
        assert child.visits == 2
        assert child.avg_score == 0.7
        assert root.visits == 2
        assert root.avg_score == 0.7

    def test_prune_branches_basic(self):
        """Test basic branch pruning."""
        root = MCTSNode("Root")
        root.visits = 10
        root.avg_score = 0.8

        good_child = MCTSNode("Good child")
        good_child.visits = 5
        good_child.avg_score = 0.85

        bad_child = MCTSNode("Bad child")
        bad_child.visits = 5
        bad_child.avg_score = 0.5  # Below threshold

        root.add_child(good_child)
        root.add_child(bad_child)

        pruned_count = TreeOperations.prune_branches([root])

        assert pruned_count == 1
        assert len(root.children) == 1
        assert root.children[0] == good_child

    def test_prune_branches_with_threshold_ratio(self):
        """Test pruning with custom threshold ratio."""
        root = MCTSNode("Root")
        root.visits = 10
        root.avg_score = 1.0

        children_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        children = []

        for i, score in enumerate(children_scores):
            child = MCTSNode(f"Child {i}")
            child.visits = 5
            child.avg_score = score
            root.add_child(child)
            children.append(child)

        pruned_count = TreeOperations.prune_branches([root], threshold_ratio=0.8)

        assert pruned_count == 3  # Children with scores 0.7, 0.6, 0.5
        assert len(root.children) == 2
        assert all(child.avg_score >= 0.8 for child in root.children)

    def test_prune_branches_skip_low_visits(self):
        """Test that nodes with low visits are skipped during pruning."""
        root = MCTSNode("Root")
        root.visits = 3  # Below MIN_VISITS_FOR_PRUNING (5)
        root.avg_score = 0.8

        child = MCTSNode("Child")
        child.visits = 2
        child.avg_score = 0.3  # Would be pruned if root had enough visits
        root.add_child(child)

        pruned_count = TreeOperations.prune_branches([root])

        assert pruned_count == 0
        assert len(root.children) == 1  # Child not pruned

    def test_prune_branches_recursive(self):
        """Test recursive pruning of nested branches."""
        root = MCTSNode("Root")
        root.visits = 10
        root.avg_score = 0.8

        child = MCTSNode("Child")
        child.visits = 8
        child.avg_score = 0.75
        root.add_child(child)

        good_grandchild = MCTSNode("Good grandchild")
        good_grandchild.visits = 4
        good_grandchild.avg_score = 0.8

        bad_grandchild = MCTSNode("Bad grandchild")
        bad_grandchild.visits = 4
        bad_grandchild.avg_score = 0.4  # Below threshold

        child.add_child(good_grandchild)
        child.add_child(bad_grandchild)

        great_grandchild = MCTSNode("Great grandchild")
        bad_grandchild.add_child(great_grandchild)

        pruned_count = TreeOperations.prune_branches([root], threshold_ratio=0.7)

        assert pruned_count == 2  # bad_grandchild + great_grandchild
        assert len(child.children) == 1
        assert child.children[0] == good_grandchild

    def test_prune_branches_multiple_roots(self):
        """Test pruning with multiple root nodes."""
        roots = []

        for i in range(3):
            root = MCTSNode(f"Root {i}")
            root.visits = 10
            root.avg_score = 0.8

            for j in range(2):
                child = MCTSNode(f"Child {i}-{j}")
                child.visits = 5
                child.avg_score = 0.9 if j == 0 else 0.4
                root.add_child(child)

            roots.append(root)

        pruned_count = TreeOperations.prune_branches(roots, threshold_ratio=0.7)

        assert pruned_count == 3
        for root in roots:
            assert len(root.children) == 1
            assert root.children[0].avg_score >= 0.56  # 0.8 * 0.7

    def test_count_descendants(self):
        """Test counting descendants of a node."""
        root = MCTSNode("Root")

        child1 = MCTSNode("Child 1")
        child2 = MCTSNode("Child 2")
        root.add_child(child1)
        root.add_child(child2)

        grandchild1 = MCTSNode("Grandchild 1")
        grandchild2 = MCTSNode("Grandchild 2")
        grandchild3 = MCTSNode("Grandchild 3")
        child1.add_child(grandchild1)
        child1.add_child(grandchild2)
        child2.add_child(grandchild3)

        great_grandchild = MCTSNode("Great grandchild")
        grandchild1.add_child(great_grandchild)

        assert TreeOperations._count_descendants(root) == 6
        assert TreeOperations._count_descendants(child1) == 3
        assert TreeOperations._count_descendants(child2) == 1
        assert TreeOperations._count_descendants(grandchild1) == 1
        assert TreeOperations._count_descendants(great_grandchild) == 0

    def test_get_tree_depths_single_node(self):
        """Test getting tree depths for single node."""
        node = MCTSNode("Node")
        depths = TreeOperations.get_tree_depths(node)

        assert depths == [0]

    def test_get_tree_depths_linear_path(self):
        """Test getting tree depths for linear path."""
        root = MCTSNode("Root")
        child = MCTSNode("Child")
        grandchild = MCTSNode("Grandchild")
        great_grandchild = MCTSNode("Great grandchild")

        root.add_child(child)
        child.add_child(grandchild)
        grandchild.add_child(great_grandchild)

        depths = TreeOperations.get_tree_depths(root)

        assert depths == [3]

    def test_get_tree_depths_branching_tree(self):
        """Test getting tree depths for branching tree."""
        root = MCTSNode("Root")

        branch1 = MCTSNode("Branch 1")
        branch2 = MCTSNode("Branch 2")
        root.add_child(branch1)
        root.add_child(branch2)

        branch1_child = MCTSNode("Branch 1 child")
        branch1.add_child(branch1_child)

        branch1_grandchild = MCTSNode("Branch 1 grandchild")
        branch1_child.add_child(branch1_grandchild)

        branch2_child = MCTSNode("Branch 2 child")
        branch2.add_child(branch2_child)

        depths = TreeOperations.get_tree_depths(root)

        assert sorted(depths) == [2, 3]

    def test_calculate_average_depth_single_root(self):
        """Test calculating average depth for single root."""
        root = MCTSNode("Root")
        child1 = MCTSNode("Child 1")
        child2 = MCTSNode("Child 2")
        grandchild = MCTSNode("Grandchild")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        avg_depth = TreeOperations.calculate_average_depth([root])

        assert avg_depth == 1.5

    def test_calculate_average_depth_multiple_roots(self):
        """Test calculating average depth for multiple roots."""
        roots = []

        root1 = MCTSNode("Root 1")
        child1 = MCTSNode("Child 1")
        grandchild1 = MCTSNode("Grandchild 1")
        root1.add_child(child1)
        child1.add_child(grandchild1)
        roots.append(root1)

        root2 = MCTSNode("Root 2")
        child2 = MCTSNode("Child 2")
        root2.add_child(child2)
        roots.append(root2)

        root3 = MCTSNode("Root 3")
        roots.append(root3)

        avg_depth = TreeOperations.calculate_average_depth(roots)

        assert avg_depth == 1.0

    def test_calculate_average_depth_empty_roots(self):
        """Test calculating average depth with empty roots list."""
        avg_depth = TreeOperations.calculate_average_depth([])
        assert avg_depth == 0.0

    def test_prune_branches_preserves_good_subtrees(self):
        """Test that pruning preserves entire good subtrees."""
        root = MCTSNode("Root")
        root.visits = 10
        root.avg_score = 0.8

        good_child = MCTSNode("Good child")
        good_child.visits = 8
        good_child.avg_score = 0.85
        root.add_child(good_child)

        for i in range(3):
            descendant = MCTSNode(f"Good descendant {i}")
            descendant.visits = 5
            descendant.avg_score = 0.8
            good_child.add_child(descendant)

        bad_child = MCTSNode("Bad child")
        bad_child.visits = 8
        bad_child.avg_score = 0.4
        root.add_child(bad_child)

        initial_good_descendants = len(good_child.children)

        pruned_count = TreeOperations.prune_branches([root])

        assert pruned_count == 1
        assert len(root.children) == 1
        assert root.children[0] == good_child
        assert len(good_child.children) == initial_good_descendants

    def test_backpropagate_none_parent(self):
        """Test backpropagation stops at node with no parent."""
        node = MCTSNode("Orphan node")

        TreeOperations.backpropagate(node, 0.5)

        assert node.visits == 1
        assert node.avg_score == 0.5

    def test_prune_branches_unvisited_children(self):
        """Test that unvisited children are not pruned."""
        root = MCTSNode("Root")
        root.visits = 10
        root.avg_score = 0.8

        visited_child = MCTSNode("Visited")
        visited_child.visits = 5
        visited_child.avg_score = 0.3

        unvisited_child = MCTSNode("Unvisited")
        unvisited_child.visits = 0

        root.add_child(visited_child)
        root.add_child(unvisited_child)

        pruned_count = TreeOperations.prune_branches([root])

        assert pruned_count == 1
        assert len(root.children) == 1
        assert root.children[0] == unvisited_child
