from __future__ import annotations

import math


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""

    def __init__(self, response: str, parent: "MCTSNode" | None = None, index: int = 0):
        self.response = response
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.index = index

        self.visits = 0
        self.total_score = 0.0
        self.avg_score = 0.0

        self.simulated_reactions: list[str] = []
        self.sub_history: list[dict[str, str]] = []
        self.general_metrics: dict[str, float] = {}
        self.goal_metrics: dict[str, float] = {}

    def add_child(self, child: "MCTSNode") -> None:
        child.parent = self
        child.index = len(self.children)
        self.children.append(child)

    def is_fully_expanded(self, max_children: int = 3) -> bool:
        return len(self.children) >= max_children

    def best_child(self, exploration_constant: float = 1.414) -> "MCTSNode":
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children, key=lambda c: self._ucb1_score(c, exploration_constant))

    def update(self, score: float) -> None:
        self.visits += 1
        self.total_score += score
        self.avg_score = self.total_score / self.visits

    def _ucb1_score(self, child: "MCTSNode", exploration_constant: float) -> float:
        if child.visits == 0:
            return float("inf")

        exploitation = child.avg_score
        exploration = exploration_constant * math.sqrt(2 * math.log(self.visits) / child.visits)
        return exploitation + exploration
