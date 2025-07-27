import asyncio
from typing import Any

from ...schema.llm.message import Message
from ...utils.logger import logger
from ..cache.semantic_cache import semantic_cache
from ..conversation_analysis.config import MCTSConfig
from ..conversation_analysis.response_generator import ResponseGenerator
from ..conversation_analysis.scorer import ConversationScorer
from ..conversation_analysis.simulator import ConversationSimulator
from .node import MCTSNode
from .tree_operations import TreeOperations


class MCTSAlgorithm:
    """Core MCTS algorithm implementation with semantic caching"""

    def __init__(
        self,
        response_generator: ResponseGenerator,
        simulator: ConversationSimulator,
        scorer: ConversationScorer,
        use_cache: bool = True,
    ):
        self.response_generator = response_generator
        self.simulator = simulator
        self.scorer = scorer
        self.tree_ops = TreeOperations()
        self.use_cache = use_cache
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }

    async def run(
        self,
        base_messages: list[Message],
        initial_responses: list[str],
        config: dict[str, Any],
    ) -> tuple[list[MCTSNode], dict[str, Any]]:
        root_nodes = [MCTSNode(response, index=i) for i, response in enumerate(initial_responses)]

        stats = {
            "total_iterations": config["iterations"],
            "nodes_created": len(root_nodes),
            "nodes_evaluated": 0,
            "pruned_branches": 0,
            "parallel_evaluations": 0,
            "average_depth_explored": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_stores": 0,
        }

        for iteration in range(config["iterations"]):
            nodes_to_process = [
                (root, await self._select_node(root, config["exploration_constant"])) for root in root_nodes
            ]

            tasks = [self._expand_and_simulate(base_messages, node, config) for _, node in nodes_to_process]

            results = await asyncio.gather(*tasks)
            stats["parallel_evaluations"] += len(tasks)

            for (root, node), (score, new_children) in zip(nodes_to_process, results):
                for child in new_children:
                    node.add_child(child)
                    stats["nodes_created"] += 1

                self.tree_ops.backpropagate(node, score)
                stats["nodes_evaluated"] += 1

            if iteration > 0 and iteration % MCTSConfig.PRUNING_INTERVAL == 0:
                pruned = self.tree_ops.prune_branches(root_nodes)
                stats["pruned_branches"] += pruned

        stats["average_depth_explored"] = self.tree_ops.calculate_average_depth(root_nodes)

        stats["cache_hits"] = self._cache_stats["hits"]
        stats["cache_misses"] = self._cache_stats["misses"]
        stats["cache_stores"] = self._cache_stats["stores"]

        if stats["nodes_evaluated"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        else:
            stats["cache_hit_rate"] = 0.0

        logger.info(
            "MCTS run completed",
            extra={
                "nodes_evaluated": stats["nodes_evaluated"],
                "cache_hit_rate": stats["cache_hit_rate"],
                "cache_hits": stats["cache_hits"],
            },
        )

        return root_nodes, stats

    async def _select_node(self, root: MCTSNode, exploration_constant: float) -> MCTSNode:
        node = root
        while node.children and node.is_fully_expanded():
            node = node.best_child(exploration_constant)
        return node

    async def _expand_and_simulate(
        self, base_messages: list[Message], node: MCTSNode, config: dict[str, Any]
    ) -> tuple[float, list[MCTSNode]]:
        new_children = []
        extended_messages = self._build_conversation_path(base_messages, node)

        if self.use_cache:
            cache_entry = await semantic_cache.get(extended_messages)
            if cache_entry:
                self._cache_stats["hits"] += 1
                logger.info(
                    "Cache hit for MCTS node",
                    extra={
                        "node_depth": self._get_node_depth(node),
                        "similarity": cache_entry.metadata.get("similarity", 1.0),
                    },
                )

                node.sub_history = cache_entry.simulation_data.get("simulation", [])
                node.simulated_reactions = cache_entry.simulation_data.get("user_reactions", [])
                node.general_metrics = cache_entry.score_data.get("general_metrics", {})
                node.goal_metrics = cache_entry.score_data.get("goal_metrics", {})

                return cache_entry.score_data.get("overall_score", 0.5), new_children
            else:
                self._cache_stats["misses"] += 1

        if not node.is_fully_expanded() and node.visits > 0:
            existing_responses = [child.response for child in node.children]

            new_response = await self.response_generator.generate_expansion_response(
                extended_messages,
                existing_responses,
                config.get("goal"),
                config["max_tokens"],
            )

            if new_response:
                new_children.append(MCTSNode(new_response))

        simulation_data = await self.simulator.simulate_conversation(
            extended_messages,
            config["simulation_depth"],
            config.get("goal"),
            config["max_tokens"],
        )

        node.sub_history = simulation_data["simulation"]
        node.simulated_reactions = simulation_data["user_reactions"]

        extended_sim_messages = extended_messages + [Message(**msg) for msg in node.sub_history]

        score_data = await self.scorer.score_simulation(
            extended_sim_messages,
            simulation_data,
            config.get("goal"),
            config["max_tokens"],
        )

        node.general_metrics = score_data["general_metrics"]
        node.goal_metrics = score_data.get("goal_metrics", {})

        if self.use_cache and node.response:  # Don't cache root node
            success = await semantic_cache.store(
                extended_messages,
                node.response,
                simulation_data,
                score_data,
                {
                    "node_depth": self._get_node_depth(node),
                    "goal": config.get("goal"),
                    "iteration": config.get("current_iteration", 0),
                },
            )
            if success:
                self._cache_stats["stores"] += 1

        return score_data["overall_score"], new_children

    def _build_conversation_path(self, base_messages: list[Message], node: MCTSNode) -> list[Message]:
        path = []
        current = node

        while current:
            path.append(current.response)
            current = current.parent

        path.reverse()
        path = path[1:]  # Remove empty root

        result = base_messages.copy()
        for response in path:
            result.append(Message(role="assistant", content=response))

        return result

    def _get_node_depth(self, node: MCTSNode) -> int:
        """Get the depth of a node in the tree."""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_lookups = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_lookups if total_lookups > 0 else 0

        return {
            **self._cache_stats,
            "total_lookups": total_lookups,
            "hit_rate": hit_rate,
        }
