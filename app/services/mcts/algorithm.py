import asyncio
from typing import Any

from .node import MCTSNode
from .tree_operations import TreeOperations
from .adaptive_exploration import AdaptiveExplorationStrategy
from ..conversation_analysis.response_generator import ResponseGenerator
from ..conversation_analysis.simulator import ConversationSimulator
from ..conversation_analysis.scorer import ConversationScorer
from ..conversation_analysis.config import MCTSConfig, EnhancedMCTSConfig
from ...schema.llm.message import Message
from ...utils.logger import logger


class MCTSAlgorithm:
    """Core MCTS algorithm implementation"""

    def __init__(
        self,
        response_generator: ResponseGenerator,
        simulator: ConversationSimulator,
        scorer: ConversationScorer,
    ):
        self.response_generator = response_generator
        self.simulator = simulator
        self.scorer = scorer
        self.tree_ops = TreeOperations()
        self.exploration_strategy = None  # Will be initialized when config is available

    async def run(
        self,
        base_messages: list[Message],
        initial_responses: list[str],
        config: dict[str, Any],
    ) -> tuple[list[MCTSNode], dict[str, Any]]:
        root_nodes = [
            MCTSNode(response, index=i) for i, response in enumerate(initial_responses)
        ]

        # Initialize exploration strategy if enhanced config is available
        enhanced_config = config.get("enhanced_config")
        if enhanced_config and isinstance(enhanced_config, EnhancedMCTSConfig):
            self.exploration_strategy = AdaptiveExplorationStrategy(enhanced_config)
            logger.info("Using adaptive exploration strategy")
        else:
            self.exploration_strategy = None
            logger.info("Using fixed exploration constant")

        stats = {
            "total_iterations": config["iterations"],
            "nodes_created": len(root_nodes),
            "nodes_evaluated": 0,
            "pruned_branches": 0,
            "parallel_evaluations": 0,
            "average_depth_explored": 0,
        }

        for iteration in range(config["iterations"]):
            # Calculate exploration constant for this iteration
            if self.exploration_strategy:
                exploration_constant = self.exploration_strategy.get_exploration_constant(
                    iteration, config["iterations"]
                )
            else:
                exploration_constant = config["exploration_constant"]

            nodes_to_process = [
                (root, await self._select_node(root, exploration_constant))
                for root in root_nodes
            ]

            tasks = [
                self._expand_and_simulate(base_messages, node, config)
                for _, node in nodes_to_process
            ]

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

        stats["average_depth_explored"] = self.tree_ops.calculate_average_depth(
            root_nodes
        )

        return root_nodes, stats

    async def _select_node(
        self, root: MCTSNode, exploration_constant: float
    ) -> MCTSNode:
        node = root
        while node.children and node.is_fully_expanded():
            node = node.best_child(exploration_constant)
        return node

    async def _expand_and_simulate(
        self, base_messages: list[Message], node: MCTSNode, config: dict[str, Any]
    ) -> tuple[float, list[MCTSNode]]:
        new_children = []

        if not node.is_fully_expanded() and node.visits > 0:
            extended_messages = self._build_conversation_path(base_messages, node)
            existing_responses = [child.response for child in node.children]

            new_response = await self.response_generator.generate_expansion_response(
                extended_messages,
                existing_responses,
                config.get("goal"),
                config["max_tokens"],
            )

            if new_response:
                new_children.append(MCTSNode(new_response))

        extended_messages = self._build_conversation_path(base_messages, node)

        simulation_data = await self.simulator.simulate_conversation(
            extended_messages,
            config["simulation_depth"],
            config.get("goal"),
            config["max_tokens"],
        )

        node.sub_history = simulation_data["simulation"]
        node.simulated_reactions = simulation_data["user_reactions"]

        extended_sim_messages = extended_messages + [
            Message(**msg) for msg in node.sub_history
        ]

        score_data = await self.scorer.score_simulation(
            extended_sim_messages,
            simulation_data,
            config.get("goal"),
            config["max_tokens"],
        )

        node.general_metrics = score_data["general_metrics"]
        node.goal_metrics = score_data.get("goal_metrics", {})

        return score_data["overall_score"], new_children

    def _build_conversation_path(
        self, base_messages: list[Message], node: MCTSNode
    ) -> list[Message]:
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
