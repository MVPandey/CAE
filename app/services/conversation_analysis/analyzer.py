from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ...schema.conversation_analysis import ConversationBranch
from ...schema.llm.message import Message
from ...services.llm_service import LLMService
from ...utils.logger import logger
from .config import ResponseConfig, ScoringConfig

if TYPE_CHECKING:
    from ...services.mcts.node import MCTSNode


class ConversationAnalyzer:
    """Analyzes conversation paths and selects optimal responses"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def analyze_best_path(
        self,
        root_nodes: list["MCTSNode"],
        original_messages: list[Message],
        goal: str | None,
        max_tokens: int,
    ) -> tuple["MCTSNode", int, str]:
        best_root = self._select_best_node(root_nodes)
        best_idx = root_nodes.index(best_root)

        analysis = await self._generate_analysis(best_root, root_nodes, original_messages, goal, max_tokens)

        return best_root, best_idx, analysis

    def convert_to_branches(self, root_nodes: list["MCTSNode"]) -> list[ConversationBranch]:
        return [
            ConversationBranch(
                response=node.response,
                simulated_user_reactions=node.simulated_reactions,
                score=node.avg_score,
                sub_history=node.sub_history,
                general_metrics=node.general_metrics,
                goal_metrics=node.goal_metrics,
                visits=node.visits,
                parent_index=None,
                children_indices=[child.index for child in node.children],
            )
            for node in root_nodes
        ]

    def _select_best_node(self, root_nodes: list["MCTSNode"]) -> "MCTSNode":
        total_visits = sum(node.visits for node in root_nodes)

        return max(
            root_nodes,
            key=lambda n: (
                n.avg_score * ScoringConfig.SCORE_WEIGHT_QUALITY
                + (n.visits / total_visits if total_visits > 0 else 0) * ScoringConfig.SCORE_WEIGHT_VISITS
            ),
        )

    async def _generate_analysis(
        self,
        best_node: "MCTSNode",
        all_nodes: list["MCTSNode"],
        messages: list[Message],
        goal: str | None,
        max_tokens: int,
    ) -> str:
        prompt = self._build_analysis_prompt(best_node, all_nodes, goal)

        try:
            response = await self.llm_service.query_llm(
                messages=[prompt] + messages,
                json_response=False,
                max_tokens=max_tokens * ResponseConfig.TOKEN_MULTIPLIER_ANALYSIS,
            )
            return response.content

        except Exception:
            logger.error("Failed to generate analysis", exc_info=True)
            return self._get_default_analysis(best_node, all_nodes.index(best_node))

    def _build_analysis_prompt(self, best_node: "MCTSNode", all_nodes: list["MCTSNode"], goal: str | None) -> Message:
        goal_section = f"<conversation_goal>{goal}</conversation_goal>\n" if goal else ""

        options_data = [
            {
                "response": node.response[:100] + "...",
                "score": node.avg_score,
                "visits": node.visits,
                "key_strength": max(node.general_metrics.items(), key=lambda x: x[1], default=("", 0)),
            }
            for node in all_nodes
        ]

        return Message(
            role="system",
            content=f"""Analyze why the selected response is optimal.
{goal_section}
<selected_response>
Response: {best_node.response}
Score: {best_node.avg_score:.3f}
Visits: {best_node.visits}
</selected_response>

<all_options>
{json.dumps(options_data, indent=2)}
</all_options>

Provide 2-3 paragraph analysis covering:
- Why this response best serves the goals
- Key strengths based on metrics
- Comparison to alternatives
- Potential considerations""",
        )

    def _get_default_analysis(self, best_node: "MCTSNode", index: int) -> str:
        return (
            f"Selected response {index + 1} based on MCTS evaluation. "
            f"This response achieved a score of {best_node.avg_score:.2f} "
            f"across {best_node.visits} simulations."
        )
