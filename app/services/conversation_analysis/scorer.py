import json
from typing import Any, Optional

from ...schema.llm.message import Message
from ...services.llm_service import LLMService
from ...utils.logger import logger
from .config import ScoringConfig


class ConversationScorer:
    """Handles scoring and evaluation of conversations"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def score_simulation(
        self,
        messages: list[Message],
        simulation_data: dict[str, Any],
        goal: Optional[str],
        max_tokens: int,
    ) -> dict[str, Any]:
        prompt = self._build_scoring_prompt(simulation_data, goal)

        try:
            result = await self.llm_service.query_llm(
                messages=[prompt] + messages, json_response=True, max_tokens=max_tokens
            )

            return self._validate_scoring_result(result)

        except Exception:
            logger.error("Failed to score simulation", exc_info=True)
            return self._get_default_scores()

    def _build_scoring_prompt(self, simulation_data: dict[str, Any], goal: Optional[str]) -> Message:
        goal_section = ""
        if goal:
            goal_section = f"""
<goal_specific_scoring>
Conversation goal: {goal}
Score 3-5 metrics specific to achieving this goal (0.0-1.0).
</goal_specific_scoring>"""

        return Message(
            role="system",
            content=f"""Score this conversation based on quality metrics.

General metrics (0.0-1.0):
- clarity: How clear and understandable
- relevance: How well responses address context
- engagement: Likelihood to maintain interest
- authenticity: How genuine and natural
- coherence: Logical flow
- respectfulness: Appropriate tone
{goal_section}

<simulation_data>
{json.dumps(simulation_data)}
</simulation_data>

Return JSON:
{{
    "general_metrics": {{"clarity": 0.85, ...}},
    "goal_metrics": {{"metric": 0.8, ...}},
    "overall_score": 0.87,
    "reasoning": "Brief explanation"
}}""",
        )

    def _validate_scoring_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if "general_metrics" not in result:
            result["general_metrics"] = {}

        for metric in ScoringConfig.GENERAL_METRICS:
            if metric not in result["general_metrics"]:
                result["general_metrics"][metric] = 0.0

        if "goal_metrics" not in result:
            result["goal_metrics"] = {}

        if "overall_score" not in result:
            scores = result["general_metrics"].values()
            result["overall_score"] = sum(scores) / len(scores) if scores else 0.0

        return result

    def _get_default_scores(self) -> dict[str, Any]:
        return {
            "general_metrics": {metric: 0.5 for metric in ScoringConfig.GENERAL_METRICS},
            "goal_metrics": {},
            "overall_score": 0.5,
        }
