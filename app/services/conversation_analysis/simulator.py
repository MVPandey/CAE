from typing import Any, Optional

from ...schema.llm.message import Message
from ...services.llm_service import LLMService
from ...utils.logger import logger
from .config import ResponseConfig


class ConversationSimulator:
    """Handles conversation simulation for MCTS evaluation"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def simulate_conversation(
        self, messages: list[Message], depth: int, goal: Optional[str], max_tokens: int
    ) -> dict[str, Any]:
        prompt = self._build_simulation_prompt(depth, goal)

        try:
            result = await self.llm_service.query_llm(
                messages=[prompt] + messages,
                json_response=True,
                max_tokens=max_tokens * ResponseConfig.TOKEN_MULTIPLIER_SIMULATION,
            )

            return {
                "simulation": result.get("simulation", []),
                "user_reactions": result.get("user_reactions", []),
            }

        except Exception:
            logger.error("Failed to simulate conversation", exc_info=True)
            return {"simulation": [], "user_reactions": []}

    def _build_simulation_prompt(self, depth: int, goal: Optional[str]) -> Message:
        goal_section = f"<conversation_goal>{goal}</conversation_goal>\n" if goal else ""

        return Message(
            role="system",
            content=f"""Simulate realistic conversation continuation.
{goal_section}
Generate {depth} back-and-forth exchanges.

Return JSON:
{{
    "simulation": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
    ],
    "user_reactions": ["User emotional state after each exchange"]
}}""",
        )
