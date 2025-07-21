import json
from typing import Optional

from ...schema.llm.message import Message
from ...services.llm_service import LLMService
from ...utils.logger import logger
from .config import ResponseConfig


class ResponseGenerator:
    """Handles generation of conversation responses"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def generate_initial_branches(
        self,
        messages: list[Message],
        num_branches: int,
        goal: Optional[str],
        max_tokens: int,
    ) -> list[str]:
        prompt = self._build_initial_branches_prompt(num_branches, goal)

        try:
            result = await self.llm_service.query_llm(
                messages=[prompt] + messages,
                json_response=True,
                max_tokens=max_tokens * ResponseConfig.TOKEN_MULTIPLIER_INITIAL,
            )

            if not isinstance(result, dict) or "responses" not in result:
                logger.error("Invalid response format", extra={"result": result})
                return ResponseConfig.DEFAULT_RESPONSES[:num_branches]

            return result["responses"]

        except Exception:
            logger.error("Failed to generate initial branches", exc_info=True)
            return ResponseConfig.DEFAULT_RESPONSES[:num_branches]

    async def generate_expansion_response(
        self,
        messages: list[Message],
        existing_responses: list[str],
        goal: Optional[str],
        max_tokens: int,
    ) -> Optional[str]:
        prompt = self._build_expansion_prompt(existing_responses, goal)

        try:
            result = await self.llm_service.query_llm(
                messages=[prompt] + messages, json_response=True, max_tokens=max_tokens
            )

            if isinstance(result, dict) and "response" in result:
                return result["response"]

            logger.error("Invalid expansion response format")
            return None

        except Exception:
            logger.error("Failed to generate expansion response", exc_info=True)
            return None

    def _build_initial_branches_prompt(self, num_branches: int, goal: Optional[str]) -> Message:
        goal_section = f"\n<conversation_goal>\nThe user wants to: {goal}\n</conversation_goal>" if goal else ""

        return Message(
            role="system",
            content=f"""Generate {num_branches} diverse responses to continue this conversation.
{goal_section}

Return JSON:
{{
    "responses": ["First response...", "Second response...", ...]
}}""",
        )

    def _build_expansion_prompt(self, existing_responses: list[str], goal: Optional[str]) -> Message:
        goal_section = f"<goal>Help achieve: {goal}</goal>\n" if goal else ""

        return Message(
            role="system",
            content=f"""Generate ONE new response different from existing ones.
{goal_section}
<previous_responses>
{json.dumps(existing_responses)}
</previous_responses>

Return JSON:
{{"response": "Your new response here"}}""",
        )
