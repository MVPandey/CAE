import time
from typing import Any

from ..db.chat import get_chat_history, create_conversation_analysis
from ..schema.llm.message import Message
from ..schema.conversation_analysis import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
)
from ..services.llm_service import LLMService
from ..utils.config import app_settings
from ..utils.logger import logger
from ..utils.exceptions import ChatHistoryNotFoundError
from .mcts import MCTSAlgorithm
from .conversation_analysis import (
    ConversationAnalyzer,
    ResponseGenerator,
    ConversationScorer,
    ConversationSimulator,
)


class ConversationAnalysisService:
    """Service for analyzing conversations using MCTS to find optimal response paths"""

    def __init__(self):
        self.llm_service = LLMService(
            base_url=app_settings.LLM_API_BASE_URL,
            api_key=app_settings.LLM_API_KEY,
            model_name=app_settings.LLM_MODEL_NAME,
        )

        self.response_generator = ResponseGenerator(self.llm_service)
        self.simulator = ConversationSimulator(self.llm_service)
        self.scorer = ConversationScorer(self.llm_service)
        self.analyzer = ConversationAnalyzer(self.llm_service)

        self.mcts = MCTSAlgorithm(self.response_generator, self.simulator, self.scorer)

    async def analyze_conversation(
        self, request: ConversationAnalysisRequest
    ) -> ConversationAnalysisResponse:
        start_time = time.time()

        logger.info(
            "Starting conversation analysis",
            extra={"chat_id": str(request.chat_id), "goal": request.conversation_goal},
        )

        history = await get_chat_history(request.chat_id)
        if not history:
            raise ChatHistoryNotFoundError(str(request.chat_id))

        messages = [
            Message(role=msg.role.value, content=msg.content) for msg in history
        ]

        initial_responses = await self.response_generator.generate_initial_branches(
            messages,
            request.num_branches,
            request.conversation_goal,
            request.max_tokens,
        )

        mcts_config = {
            "iterations": request.mcts_iterations,
            "simulation_depth": request.simulation_depth,
            "exploration_constant": request.exploration_constant,
            "goal": request.conversation_goal,
            "max_tokens": request.max_tokens,
        }

        root_nodes, mcts_stats = await self.mcts.run(
            messages, initial_responses, mcts_config
        )

        best_node, best_idx, analysis = await self.analyzer.analyze_best_path(
            root_nodes, messages, request.conversation_goal, request.max_tokens
        )

        branches = self.analyzer.convert_to_branches(root_nodes)

        scores = {
            "best_score": best_node.avg_score,
            "average_score": sum(node.avg_score for node in root_nodes)
            / len(root_nodes),
            "score_variance": self._calculate_variance(
                [node.avg_score for node in root_nodes]
            ),
        }

        db_result = await create_conversation_analysis(
            chat_id=request.chat_id,
            conversation_goal=request.conversation_goal,
            branches=[self._branch_to_dict(b) for b in branches],
            selected_branch_index=best_idx,
            selected_response=best_node.response,
            analysis=analysis,
            scores=scores,
            mcts_statistics=mcts_stats,
        )

        logger.info(
            f"Analysis completed in {time.time() - start_time:.2f}s",
            extra={"chat_id": str(request.chat_id), "best_score": best_node.avg_score},
        )

        return ConversationAnalysisResponse(
            id=db_result["id"],
            chat_id=db_result["chat_id"],
            created_at=db_result["created_at"],
            conversation_goal=request.conversation_goal,
            branches=branches,
            selected_branch_index=best_idx,
            selected_response=best_node.response,
            analysis=analysis,
            overall_scores=scores,
            mcts_statistics=mcts_stats,
        )

    def _calculate_variance(self, values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _branch_to_dict(self, branch) -> dict[str, Any]:
        return {
            "response": branch.response,
            "simulated_user_reactions": branch.simulated_user_reactions,
            "score": branch.score,
            "sub_history": branch.sub_history,
            "general_metrics": branch.general_metrics,
            "goal_metrics": branch.goal_metrics,
            "visits": branch.visits,
        }
