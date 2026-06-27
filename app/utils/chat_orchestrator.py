import logging
import time
from typing import Iterator

from app.core.interfaces import IntentDetector, LLMRunner, PromptBuilder
from app.core.models import ChatHistory
from app.utils.recipe_search import RecipeSearch

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Deep module that orchestrates a conversational turn:
    validate -> detect intent -> search recipes -> build prompt -> stream LLM tokens.
    """

    def __init__(
        self,
        recipe_search: RecipeSearch,
        intent_detector: IntentDetector,
        llm_runner: LLMRunner,
        prompt_builder: PromptBuilder,
    ):
        self._recipe_search = recipe_search
        self._intent_detector = intent_detector
        self._llm_runner = llm_runner
        self._prompt_builder = prompt_builder

    def chat(self, chat_history: ChatHistory) -> Iterator[str]:
        t_start = time.perf_counter()

        # Validation
        if not chat_history.messages:
            raise ValueError("Chat history cannot be empty")

        latest_user_message = chat_history.latest_user_message
        if not latest_user_message:
            raise ValueError("No user message found in chat history")

        logger.debug("latest_user_message='%s'", latest_user_message)

        # Detect intent
        t_intent_start = time.perf_counter()
        intent = self._intent_detector.detect(latest_user_message)
        t_intent_elapsed = time.perf_counter() - t_intent_start
        logger.debug(
            "detected intent=%s intent_detection_time_ms=%.2f",
            intent.value,
            t_intent_elapsed * 1000,
        )

        # Retrieve recipes
        t_search_start = time.perf_counter()
        recipes = self._recipe_search.search(
            latest_user_message, intent=intent, top_k=3
        )
        t_search_elapsed = time.perf_counter() - t_search_start
        logger.debug(
            "retrieved recipes count=%d recipe_search_time_ms=%.2f",
            len(recipes),
            t_search_elapsed * 1000,
        )

        # Build messages
        t_prompt_start = time.perf_counter()
        messages = self._prompt_builder.build_messages(chat_history, intent, recipes)
        t_prompt_elapsed = time.perf_counter() - t_prompt_start
        logger.debug(
            "built messages count=%d prompt_build_time_ms=%.2f",
            len(messages),
            t_prompt_elapsed * 1000,
        )

        # Stream response
        t_llm_start = time.perf_counter()
        yield from self._llm_runner.stream_response(messages)
        t_llm_elapsed = time.perf_counter() - t_llm_start
        logger.debug("llm_stream_time_ms=%.2f", t_llm_elapsed * 1000)

        t_total = time.perf_counter() - t_start
        logger.debug("chat_turn_total_time_ms=%.2f", t_total * 1000)
