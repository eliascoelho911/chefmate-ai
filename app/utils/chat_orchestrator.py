import logging
from typing import Iterator

from app.core.interfaces import LLMRunner, PromptBuilder
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
        llm_runner: LLMRunner,
        prompt_builder: PromptBuilder,
    ):
        self._recipe_search = recipe_search
        self._llm_runner = llm_runner
        self._prompt_builder = prompt_builder

    def chat(self, chat_history: ChatHistory) -> Iterator[str]:
        # Validation
        if not chat_history.messages:
            raise ValueError("Chat history cannot be empty")

        latest_user_message = chat_history.latest_user_message
        if not latest_user_message:
            raise ValueError("No user message found in chat history")

        logger.debug("latest_user_message='%s'", latest_user_message)

        # Detect intent
        intent = self._recipe_search.detect_intent(latest_user_message)
        logger.debug("detected intent='%s'", intent)

        # Retrieve recipes
        recipes = self._recipe_search.search(
            latest_user_message, intent=intent, top_k=3
        )
        logger.debug("retrieved recipes count=%d", len(recipes))

        # Build messages
        messages = self._prompt_builder.build_messages(chat_history, intent, recipes)
        logger.debug("built messages count=%d", len(messages))

        # Stream response
        yield from self._llm_runner.stream_response(messages)
