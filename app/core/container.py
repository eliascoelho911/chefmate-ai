from dataclasses import dataclass
from typing import Optional

from app.core.interfaces import Embedder
from app.utils.chat_orchestrator import ChatOrchestrator
from app.utils.ingredient_translator import IngredientTranslator
from app.utils.ingredient_search_service import IngredientSearchService
from app.utils.recipe_search import RecipeSearch
from app.utils.llm_model import LLMRunner
from app.utils.recipe_translator import RecipeTranslator
from app.utils.intent_detector import IntentDetector
from app.utils.recipe_repository import RecipeRepository
from app.utils.vector_index import VectorIndex


@dataclass
class AppContainer:
    """
    Concrete container holding all application dependencies.
    Replaces the untyped GlobalState singleton with a typed, injectable seam.
    """

    config: dict
    recipe_search: RecipeSearch
    llm_runner: LLMRunner
    intent_detector: IntentDetector
    embedder: Embedder
    vector_index: VectorIndex
    recipe_repository: RecipeRepository
    chat_orchestrator: ChatOrchestrator
    ingredient_translator: IngredientTranslator
    recipe_translator: RecipeTranslator
    ingredient_search_service: IngredientSearchService


# Module-level singleton. Set once at startup; read-only thereafter.
_container: Optional[AppContainer] = None


def set_container(container: AppContainer) -> None:
    """Set the global application container. Called exactly once during startup."""
    global _container
    _container = container


def get_container() -> AppContainer:
    """
    Return the application container for use with FastAPI Depends.
    Raises RuntimeError if startup has not completed.
    """
    if _container is None:
        raise RuntimeError(
            "Container not initialized. Call init_dependencies() before handling requests."
        )
    return _container
