import logging
from typing import List, Optional

from app.core.intent import Intent
from app.core.interfaces import Embedder, IntentDetector, RecipeRetriever
from app.core.models import Recipe

logger = logging.getLogger(__name__)


class RecipeSearch:
    """
    Deep module that orchestrates recipe search:
    embed query -> detect intent -> retrieve recipes.

    The public interface is a single method:
        search(query, top_k, required_ingredients) -> List[Recipe]

    Intent detection is an internal step; callers that already know
    the intent (e.g. API endpoints) may pass it via the optional
    *intent* parameter.
    """

    def __init__(
        self,
        embedder: Embedder,
        intent_detector: IntentDetector,
        retriever: RecipeRetriever,
    ):
        self._embedder = embedder
        self._intent_detector = intent_detector
        self._retriever = retriever

    def search(
        self,
        query: str,
        intent: Optional[Intent] = None,
        top_k: int = 5,
        required_ingredients: Optional[List[str]] = None,
    ) -> List[Recipe]:
        resolved_intent = (
            intent if intent is not None else self._intent_detector.detect(query)
        )
        logger.debug("query='%s' intent=%s", query, resolved_intent.value)
        embedding = self._embedder.embed(query)
        logger.debug("embedding len=%d", len(embedding))

        if required_ingredients:
            results = self._retriever.retrieve_with_ingredients(
                embedding, resolved_intent, top_k, required_ingredients
            )
        else:
            results = self._retriever.retrieve(embedding, resolved_intent, top_k)

        logger.debug("results count=%d", len(results))
        return results
