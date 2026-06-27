import logging
import time
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
        t_start = time.perf_counter()

        t_intent_start = time.perf_counter()
        resolved_intent = (
            intent if intent is not None else self._intent_detector.detect(query)
        )
        t_intent_elapsed = time.perf_counter() - t_intent_start
        logger.debug(
            "query='%s' intent=%s intent_detection_time_ms=%.2f",
            query,
            resolved_intent.value,
            t_intent_elapsed * 1000,
        )

        t_embed_start = time.perf_counter()
        embedding = self._embedder.embed(query)
        t_embed_elapsed = time.perf_counter() - t_embed_start
        logger.debug(
            "embedding len=%d embedding_time_ms=%.2f",
            len(embedding),
            t_embed_elapsed * 1000,
        )

        t_retrieve_start = time.perf_counter()
        if required_ingredients:
            results = self._retriever.retrieve_with_ingredients(
                embedding, resolved_intent, top_k, required_ingredients
            )
        else:
            results = self._retriever.retrieve(embedding, resolved_intent, top_k)
        t_retrieve_elapsed = time.perf_counter() - t_retrieve_start
        logger.debug(
            "results count=%d retrieve_time_ms=%.2f",
            len(results),
            t_retrieve_elapsed * 1000,
        )

        t_total = time.perf_counter() - t_start
        logger.debug("search_total_time_ms=%.2f", t_total * 1000)
        return results
