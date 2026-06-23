import logging
from typing import List
from app.core.interfaces import Embedder, IntentDetector, RecipeRetriever
from app.core.models import Recipe

logger = logging.getLogger(__name__)


class RecipeSearch:
    """
    Deep module that orchestrates recipe search:
    embed query -> detect intent -> retrieve recipes.
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

    def search(self, query: str, top_k: int = 5) -> List[Recipe]:
        intent = self._intent_detector.detect(query)
        logger.debug("query='%s' intent='%s'", query, intent)
        embedding = self._embedder.embed(query)
        logger.debug("embedding len=%d", len(embedding))
        results = self._retriever.retrieve(embedding, intent, top_k)
        logger.debug("results count=%d", len(results))
        return results
