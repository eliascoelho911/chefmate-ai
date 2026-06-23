import logging
from typing import List
import numpy as np
from app.core.interfaces import RecipeRetriever
from app.core.models import Recipe
from app.utils.faiss_handler import FAISSHandler

logger = logging.getLogger(__name__)


class FAISSRecipeRetriever:
    """
    Adapter that satisfies the RecipeRetriever interface using FAISS + SQLite.
    """

    def __init__(self, faiss_handler: FAISSHandler):
        self._faiss_handler = faiss_handler

    def retrieve(
        self, query_embedding: List[float], intent: str, top_k: int
    ) -> List[Recipe]:
        embedding_array = np.array(query_embedding)
        results = self._faiss_handler.search_by_intent(embedding_array, intent, top_k)
        logger.debug("raw results count=%d", len(results))
        if results:
            logger.debug("first result keys=%s", list(results[0].keys()))
        recipes = [
            Recipe(
                faiss_index=r["faiss_index"],
                name=r["name"],
                ingredients_with_quantities=r.get("ingredients_with_quantities", []),
                recipe_instructions=r.get("recipe_instructions", []),
                category=r.get("category", ""),
                calories=r.get("calories", ""),
                total_time=r.get("total_time", ""),
                rating=r.get("rating"),
                images=r.get("images", []),
            )
            for r in results
        ]
        logger.debug("recipes count=%d", len(recipes))
        return recipes
