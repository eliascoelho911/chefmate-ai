import logging
from typing import List

import numpy as np

from app.core.intent import Intent
from app.core.interfaces import RecipeRetriever as RecipeRetrieverProtocol
from app.core.models import Recipe
from app.utils.recipe_repository import RecipeRepository
from app.utils.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class FAISSRecipeRetriever:
    """
    Deep module that orchestrates vector search + metadata hydration.

    Interface:
        retrieve(query_embedding, intent, top_k) -> List[Recipe]

    The caller provides an embedding and an Intent; this module decides
    which FAISS index to query, fetches vectors, and hydrates Recipe
    objects via the RecipeRepository.  All FAISS and SQLite specifics
    are hidden behind injected adapters.
    """

    # Mapping from domain intent to the physical FAISS index name.
    _INTENT_TO_INDEX = {
        Intent.INGREDIENT_SEARCH: "ingredients_embedding",
        Intent.SPECIFIC_RECIPE: "title_embedding",
        Intent.RECIPE_GENERATION: "ingredients_with_quantities_embedding",
        Intent.STEP_NAVIGATION: "ingredients_with_quantities_embedding",
        Intent.DIET_FILTER: "ingredients_embedding",
        Intent.NUTRITION_INFO: "ingredients_embedding",
        Intent.TIME_FILTER: "ingredients_embedding",
        Intent.RATING_FILTER: "title_embedding",
    }

    def __init__(
        self,
        vector_index: VectorIndex,
        recipe_repository: RecipeRepository,
    ):
        self._vector_index = vector_index
        self._recipe_repository = recipe_repository

    def retrieve(
        self, query_embedding: List[float], intent: Intent, top_k: int
    ) -> List[Recipe]:
        logger.debug("retrieve intent=%s top_k=%d", intent.value, top_k)

        embedding_array = np.array(query_embedding)

        # Fallback: if intent is not mapped, search all indexes and merge.
        if intent not in self._INTENT_TO_INDEX:
            logger.debug("intent %s not mapped, using default_search", intent.value)
            return self._default_search(embedding_array, top_k)

        index_name = self._INTENT_TO_INDEX[intent]
        distances, indices = self._vector_index.search(
            embedding_array, index_name, top_k
        )
        return self._hydrate(distances[0], indices[0], top_k)

    def _default_search(self, embedding: np.ndarray, top_k: int) -> List[Recipe]:
        logger.debug("default_search top_k=%d", top_k)
        all_results: List[tuple[float, int]] = []

        for index_name in self._vector_index.list_indexes():
            distances, indices = self._vector_index.search(embedding, index_name, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                all_results.append((float(dist), int(idx)))

        all_results.sort(key=lambda x: x[0])
        unique_indices = [idx for _, idx in all_results[:top_k]]
        return self._recipe_repository.get_by_indices(unique_indices)

    def _hydrate(
        self, distances: np.ndarray, indices: np.ndarray, top_k: int
    ) -> List[Recipe]:
        valid_indices = [int(idx) for idx in indices if idx != -1]
        recipes = self._recipe_repository.get_by_indices(valid_indices)
        logger.debug("hydrated recipes count=%d", len(recipes))
        return recipes[:top_k]
