import logging
from typing import List, Optional

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
        retrieve_with_ingredients(query_embedding, intent, top_k, required) -> List[Recipe]

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

    # How many more results to fetch from FAISS when filtering by ingredients.
    _OVERFETCH = 5

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
        return self._do_retrieve(query_embedding, intent, top_k)

    def retrieve_with_ingredients(
        self,
        query_embedding: List[float],
        intent: Intent,
        top_k: int,
        required_ingredients: List[str],
    ) -> List[Recipe]:
        if not required_ingredients:
            return self._do_retrieve(query_embedding, intent, top_k)

        logger.debug(
            "retrieve_with_ingredients intent=%s top_k=%d required=%s",
            intent.value,
            top_k,
            required_ingredients,
        )

        # 1. Find strict matches (all ingredients present)
        strict_ids = self._recipe_repository.get_strict_ids(required_ingredients)
        strict_set = set(strict_ids)
        logger.debug("strict matches count=%d", len(strict_set))

        # 2. Find partial matches (at least one ingredient)
        partial_raw = self._recipe_repository.get_partial_ids(
            required_ingredients, exclude=strict_ids
        )
        partial_map = {idx: count for idx, count in partial_raw}
        logger.debug("partial matches count=%d", len(partial_map))

        # 3. Over-fetch from FAISS to score candidates
        import numpy as np

        index_name = self._INTENT_TO_INDEX.get(intent, "ingredients_embedding")
        search_k = max(top_k * self._OVERFETCH, len(strict_set) + len(partial_map))
        distances, indices = self._vector_index.search(
            np.array(query_embedding), index_name, search_k
        )

        # 4. Score each candidate
        scored: List[tuple[float, int]] = []
        total_required = len(required_ingredients)

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            idx_int = int(idx)

            # Normalize FAISS distance (L2) to a 0-1 score where higher is better
            faiss_score = 1.0 / (1.0 + float(dist))

            if idx_int in strict_set:
                coverage = 1.0
            elif idx_int in partial_map:
                coverage = partial_map[idx_int] / total_required
            else:
                coverage = 0.0

            # Combined score: coverage is weighted heavily, FAISS breaks ties
            combined = (coverage * 0.7) + (faiss_score * 0.3)
            scored.append((combined, idx_int))

        # 5. Sort descending by combined score
        scored.sort(key=lambda x: x[0], reverse=True)
        best_indices = [idx for _, idx in scored[:top_k]]

        return self._recipe_repository.get_by_indices(best_indices)

    def _do_retrieve(
        self, query_embedding: List[float], intent: Intent, top_k: int
    ) -> List[Recipe]:
        logger.debug("retrieve intent=%s top_k=%d", intent.value, top_k)

        import numpy as np

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

    def _default_search(self, embedding, top_k: int) -> List[Recipe]:
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

    def _hydrate(self, distances, indices, top_k: int) -> List[Recipe]:
        valid_indices = [int(idx) for idx in indices if idx != -1]
        recipes = self._recipe_repository.get_by_indices(valid_indices)
        logger.debug("hydrated recipes count=%d", len(recipes))
        return recipes[:top_k]
