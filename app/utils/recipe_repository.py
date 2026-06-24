import logging
from typing import List, Optional

from app.core.models import Recipe
from app.utils.sqlite_store import RecipeSQLiteStore

logger = logging.getLogger(__name__)


class RecipeRepository:
    """
    Adapter that satisfies the recipe-storage seam using SQLite.

    Interface:
        get_by_indices(indices) -> List[Recipe]

    Hides column-to-field mapping, JSON deserialization, and the
    SQLite schema from the rest of the application.
    """

    def __init__(self, sqlite_store: RecipeSQLiteStore):
        self._store = sqlite_store

    def get_by_indices(self, indices: List[int]) -> List[Recipe]:
        """Batch-fetch recipes by FAISS index, skipping misses."""
        if not indices:
            return []

        recipes: List[Recipe] = []
        for idx in indices:
            row = self._store.get_recipe_by_faiss_index(idx)
            if row is None:
                logger.debug("SQLite miss for faiss_index=%d", idx)
                continue
            recipes.append(self._row_to_recipe(row))

        return recipes

    @staticmethod
    def _row_to_recipe(row: dict) -> Recipe:
        """Map a deserialized SQLite row dict to a typed Recipe model."""
        return Recipe(
            faiss_index=row["faiss_index"],
            name=row.get("name") or "",
            ingredients_with_quantities=row.get("ingredients_with_quantities") or [],
            recipe_instructions=row.get("recipe_instructions") or [],
            category=row.get("recipe_category") or "",
            calories=str(row.get("calories") or ""),
            total_time=row.get("total_time") or "",
            rating=row.get("aggregated_rating"),
            images=row.get("images") or [],
        )
