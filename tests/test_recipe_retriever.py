import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.intent import Intent
from app.core.models import Recipe
from app.utils.recipe_repository import RecipeRepository
from app.utils.recipe_retriever import FAISSRecipeRetriever
from app.utils.vector_index import VectorIndex


class FakeVectorIndex:
    """In-memory stand-in for VectorIndex."""

    def __init__(self):
        self._indexes = {
            "ingredients_embedding": [
                (0.1, 10),
                (0.2, 20),
                (0.3, 30),
            ],
            "title_embedding": [
                (0.05, 40),
                (0.15, 50),
            ],
            "ingredients_with_quantities_embedding": [
                (0.12, 60),
            ],
        }

    def search(self, embedding, index_name, top_k):
        results = self._indexes.get(index_name, [])
        trimmed = results[:top_k]
        distances = np.array([[d for d, _ in trimmed] + [0.0] * (top_k - len(trimmed))])
        indices = np.array([[i for _, i in trimmed] + [-1] * (top_k - len(trimmed))])
        return distances, indices

    def list_indexes(self):
        return list(self._indexes.keys())


class FakeRecipeRepository:
    """In-memory stand-in for RecipeRepository."""

    def __init__(self):
        self._recipes = {
            10: Recipe(
                faiss_index=10,
                name="Chicken Rice",
                ingredients_cleaned=["rice", "chicken"],
                ingredients_with_quantities=["1 cup rice", "200g chicken"],
                recipe_instructions=["Cook rice", "Grill chicken"],
                category="Main",
                calories="450",
                total_time="30 min",
                rating=4.5,
                images=[],
            ),
            20: Recipe(
                faiss_index=20,
                name="Beef Stew",
                ingredients_cleaned=["beef", "carrots"],
                ingredients_with_quantities=["500g beef", "2 carrots"],
                recipe_instructions=["Brown beef", "Simmer"],
                category="Main",
                calories="600",
                total_time="2 hours",
                rating=4.8,
                images=[],
            ),
        }

    def get_by_indices(self, indices):
        return [self._recipes[i] for i in indices if i in self._recipes]

    def get_strict_ids(self, required):
        return []

    def get_partial_ids(self, required, exclude):
        return []


def test_retrieve_maps_intent_to_index():
    retriever = FAISSRecipeRetriever(
        vector_index=FakeVectorIndex(),
        recipe_repository=FakeRecipeRepository(),
    )
    results = retriever.retrieve([0.0] * 384, Intent.INGREDIENT_SEARCH, top_k=2)
    assert len(results) == 2
    assert results[0].faiss_index == 10
    assert results[1].faiss_index == 20


def test_retrieve_returns_empty_when_no_matches():
    empty_repo = FakeRecipeRepository()
    empty_repo._recipes = {}
    retriever = FAISSRecipeRetriever(
        vector_index=FakeVectorIndex(),
        recipe_repository=empty_repo,
    )
    results = retriever.retrieve([0.0] * 384, Intent.INGREDIENT_SEARCH, top_k=2)
    assert results == []


def test_retrieve_defaults_to_all_indexes_for_unclear_intent():
    retriever = FAISSRecipeRetriever(
        vector_index=FakeVectorIndex(),
        recipe_repository=FakeRecipeRepository(),
    )
    results = retriever.retrieve([0.0] * 384, Intent.UNCLEAR, top_k=2)
    # Should search across all indexes and return top-k by distance
    assert len(results) <= 2


def test_retrieve_limits_to_top_k():
    retriever = FAISSRecipeRetriever(
        vector_index=FakeVectorIndex(),
        recipe_repository=FakeRecipeRepository(),
    )
    results = retriever.retrieve([0.0] * 384, Intent.INGREDIENT_SEARCH, top_k=1)
    assert len(results) == 1
