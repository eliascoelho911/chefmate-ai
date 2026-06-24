import pytest
import sqlite3
from app.core.models import Recipe
from app.utils.recipe_repository import RecipeRepository
from app.utils.sqlite_store import RecipeSQLiteStore


def test_get_by_indices_returns_recipes():
    # In-memory database
    store = RecipeSQLiteStore(":memory:")
    store.conn.execute(
        """
        INSERT INTO recipes (
            faiss_index, recipe_id, name, author_id, author_name,
            cook_time, prep_time, total_time, date_published,
            description, images, recipe_category, keywords,
            recipe_ingredient_quantities, recipe_ingredient_parts,
            aggregated_rating, review_count, calories, fat_content,
            saturated_fat_content, cholesterol_content, sodium_content,
            carbohydrate_content, fiber_content, sugar_content,
            protein_content, recipe_servings, recipe_yield,
            recipe_instructions, ingredients_raw, ingredients_cleaned,
            ingredients_with_quantities
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            100,
            "Test Recipe",
            1,
            "Chef",
            "10 min",
            "5 min",
            "15 min",
            "2024-01-01",
            "A test recipe",
            '["img1.jpg"]',
            "Dessert",
            '["sweet"]',
            '["1 cup sugar"]',
            '["sugar"]',
            4.5,
            10,
            300.0,
            10.0,
            5.0,
            50.0,
            200.0,
            40.0,
            2.0,
            20.0,
            5.0,
            4.0,
            "4 servings",
            '["Mix ingredients", "Bake"]',
            '["sugar, flour"]',
            '["sugar", "flour"]',
            '["1 cup sugar", "2 cups flour"]',
        ),
    )
    store.conn.commit()

    repo = RecipeRepository(store)
    recipes = repo.get_by_indices([1])

    assert len(recipes) == 1
    recipe = recipes[0]
    assert isinstance(recipe, Recipe)
    assert recipe.faiss_index == 1
    assert recipe.name == "Test Recipe"
    assert recipe.category == "Dessert"
    assert recipe.ingredients_with_quantities == ["1 cup sugar", "2 cups flour"]


def test_get_by_indices_skips_missing():
    store = RecipeSQLiteStore(":memory:")
    repo = RecipeRepository(store)
    recipes = repo.get_by_indices([999])
    assert recipes == []


def test_get_by_indices_batch_order_preserved():
    store = RecipeSQLiteStore(":memory:")
    for i in range(3):
        store.conn.execute(
            "INSERT INTO recipes (faiss_index, name, recipe_id) VALUES (?, ?, ?)",
            (i, f"Recipe {i}", i),
        )
    store.conn.commit()

    repo = RecipeRepository(store)
    recipes = repo.get_by_indices([2, 0, 1])
    assert [r.faiss_index for r in recipes] == [2, 0, 1]
