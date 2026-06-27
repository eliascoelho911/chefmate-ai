import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from app.core.intent import Intent
from app.core.models import Recipe
from app.utils.ingredient_translator import IngredientTranslator, TranslationError
from app.utils.recipe_search import RecipeSearch
from app.utils.recipe_translator import RecipeTranslationError, RecipeTranslator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngredientQueryItem:
    text: str
    generalize: bool = True


class IngredientSearchService:
    """
    Deep module that orchestrates ingredient-based recipe search with translation.

    Hides the full pipeline behind a single method:
        search(items, intent, top_k) -> List[Recipe]

    Internally it:
      1. Normalizes comma-separated ingredients,
      2. Translates PT -> EN (generalized or literal),
      3. Embeds and retrieves recipes via RecipeSearch,
      4. Translates recipes back EN -> PT.
    """

    def __init__(
        self,
        recipe_search: RecipeSearch,
        ingredient_translator: IngredientTranslator,
        recipe_translator: RecipeTranslator,
    ):
        self._recipe_search = recipe_search
        self._ingredient_translator = ingredient_translator
        self._recipe_translator = recipe_translator

    def search(
        self,
        items: List[IngredientQueryItem],
        intent: Intent,
        top_k: int = 5,
    ) -> List[Recipe]:
        t_start = time.perf_counter()

        normalized = self._normalize_items(items)
        query_str = ", ".join(item.text for item in normalized)

        translated = self._translate_ingredients(normalized)

        results = self._retrieve(query_str, intent, top_k, translated)

        results = self._translate_recipes(results)

        t_total = time.perf_counter() - t_start
        logger.debug("ingredient_search_service total_time_ms=%.2f", t_total * 1000)
        return results

    @staticmethod
    def _normalize_items(items: List[IngredientQueryItem]) -> List[IngredientQueryItem]:
        """Flatten comma-separated values while preserving generalize flags."""
        normalized: List[IngredientQueryItem] = []
        for item in items:
            for part in item.text.split(","):
                part = part.strip()
                if part:
                    normalized.append(
                        IngredientQueryItem(text=part, generalize=item.generalize)
                    )
        return normalized

    def _translate_ingredients(self, items: List[IngredientQueryItem]) -> List[str]:
        t_start = time.perf_counter()

        to_generalize = [item.text for item in items if item.generalize]
        to_keep_specific = [item.text for item in items if not item.generalize]

        try:
            translated_generalized = (
                self._ingredient_translator.translate_batch(
                    to_generalize, generalize=True
                )
                if to_generalize
                else []
            )
            translated_literal = (
                self._ingredient_translator.translate_batch(
                    to_keep_specific, generalize=False
                )
                if to_keep_specific
                else []
            )

            # Map back to original order
            trans_map: dict[str, str] = {}
            for orig, trans in zip(to_generalize, translated_generalized):
                trans_map[orig.lower()] = trans
            for orig, trans in zip(to_keep_specific, translated_literal):
                trans_map[orig.lower()] = trans

            translated = [trans_map.get(item.text.lower(), item.text) for item in items]
        except TranslationError as exc:
            logger.warning(
                "Ingredient translation failed, using original terms: %s", exc
            )
            translated = [item.text for item in items]

        t_elapsed = time.perf_counter() - t_start
        logger.debug("ingredient_translation_time_ms=%.2f", t_elapsed * 1000)
        return translated

    def _retrieve(
        self,
        query_str: str,
        intent: Intent,
        top_k: int,
        required_ingredients: List[str],
    ) -> List[Recipe]:
        t_start = time.perf_counter()
        results = self._recipe_search.search(
            query_str,
            intent=intent,
            top_k=top_k,
            required_ingredients=required_ingredients,
        )
        t_elapsed = time.perf_counter() - t_start
        logger.debug("recipe_search_time_ms=%.2f", t_elapsed * 1000)
        return results

    def _translate_recipes(self, recipes: List[Recipe]) -> List[Recipe]:
        t_start = time.perf_counter()
        try:
            recipes = self._recipe_translator.translate_recipes(recipes)
        except RecipeTranslationError as exc:
            logger.warning(
                "Recipe translation failed, returning English recipes: %s", exc
            )
        t_elapsed = time.perf_counter() - t_start
        logger.debug("recipe_translation_time_ms=%.2f", t_elapsed * 1000)
        return recipes
