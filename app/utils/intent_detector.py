import logging
import re
from typing import Dict

from app.core.intent import Intent

logger = logging.getLogger(__name__)


class IntentDetector:
    """
    Heuristic intent detector that maps free-text user input to a
    typed Intent value.
    """

    def __init__(self):
        self.recipe_keywords = [
            "recipe",
            "make",
            "cook",
            "prepare",
            "how to",
            "receita",
            "fazer",
            "cozinhar",
            "preparar",
            "como fazer",
            "sugere",
            "sugira",
            "quero",
            "gostaria",
        ]
        self.ingredient_keywords = [
            "have",
            "with",
            "using",
            "ingredients",
            "tenho",
            "com",
            "usando",
            "ingredientes",
        ]
        self.step_keywords = [
            "next step",
            "what's next",
            "then",
            "after that",
            "proximo passo",
            "próximo passo",
            "depois",
            "e agora",
            "o que vem depois",
        ]
        self.diet_keywords = [
            "vegan",
            "vegetarian",
            "gluten",
            "keto",
            "halal",
            "diet",
            "vegano",
            "vegetariano",
            "glúten",
            "dieta",
        ]
        self.nutrition_keywords = [
            "calories",
            "nutrition",
            "protein",
            "carbs",
            "calorias",
            "nutricao",
            "nutrição",
            "proteina",
            "proteína",
            "carboidratos",
        ]
        self.time_keywords = [
            "quick",
            "under",
            "minutes",
            "fast",
            "less than",
            "rapido",
            "rápido",
            "rapida",
            "rápida",
            "em menos de",
            "minutos",
        ]
        self.rating_keywords = [
            "top",
            "best",
            "highest rated",
            "popular",
            "melhor",
            "mais bem avaliado",
            "mais popular",
        ]

    def detect(self, user_input: str) -> Intent:
        """Satisfies the IntentDetector protocol."""
        return self._detect_intent(user_input)

    def _detect_intent(self, user_input: str) -> Intent:
        logger.debug("detect_intent input='%s'", user_input)
        user_input = user_input.lower()

        # --- Specific Recipe ---
        if re.search(r"(recipe for|how to make|tell me about)\s+[a-z ]+", user_input):
            return Intent.SPECIFIC_RECIPE
        if re.search(r"(receita de|como fazer|me fale sobre)\s+[a-zà-ú ]+", user_input):
            return Intent.SPECIFIC_RECIPE

        # --- Ingredient-Based Search ---
        if (
            any(keyword in user_input for keyword in self.ingredient_keywords)
            and "," in user_input
        ):
            return Intent.INGREDIENT_SEARCH

        # --- Step Navigation ---
        if any(kw in user_input for kw in self.step_keywords):
            return Intent.STEP_NAVIGATION

        # --- Dietary Filter ---
        if any(kw in user_input for kw in self.diet_keywords):
            return Intent.DIET_FILTER

        # --- Recipe Generation ---
        if any(kw in user_input for kw in self.recipe_keywords) and (
            "recipe" in user_input or "receita" in user_input
        ):
            return Intent.RECIPE_GENERATION

        # --- Portuguese generic meal suggestion (fallback) ---
        if re.search(
            r"\b(sugere|sugira|quero|gostaria|me indique)\b.*\b(jantar|almoco|almoço|cafe|cafe da manha|lanche|refeição|refeicao)\b",
            user_input,
        ):
            return Intent.RECIPE_GENERATION

        # --- Nutrition Info ---
        if any(kw in user_input for kw in self.nutrition_keywords):
            return Intent.NUTRITION_INFO

        # --- Time Filter ---
        if any(kw in user_input for kw in self.time_keywords):
            return Intent.TIME_FILTER

        # --- Rating Filter ---
        if any(kw in user_input for kw in self.rating_keywords):
            return Intent.RATING_FILTER

        # --- Default Case ---
        return Intent.UNCLEAR
