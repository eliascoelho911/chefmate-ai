import json
import logging
import math
import re
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.core.models import Recipe

logger = logging.getLogger(__name__)

# Overhead fixo do prompt JSON (chaves, vírgulas, espaços, etc.)
_JSON_OVERHEAD_TOKENS = 200
# Tamanho máximo de receitas por chunk para evitar respostas longas e instáveis
_MAX_RECIPES_PER_CHUNK = 3


class RecipeTranslationError(Exception):
    """Raised when recipe translation fails and no fallback is allowed."""

    pass


class RecipeTranslator:
    """
    Translates recipe fields from English to Portuguese (pt-BR) using
    OpenRouter (the same client/model used by IngredientTranslator).

    Interface:
        translate_recipes(recipes: List[Recipe]) -> List[Recipe]
    """

    _SYSTEM_PROMPT = (
        "You are a culinary translation assistant. "
        "Translate the given recipe fields from English to Portuguese (pt-BR). "
        "Preserve culinary terminology and keep measurements/quantities unchanged. "
        "Return ONLY a JSON array where each element corresponds to a recipe, "
        "containing the translated fields: name, ingredients_cleaned, "
        "ingredients_with_quantities, recipe_instructions, and category. "
        "Do not translate numeric values, units, or ratings."
    )

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self._client = client
        self._model = model

    def translate_recipes(self, recipes: List[Recipe]) -> List[Recipe]:
        if not recipes:
            return []

        total = len(recipes)
        if total <= _MAX_RECIPES_PER_CHUNK:
            return self._translate_chunk(recipes)

        num_chunks = math.ceil(total / _MAX_RECIPES_PER_CHUNK)
        logger.info(
            "Splitting %d recipes into %d chunks (max %d per chunk)",
            total,
            num_chunks,
            _MAX_RECIPES_PER_CHUNK,
        )

        translated: List[Recipe] = []
        for i in range(num_chunks):
            start = i * _MAX_RECIPES_PER_CHUNK
            end = start + _MAX_RECIPES_PER_CHUNK
            chunk = recipes[start:end]
            logger.info(
                "Translating chunk %d/%d (%d recipes)", i + 1, num_chunks, len(chunk)
            )
            translated.extend(self._translate_chunk(chunk))

        return translated

    def _translate_chunk(self, recipes: List[Recipe]) -> List[Recipe]:
        payload = self._build_payload(recipes)
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self._SYSTEM_PROMPT,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
        messages = [system_msg, user_msg]

        # max_tokens dinâmico: evita truncamento para listas grandes
        estimated_tokens = self._estimate_tokens(payload) + _JSON_OVERHEAD_TOKENS
        max_tokens = max(512, estimated_tokens)

        try:
            logger.info(
                "Translating %d recipes via OpenRouter model=%s (max_tokens=%d)",
                len(recipes),
                self._model,
                max_tokens,
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            content = choice.message.content or ""

            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.error(
                    "LLM response truncated due to max_tokens (finish_reason=length). "
                    "Consider reducing _MAX_RECIPES_PER_CHUNK."
                )
                raise RecipeTranslationError(
                    "Recipe translation response was truncated by token limit."
                )

            return self._parse_response(content, recipes)
        except Exception as exc:
            logger.error("OpenRouter recipe translation failed: %s", exc)
            raise RecipeTranslationError(
                "Unable to translate recipes at this time. Please try again later."
            ) from exc

    @staticmethod
    def _build_payload(recipes: List[Recipe]) -> List[dict]:
        payload = []
        for recipe in recipes:
            payload.append(
                {
                    "name": recipe.name,
                    "ingredients_cleaned": recipe.ingredients_cleaned,
                    "ingredients_with_quantities": recipe.ingredients_with_quantities,
                    "recipe_instructions": recipe.recipe_instructions,
                    "category": recipe.category,
                }
            )
        return payload

    @staticmethod
    def _estimate_tokens(payload: List[dict]) -> int:
        # Heurística simples: ~4 chars/token em média para inglês
        text = json.dumps(payload, ensure_ascii=False)
        return len(text) // 4

    def _parse_response(
        self, content: str, original_recipes: List[Recipe]
    ) -> List[Recipe]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse recipe translation JSON: %s", content)
            data = self._extract_array_from_truncated_json(content)
            if not data:
                raise RecipeTranslationError(
                    "Invalid recipe translation response from LLM."
                ) from exc

        if not isinstance(data, list):
            # Alguns modelos podem retornar {"recipes": [...]}
            if isinstance(data, dict) and "recipes" in data:
                data = data["recipes"]
            else:
                raise RecipeTranslationError(
                    "Recipe translation response is not a JSON array."
                )

        if len(data) != len(original_recipes):
            logger.warning(
                "Recipe translation returned %d items but expected %d",
                len(data),
                len(original_recipes),
            )

        result: List[Recipe] = []
        for i, original in enumerate(original_recipes):
            if i >= len(data):
                # Se faltar tradução, manter original
                result.append(original)
                continue

            item = data[i]
            if not isinstance(item, dict):
                result.append(original)
                continue

            result.append(
                Recipe(
                    faiss_index=original.faiss_index,
                    name=item.get("name", original.name),
                    ingredients_cleaned=item.get(
                        "ingredients_cleaned", original.ingredients_cleaned
                    ),
                    ingredients_with_quantities=item.get(
                        "ingredients_with_quantities",
                        original.ingredients_with_quantities,
                    ),
                    recipe_instructions=item.get(
                        "recipe_instructions", original.recipe_instructions
                    ),
                    category=item.get("category", original.category),
                    calories=original.calories,
                    total_time=original.total_time,
                    rating=original.rating,
                    images=original.images,
                )
            )
        return result

    @staticmethod
    def _extract_array_from_truncated_json(text: str) -> List[dict]:
        """
        Tenta extrair objetos JSON de um array possivelmente truncado.
        Último recurso quando json.loads falha.
        """
        # Tenta encontrar um array JSON
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: tenta extrair objetos individuais
        objects = re.findall(r"\{[^{}]*\}", text)
        if objects:
            parsed = []
            for obj in objects:
                try:
                    parsed.append(json.loads(obj))
                except json.JSONDecodeError:
                    continue
            if parsed:
                logger.warning(
                    "Recovered %d recipe objects from truncated/malformed JSON via regex",
                    len(parsed),
                )
                return parsed

        return []
