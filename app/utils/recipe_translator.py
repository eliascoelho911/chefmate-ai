import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.core.models import Recipe

logger = logging.getLogger(__name__)

# Overhead fixo do prompt JSON (chaves, vírgulas, espaços, etc.)
_JSON_OVERHEAD_TOKENS = 512
# Fator de expansão estimado: tradução pt-BR tende a ser ~1.3x o tamanho do en
_OUTPUT_TOKEN_FACTOR = 1.5
# Tamanho máximo de receitas por chunk para evitar respostas longas e instáveis
_MAX_RECIPES_PER_CHUNK = 2


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
        "IMPORTANT: You must ALWAYS translate and include the 'name' field for every recipe. "
        "Do not omit or leave the recipe name in English. "
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
        t_start = time.perf_counter()
        if not recipes:
            return []

        total = len(recipes)
        if total <= _MAX_RECIPES_PER_CHUNK:
            result = self._translate_chunk(recipes)
            t_total = time.perf_counter() - t_start
            logger.debug(
                "translate_recipes total=%d total_time_ms=%.2f",
                total,
                t_total * 1000,
            )
            return result

        num_chunks = math.ceil(total / _MAX_RECIPES_PER_CHUNK)
        logger.info(
            "Splitting %d recipes into %d chunks (max %d per chunk)",
            total,
            num_chunks,
            _MAX_RECIPES_PER_CHUNK,
        )

        chunks = [
            recipes[i * _MAX_RECIPES_PER_CHUNK : (i + 1) * _MAX_RECIPES_PER_CHUNK]
            for i in range(num_chunks)
        ]

        translated: List[Recipe] = []
        if num_chunks == 1:
            translated.extend(self._translate_chunk(chunks[0]))
        else:
            logger.info("Translating %d chunks in parallel", num_chunks)
            with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                for chunk_result in executor.map(self._translate_chunk, chunks):
                    translated.extend(chunk_result)

        t_total = time.perf_counter() - t_start
        logger.debug(
            "translate_recipes total=%d chunks=%d total_time_ms=%.2f",
            total,
            num_chunks,
            t_total * 1000,
        )
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
        estimated_input_tokens = self._estimate_tokens(payload)
        max_tokens = max(
            1024,
            int(estimated_input_tokens * _OUTPUT_TOKEN_FACTOR) + _JSON_OVERHEAD_TOKENS,
        )

        try:
            logger.info(
                "Translating %d recipes via OpenRouter model=%s (max_tokens=%d)",
                len(recipes),
                self._model,
                max_tokens,
            )
            t_llm_start = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            t_llm_elapsed = time.perf_counter() - t_llm_start
            logger.debug(
                "recipe_translation_llm_time_ms=%.2f recipes=%d",
                t_llm_elapsed * 1000,
                len(recipes),
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning(
                    "LLM response truncated due to max_tokens (finish_reason=length). "
                    "Attempting to parse partial response."
                )

            return self._parse_response(content, recipes)
        except Exception as exc:
            logger.error("OpenRouter recipe translation failed: %s", exc)
            logger.warning("Returning untranslated recipes due to error")
            return recipes

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

        data: List[dict] = []
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse recipe translation JSON: %s", content)
            data = self._extract_array_from_truncated_json(content)
        else:
            if isinstance(parsed, list):
                data = parsed
            elif isinstance(parsed, dict) and "recipes" in parsed:
                data = parsed["recipes"]
            else:
                logger.error(
                    "Recipe translation response is not a JSON array or object with 'recipes' key."
                )

        if not data:
            logger.warning("No translations parsed; returning original recipes")
            return original_recipes

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
        # 1. Tenta encontrar um array JSON completo
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            array_text = match.group(0)
            try:
                return json.loads(array_text)
            except json.JSONDecodeError:
                pass

            # 2. Tenta fechar o array se estiver truncado no final
            fixed = array_text.rstrip()
            if not fixed.endswith("]"):
                fixed = fixed + "]"
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # 3. Tenta remover o último objeto incompleto e fechar o array
            last_comma = fixed.rfind(",", 0, len(fixed) - 1)
            if last_comma > 0:
                trimmed = fixed[:last_comma] + "]"
                try:
                    return json.loads(trimmed)
                except json.JSONDecodeError:
                    pass

        # 4. Fallback: tenta extrair objetos individuais completos
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
