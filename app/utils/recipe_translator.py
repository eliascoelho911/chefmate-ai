import hashlib
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Protocol

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.core.models import Recipe

logger = logging.getLogger(__name__)

# Tamanho máximo de receitas por chunk. Aumentado de 2 para 5 para reduzir
# o número de chamadas paralelas e o overhead de TTFT + prompt fixo.
_MAX_RECIPES_PER_CHUNK = 5

# Limite de workers paralelos contra a OpenRouter para evitar rate limits.
_MAX_PARALLEL_WORKERS = 4

# Campos traduzíveis e seus respectivos códigos compactos no formato delimitado.
_FIELD_MAP = {
    "N": "name",
    "C": "category",
    "I": "ingredients_cleaned",
    "Q": "ingredients_with_quantities",
    "S": "recipe_instructions",
}


class RecipeTranslationError(Exception):
    """Raised when recipe translation fails and no fallback is allowed."""

    pass


class RecipeCache(Protocol):
    """Seam for recipe translation caching."""

    def get(self, key: str) -> Optional[dict]:
        """Return cached translation dict or None if absent."""
        ...

    def set(self, key: str, translation: dict) -> None:
        """Store a translation dict. Overwrites existing entries silently."""
        ...

    def load(self) -> None:
        """Hydrate cache from external storage (no-op for pure memory)."""
        ...

    def save(self) -> None:
        """Persist cache to external storage (no-op for pure memory)."""
        ...


class InMemoryRecipeCache:
    """Simple volatile cache backed by a dict, keyed by SHA-256 of source fields."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        return self._store.get(key)

    def set(self, key: str, translation: dict) -> None:
        self._store[key] = translation

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass


def _recipe_translation_key(recipe: Recipe) -> str:
    """Build a deterministic cache key from translatable fields."""
    payload = (
        recipe.name
        + "|"
        + recipe.category
        + "|"
        + "\n".join(recipe.ingredients_cleaned)
        + "|"
        + "\n".join(recipe.ingredients_with_quantities)
        + "|"
        + "\n".join(recipe.recipe_instructions)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RecipeTranslator:
    """
    Translates recipe fields from English to Portuguese (pt-BR) using
    OpenRouter (the same client/model used by IngredientTranslator).

    Optimisations:
      - Chunk size increased to 5 recipes to amortise TTFT and system-prompt cost.
      - Delimited compact format (instead of JSON) to reduce token count ~20%.
      - Only translatable fields are sent to the LLM.
      - In-memory caching avoids re-translating identical recipes.
      - max_tokens is set dynamically based on chunk size.
      - Parallel workers are capped to avoid rate-limit contention.

    Interface:
        translate_recipes(recipes: List[Recipe]) -> List[Recipe]
    """

    # Minimal system prompt to save tokens while keeping instructions clear.
    _SYSTEM_PROMPT = (
        "Translate recipe fields from English to Portuguese (pt-BR). "
        "Preserve culinary terminology and keep measurements/quantities unchanged. "
        "Return ONLY the translated recipes using the exact same delimited format provided. "
        "Use the field codes N, C, I, Q, S. Separate list items with '|'. "
        "Do not translate numeric values, units, or ratings. "
        "Do not add markdown or extra text."
    )

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache: Optional[RecipeCache] = None,
    ):
        self._client = client
        self._model = model
        self._cache = cache or InMemoryRecipeCache()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def translate_recipes(self, recipes: List[Recipe]) -> List[Recipe]:
        t_start = time.perf_counter()
        if not recipes:
            return []

        total = len(recipes)

        # 1. Check cache ---------------------------------------------------
        cached_results: dict[int, Recipe] = {}
        to_translate: List[Recipe] = []
        cache_hits = 0

        for idx, recipe in enumerate(recipes):
            key = _recipe_translation_key(recipe)
            hit = self._cache.get(key)
            if hit is not None:
                cached_results[idx] = self._apply_translation_dict(recipe, hit)
                cache_hits += 1
            else:
                to_translate.append(recipe)

        if cache_hits:
            logger.info(
                "Recipe translation cache hit for %d/%d recipes",
                cache_hits,
                total,
            )

        # 2. Translate misses in chunks -------------------------------------
        translated: List[Recipe] = []
        if to_translate:
            translated = self._translate_recipes_uncached(to_translate)
            # Populate cache
            for original, tr in zip(to_translate, translated):
                key = _recipe_translation_key(original)
                self._cache.set(
                    key,
                    {
                        "name": tr.name,
                        "category": tr.category,
                        "ingredients_cleaned": tr.ingredients_cleaned,
                        "ingredients_with_quantities": tr.ingredients_with_quantities,
                        "recipe_instructions": tr.recipe_instructions,
                    },
                )
            self._cache.save()

        # 3. Reassemble in original order -----------------------------------
        result: List[Recipe] = [None] * total  # type: ignore[list-item]
        tr_iter = iter(translated)
        for idx in range(total):
            if idx in cached_results:
                result[idx] = cached_results[idx]
            else:
                result[idx] = next(tr_iter)

        t_total = time.perf_counter() - t_start
        logger.debug(
            "translate_recipes total=%d cached=%d translated=%d total_time_ms=%.2f",
            total,
            cache_hits,
            len(to_translate),
            t_total * 1000,
        )
        return result

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _translate_recipes_uncached(self, recipes: List[Recipe]) -> List[Recipe]:
        total = len(recipes)
        if total == 0:
            return []

        if total <= _MAX_RECIPES_PER_CHUNK:
            return self._translate_chunk(recipes)

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
        workers = min(num_chunks, _MAX_PARALLEL_WORKERS)
        if workers == 1:
            translated.extend(self._translate_chunk(chunks[0]))
        else:
            logger.info("Translating %d chunks with %d workers", num_chunks, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for chunk_result in executor.map(self._translate_chunk, chunks):
                    translated.extend(chunk_result)

        return translated

    def _translate_chunk(self, recipes: List[Recipe]) -> List[Recipe]:
        payload = self._build_delimited_payload(recipes)
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self._SYSTEM_PROMPT,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": payload,
        }
        messages = [system_msg, user_msg]

        max_tokens = self._estimate_max_tokens(len(recipes))

        try:
            logger.info(
                "Translating %d recipes via OpenRouter model=%s max_tokens=%d",
                len(recipes),
                self._model,
                max_tokens,
            )
            t_llm_start = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                top_p=0.9,
                max_tokens=max_tokens,
            )
            t_llm_elapsed = time.perf_counter() - t_llm_start
            logger.debug(
                "recipe_translation_llm_time_ms=%.2f recipes=%d max_tokens=%d",
                t_llm_elapsed * 1000,
                len(recipes),
                max_tokens,
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

    # --------------------------------------------------------------------- #
    # Payload / parsing
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_delimited_payload(recipes: List[Recipe]) -> str:
        """Build a compact delimited string containing only translatable fields."""
        parts: List[str] = []
        for i, recipe in enumerate(recipes, start=1):
            block = [
                f"[{i}]",
                f"N:{recipe.name}",
                f"C:{recipe.category}",
                f"I:{'|'.join(recipe.ingredients_cleaned)}",
                f"Q:{'|'.join(recipe.ingredients_with_quantities)}",
                f"S:{'|'.join(recipe.recipe_instructions)}",
            ]
            parts.append("\n".join(block))
        return "\n---\n".join(parts)

    def _parse_response(
        self, content: str, original_recipes: List[Recipe]
    ) -> List[Recipe]:
        # 1. Try the primary delimited parser -------------------------------
        parsed = self._parse_delimited(content, original_recipes)
        if parsed is not None:
            return parsed

        # 2. Fallback: JSON (model may occasionally output JSON) ------------
        logger.warning("Delimited parse failed; attempting JSON fallback")
        parsed = self._parse_json(content, original_recipes)
        if parsed is not None:
            return parsed

        # 3. Fallback: regex object extraction ------------------------------
        logger.warning("JSON fallback failed; attempting regex fallback")
        parsed = self._parse_regex(content, original_recipes)
        if parsed is not None:
            return parsed

        logger.error("All parsing strategies failed; returning original recipes")
        return original_recipes

    # -- Delimited parser --------------------------------------------------
    def _parse_delimited(
        self, content: str, original_recipes: List[Recipe]
    ) -> Optional[List[Recipe]]:
        """Parse the compact delimited format. Returns None on failure."""
        try:
            blocks = [b.strip() for b in content.split("---") if b.strip()]
            if not blocks:
                return None

            data: List[dict] = []
            for block in blocks:
                entry: dict[str, str | List[str]] = {}
                for raw_line in block.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("["):
                        continue
                    if ":" not in line:
                        continue
                    code, value = line.split(":", 1)
                    code = code.strip().upper()
                    value = value.strip()
                    if code in ("I", "Q", "S"):
                        entry[_FIELD_MAP[code]] = [
                            v.strip() for v in value.split("|") if v.strip()
                        ]
                    elif code in ("N", "C"):
                        entry[_FIELD_MAP[code]] = value
                if entry:
                    data.append(entry)

            if not data:
                return None

            return self._merge_parsed(data, original_recipes)
        except Exception as exc:
            logger.debug("Delimited parse exception: %s", exc)
            return None

    # -- JSON fallback parser ----------------------------------------------
    def _parse_json(
        self, content: str, original_recipes: List[Recipe]
    ) -> Optional[List[Recipe]]:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                data = parsed
            elif isinstance(parsed, dict):
                # Accept either a top-level list under "recipes" or direct keys
                if "recipes" in parsed:
                    data = parsed["recipes"]
                else:
                    data = [parsed]
            else:
                return None

            return self._merge_parsed(data, original_recipes)
        except Exception as exc:
            logger.debug("JSON parse exception: %s", exc)
            return None

    # -- Regex fallback parser ---------------------------------------------
    def _parse_regex(
        self, content: str, original_recipes: List[Recipe]
    ) -> Optional[List[Recipe]]:
        """Last-resort extraction of JSON-like objects."""
        try:
            objects = re.findall(r"\{[^{}]*\}", content)
            if not objects:
                return None
            parsed: List[dict] = []
            for obj in objects:
                try:
                    parsed.append(json.loads(obj))
                except json.JSONDecodeError:
                    continue
            if not parsed:
                return None
            logger.warning(
                "Recovered %d recipe objects from truncated/malformed text via regex",
                len(parsed),
            )
            return self._merge_parsed(parsed, original_recipes)
        except Exception as exc:
            logger.debug("Regex parse exception: %s", exc)
            return None

    # -- Merge helpers -----------------------------------------------------
    def _merge_parsed(
        self, data: List[dict], original_recipes: List[Recipe]
    ) -> Optional[List[Recipe]]:
        """Merge parsed dicts back into Recipe objects, preserving untranslated fields."""
        if not data:
            return None

        if len(data) != len(original_recipes):
            logger.warning(
                "Recipe translation returned %d items but expected %d",
                len(data),
                len(original_recipes),
            )

        result: List[Recipe] = []
        for i, original in enumerate(original_recipes):
            if i >= len(data):
                result.append(original)
                continue
            item = data[i]
            if not isinstance(item, dict):
                result.append(original)
                continue
            result.append(self._apply_translation_dict(original, item))
        return result

    @staticmethod
    def _apply_translation_dict(original: Recipe, item: dict) -> Recipe:
        """Build a new Recipe from original, overlaying translated fields."""
        return Recipe(
            faiss_index=original.faiss_index,
            name=item.get("name") or original.name,
            ingredients_cleaned=_safe_list(
                item.get("ingredients_cleaned"), original.ingredients_cleaned
            ),
            ingredients_with_quantities=_safe_list(
                item.get("ingredients_with_quantities"),
                original.ingredients_with_quantities,
            ),
            recipe_instructions=_safe_list(
                item.get("recipe_instructions"), original.recipe_instructions
            ),
            category=item.get("category") or original.category,
            calories=original.calories,
            total_time=original.total_time,
            rating=original.rating,
            images=original.images,
        )

    # --------------------------------------------------------------------- #
    # Token estimation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _estimate_max_tokens(num_recipes: int) -> int:
        """Heuristic: allocate ~2 k tokens per recipe, capped at 16 k."""
        return min(2048 * num_recipes, 16384)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_list(value, fallback: List[str]) -> List[str]:
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return fallback
