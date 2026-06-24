import json
import logging
import re
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when ingredient translation fails and no fallback is allowed."""

    pass


class IngredientTranslator:
    """
    Translates culinary ingredients from Portuguese (pt-BR) to English using
    OpenRouter, with SQLite-backed caching.

    Interface:
        translate_batch(ingredients: List[str]) -> List[str]
    """

    _SYSTEM_PROMPT = (
        "You are a culinary translation assistant. "
        "Translate the given food ingredients from Portuguese (pt-BR) to English. "
        "Return ONLY a JSON object mapping each original term to its English translation. "
        "Use common culinary names. If a term is already in English, keep it. "
        'Example: {"frango": "chicken", "arroz": "rice"}'
    )

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache_store,
    ):
        self._client = client
        self._model = model
        self._cache = cache_store

    def translate_batch(self, ingredients: List[str]) -> List[str]:
        if not ingredients:
            return []

        # Deduplicate while preserving order
        unique = list(dict.fromkeys(ingredients))
        to_translate: List[str] = []
        cached: dict[str, str] = {}

        for ing in unique:
            ing_clean = ing.strip().lower()
            if not ing_clean:
                continue
            cached_en = self._cache.get_translation(ing_clean)
            if cached_en is not None:
                cached[ing_clean] = cached_en
            else:
                to_translate.append(ing_clean)

        if to_translate:
            translated = self._call_llm(to_translate)
            for pt, en in translated.items():
                self._cache.save_translation(pt, en)
                cached[pt] = en

        # Map back to original input order and casing (but use lowercase key)
        result: List[str] = []
        for ing in ingredients:
            key = ing.strip().lower()
            if key in cached:
                result.append(cached[key])
            else:
                # Should not happen unless LLM omitted a key; fallback to original
                result.append(ing)
        return result

    def _call_llm(self, ingredients: List[str]) -> dict[str, str]:
        payload = {pt: pt for pt in ingredients}
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self._SYSTEM_PROMPT,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
        messages = [system_msg, user_msg]

        try:
            logger.info(
                "Translating %d ingredients via OpenRouter model=%s",
                len(ingredients),
                self._model,
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=256,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return self._parse_response(content, ingredients)
        except Exception as exc:
            logger.error("OpenRouter translation failed: %s", exc)
            raise TranslationError(
                "Unable to translate ingredients at this time. Please try again later."
            ) from exc

    def _parse_response(self, content: str, expected: List[str]) -> dict[str, str]:
        # Extract JSON from markdown code fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse translation JSON: %s", content)
            raise TranslationError("Invalid translation response from LLM.") from exc

        if not isinstance(data, dict):
            raise TranslationError("Translation response is not a JSON object.")

        result: dict[str, str] = {}
        for pt in expected:
            en = data.get(pt)
            if not isinstance(en, str) or not en.strip():
                # If LLM omitted a key or gave empty value, keep original as fallback
                result[pt] = pt
                logger.warning("Missing translation for '%s', using original", pt)
            else:
                result[pt] = en.strip().lower()
        return result
