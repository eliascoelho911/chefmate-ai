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

logger = logging.getLogger(__name__)

# Tokens estimados por ingrediente (pt + en + JSON overhead) — margem segura
_TOKENS_PER_INGREDIENT = 30
# Overhead fixo do prompt JSON (chaves, vírgulas, espaços, etc.)
_JSON_OVERHEAD_TOKENS = 100
# Tamanho máximo de um chunk para evitar respostas longas e instáveis
_MAX_CHUNK_SIZE = 10


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
        "IMPORTANT: Prefer generic, broad ingredient terms that are most likely to appear in recipe indexes. "
        "Avoid overly specific cuts or preparations. For example, use 'chicken' instead of 'chicken breast', "
        "'rice' instead of 'brown rice', and 'broccoli' instead of 'broccoli florets'. "
        'Example: {"frango": "chicken", "arroz": "rice", "peito de frango": "chicken"}'
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
            translated = self._translate_in_chunks(to_translate)
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

    def _translate_in_chunks(self, ingredients: List[str]) -> dict[str, str]:
        """
        Divide ingredientes em chunks para evitar respostas truncadas
        por max_tokens. Chunks menores = maior confiabilidade de JSON válido.
        """
        total = len(ingredients)
        if total <= _MAX_CHUNK_SIZE:
            return self._call_llm(ingredients)

        num_chunks = math.ceil(total / _MAX_CHUNK_SIZE)
        logger.info(
            "Splitting %d ingredients into %d chunks (max %d per chunk)",
            total,
            num_chunks,
            _MAX_CHUNK_SIZE,
        )

        merged: dict[str, str] = {}
        for i in range(num_chunks):
            start = i * _MAX_CHUNK_SIZE
            end = start + _MAX_CHUNK_SIZE
            chunk = ingredients[start:end]
            logger.info(
                "Translating chunk %d/%d (%d items)", i + 1, num_chunks, len(chunk)
            )
            chunk_result = self._call_llm(chunk)
            merged.update(chunk_result)

        return merged

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

        # max_tokens dinâmico: evita truncamento para listas grandes
        estimated_tokens = (
            len(ingredients) * _TOKENS_PER_INGREDIENT + _JSON_OVERHEAD_TOKENS
        )
        max_tokens = max(256, estimated_tokens)

        try:
            logger.info(
                "Translating %d ingredients via OpenRouter model=%s (max_tokens=%d)",
                len(ingredients),
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

            # Detectar truncamento por max_tokens (causa raiz do JSON inválido)
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.error(
                    "LLM response truncated due to max_tokens (finish_reason=length). "
                    "Consider increasing _TOKENS_PER_INGREDIENT or reducing _MAX_CHUNK_SIZE."
                )
                raise TranslationError(
                    "Translation response was truncated by token limit."
                )

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
            # Fallback de última instância: tentar extrair pares com regex
            data = self._extract_pairs_from_truncated_json(content)
            if not data:
                raise TranslationError(
                    "Invalid translation response from LLM."
                ) from exc

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

    @staticmethod
    def _extract_pairs_from_truncated_json(text: str) -> dict[str, str]:
        """
        Tenta extrair pares 'chave': 'valor' de um JSON possivelmente truncado.
        Útil como último recurso quando json.loads falha por corte no final.
        """
        pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')
        matches = pattern.findall(text)
        if not matches:
            return {}
        extracted = {}
        for key, val in matches:
            extracted[key] = val
        logger.warning(
            "Recovered %d translation pairs from truncated/malformed JSON via regex",
            len(extracted),
        )
        return extracted
