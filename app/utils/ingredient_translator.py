import json
import logging
import math
import re
import time
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

logger = logging.getLogger(__name__)

# Tamanho máximo de um chunk para evitar respostas longas e instáveis
_MAX_CHUNK_SIZE = 10


class TranslationError(Exception):
    """Raised when ingredient translation fails and no fallback is allowed."""

    pass


class IngredientTranslator:
    """
    Translates culinary ingredients from Portuguese (pt-BR) to English using
    OpenRouter.

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

    _SYSTEM_PROMPT_LITERAL = (
        "You are a culinary translation assistant. "
        "Translate the given food ingredients from Portuguese (pt-BR) to English. "
        "Return ONLY a JSON object mapping each original term to its English translation. "
        "Use common culinary names. If a term is already in English, keep it. "
        "IMPORTANT: Preserve the specific meaning of each term. Do not generalize. "
        "For example, use 'chicken breast' for 'peito de frango', 'brown rice' for 'arroz integral', "
        "and 'broccoli florets' for 'floretes de brócolis'. "
        'Example: {"peito de frango": "chicken breast", "arroz": "rice", "floretes de brócolis": "broccoli florets"}'
    )

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self._client = client
        self._model = model

    def translate_batch(
        self, ingredients: List[str], generalize: bool = True
    ) -> List[str]:
        t_start = time.perf_counter()
        if not ingredients:
            return []

        # Deduplicate while preserving order
        unique = list(dict.fromkeys(ingredients))
        to_translate: List[str] = []

        for ing in unique:
            ing_clean = ing.strip().lower()
            if not ing_clean:
                continue
            to_translate.append(ing_clean)

        translated: dict[str, str] = {}
        if to_translate:
            translated = self._translate_in_chunks(to_translate, generalize=generalize)

        # Map back to original input order and casing (but use lowercase key)
        result: List[str] = []
        for ing in ingredients:
            key = ing.strip().lower()
            if key in translated:
                result.append(translated[key])
            else:
                # Should not happen unless LLM omitted a key; fallback to original
                result.append(ing)

        t_total = time.perf_counter() - t_start
        logger.debug(
            "translate_batch total_ingredients=%d translated=%d total_time_ms=%.2f",
            len(ingredients),
            len(result),
            t_total * 1000,
        )
        return result

    def _translate_in_chunks(
        self, ingredients: List[str], generalize: bool = True
    ) -> dict[str, str]:
        """
        Divide ingredientes em chunks para evitar respostas truncadas
        por max_tokens. Chunks menores = maior confiabilidade de JSON válido.
        """
        total = len(ingredients)
        if total <= _MAX_CHUNK_SIZE:
            return self._call_llm(ingredients, generalize=generalize)

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
            chunk_result = self._call_llm(chunk, generalize=generalize)
            merged.update(chunk_result)

        return merged

    def _call_llm(
        self, ingredients: List[str], generalize: bool = True
    ) -> dict[str, str]:
        payload = {pt: pt for pt in ingredients}
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self._SYSTEM_PROMPT
            if generalize
            else self._SYSTEM_PROMPT_LITERAL,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
        messages = [system_msg, user_msg]

        try:
            logger.info(
                "Translating %d ingredients via OpenRouter model=%s (generalize=%s)",
                len(ingredients),
                self._model,
                generalize,
            )
            t_llm_start = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            t_llm_elapsed = time.perf_counter() - t_llm_start
            logger.debug(
                "ingredient_translation_llm_time_ms=%.2f items=%d",
                t_llm_elapsed * 1000,
                len(ingredients),
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            # Detectar truncamento por max_tokens (causa raiz do JSON inválido)
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning(
                    "LLM response truncated due to max_tokens (finish_reason=length). "
                    "Attempting to parse partial response."
                )

            return self._parse_response(content, ingredients)
        except Exception as exc:
            logger.error("OpenRouter translation failed: %s", exc)
            logger.warning("Returning untranslated ingredients due to error")
            return {pt: pt for pt in ingredients}

    def _parse_response(self, content: str, expected: List[str]) -> dict[str, str]:
        # Extract JSON from markdown code fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data: dict = {}
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse translation JSON: %s", content)
            # Fallback de última instância: tentar extrair pares com regex
            data = self._extract_pairs_from_truncated_json(content)
        else:
            if isinstance(parsed, dict):
                data = parsed
            else:
                logger.error("Translation response is not a JSON object.")

        if not data:
            logger.warning("No translations parsed; returning original ingredients")
            return {pt: pt for pt in expected}

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
