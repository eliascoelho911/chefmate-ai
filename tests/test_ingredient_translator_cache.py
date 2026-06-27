"""
Unit tests for IngredientTranslator cache behaviour.

These do NOT hit the live OpenRouter API — the LLM client is mocked.
"""

from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from app.utils.ingredient_translator import IngredientTranslator
from app.utils.translation_cache import InMemoryTranslationCache


def _make_completion(content: str) -> ChatCompletion:
    """Helper to build a minimal ChatCompletion response."""
    return ChatCompletion(
        id="test",
        object="chat.completion",
        created=0,
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


@pytest.fixture
def mock_client():
    client = MagicMock(spec=OpenAI)
    # Ensure the attribute chain exists for chat.completions.create
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client


class TestCacheBehaviour:
    def test_cache_avoids_llm_call_when_all_terms_cached(self, mock_client):
        cache = InMemoryTranslationCache()
        cache.set("frango", True, "chicken")
        cache.set("arroz", True, "rice")

        translator = IngredientTranslator(client=mock_client, model="test", cache=cache)
        result = translator.translate_batch(["frango", "arroz"], generalize=True)

        assert result == ["chicken", "rice"]
        mock_client.chat.completions.create.assert_not_called()

    def test_partial_cache_hits_only_translate_misses(self, mock_client):
        cache = InMemoryTranslationCache()
        cache.set("frango", True, "chicken")

        mock_client.chat.completions.create.return_value = _make_completion(
            '{"arroz": "rice"}'
        )

        translator = IngredientTranslator(client=mock_client, model="test", cache=cache)
        result = translator.translate_batch(["frango", "arroz"], generalize=True)

        assert result == ["chicken", "rice"]
        mock_client.chat.completions.create.assert_called_once()

    def test_cache_populated_after_llm_call(self, mock_client):
        cache = InMemoryTranslationCache()
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"batata": "potato"}'
        )

        translator = IngredientTranslator(client=mock_client, model="test", cache=cache)
        translator.translate_batch(["batata"], generalize=True)

        assert cache.get("batata", True) == "potato"

    def test_generalize_flag_is_part_of_cache_key(self, mock_client):
        cache = InMemoryTranslationCache()
        cache.set("peito de frango", True, "chicken")

        mock_client.chat.completions.create.return_value = _make_completion(
            '{"peito de frango": "chicken breast"}'
        )

        translator = IngredientTranslator(client=mock_client, model="test", cache=cache)

        # Cached generalization
        result_gen = translator.translate_batch(["peito de frango"], generalize=True)
        assert result_gen == ["chicken"]
        mock_client.chat.completions.create.assert_not_called()

        # Literal translation is a cache miss
        result_lit = translator.translate_batch(["peito de frango"], generalize=False)
        assert result_lit == ["chicken breast"]
        mock_client.chat.completions.create.assert_called_once()

    def test_deduplication_still_works_with_cache(self, mock_client):
        cache = InMemoryTranslationCache()
        cache.set("frango", True, "chicken")

        translator = IngredientTranslator(client=mock_client, model="test", cache=cache)
        result = translator.translate_batch(
            ["frango", "frango", "frango"], generalize=True
        )

        assert result == ["chicken", "chicken", "chicken"]
        mock_client.chat.completions.create.assert_not_called()

    def test_empty_list_returns_empty_and_no_llm_call(self, mock_client):
        translator = IngredientTranslator(
            client=mock_client, model="test", cache=InMemoryTranslationCache()
        )
        result = translator.translate_batch([], generalize=True)
        assert result == []
        mock_client.chat.completions.create.assert_not_called()
