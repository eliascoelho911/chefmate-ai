from unittest.mock import MagicMock

from app.core.intent import Intent
from app.core.models import ChatHistory, ChatMessage, Recipe
from app.utils.chat_orchestrator import ChatOrchestrator


class FakeRecipeSearch:
    def search(self, query, intent=None, top_k=5):
        return [
            Recipe(
                faiss_index=0,
                name="Fake Recipe",
                ingredients_with_quantities=["1 cup flour"],
                recipe_instructions=["Mix and bake"],
                category="Dessert",
                calories="200",
                total_time="30 min",
                rating=4.5,
                images=[],
            )
        ]


class FakeIntentDetector:
    def detect(self, user_input):
        return Intent.RECIPE_GENERATION


class FakeLLMRunner:
    def stream_response(self, messages: list):
        yield "Hello"
        yield " "
        yield "world"


class FakePromptBuilder:
    def build_messages(self, chat_history, intent, recipes):
        return [{"role": "system", "content": "test"}]


def test_chat_orchestrator_returns_tokens():
    orchestrator = ChatOrchestrator(
        recipe_search=FakeRecipeSearch(),
        intent_detector=FakeIntentDetector(),
        llm_runner=FakeLLMRunner(),
        prompt_builder=FakePromptBuilder(),
    )
    history = ChatHistory(messages=[ChatMessage(role="user", content="hello")])
    tokens = list(orchestrator.chat(history))
    assert tokens == ["Hello", " ", "world"]


def test_chat_orchestrator_rejects_empty_history():
    orchestrator = ChatOrchestrator(
        recipe_search=FakeRecipeSearch(),
        intent_detector=FakeIntentDetector(),
        llm_runner=FakeLLMRunner(),
        prompt_builder=FakePromptBuilder(),
    )
    history = ChatHistory(messages=[])
    try:
        list(orchestrator.chat(history))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_chat_orchestrator_rejects_no_user_message():
    orchestrator = ChatOrchestrator(
        recipe_search=FakeRecipeSearch(),
        intent_detector=FakeIntentDetector(),
        llm_runner=FakeLLMRunner(),
        prompt_builder=FakePromptBuilder(),
    )
    history = ChatHistory(messages=[ChatMessage(role="assistant", content="hi")])
    try:
        list(orchestrator.chat(history))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "user message" in str(e).lower()


if __name__ == "__main__":
    test_chat_orchestrator_returns_tokens()
    test_chat_orchestrator_rejects_empty_history()
    test_chat_orchestrator_rejects_no_user_message()
    print("All tests passed!")
