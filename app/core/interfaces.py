from typing import Iterator, Protocol, List
from app.core.models import ChatHistory, Recipe


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


class IntentDetector(Protocol):
    def detect(self, user_input: str) -> str: ...


class RecipeRetriever(Protocol):
    def retrieve(
        self, query_embedding: List[float], intent: str, top_k: int
    ) -> List[Recipe]: ...


class LLMRunner(Protocol):
    def stream_response(self, messages: list) -> Iterator[str]: ...


class PromptBuilder(Protocol):
    def build_messages(
        self, chat_history: ChatHistory, intent: str, recipes: List[Recipe]
    ) -> List[dict]: ...
