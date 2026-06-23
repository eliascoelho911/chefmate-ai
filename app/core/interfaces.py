from typing import Protocol, List
from app.core.models import Recipe


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


class IntentDetector(Protocol):
    def detect(self, user_input: str) -> str: ...


class RecipeRetriever(Protocol):
    def retrieve(
        self, query_embedding: List[float], intent: str, top_k: int
    ) -> List[Recipe]: ...
