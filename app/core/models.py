from typing import List, Optional
from pydantic import BaseModel


class Recipe(BaseModel):
    faiss_index: int
    name: str
    ingredients_with_quantities: List[str]
    recipe_instructions: List[str]
    category: str
    calories: str
    total_time: str
    rating: Optional[float] = None
    images: List[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    messages: List[ChatMessage]

    @property
    def latest_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""
