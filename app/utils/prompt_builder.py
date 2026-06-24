from typing import List, Optional

from app.core.intent import Intent
from app.core.models import ChatHistory, Recipe
from app.utils.prompt import _format_recipe_as_markdown


class PromptBuilder:
    """
    Constrói mensagens no formato Chat Completions a partir de:
    - histórico de chat tipado
    - intent detectada
    - receitas recuperadas
    """

    _BASE_PROMPT = (
        "You are a helpful, friendly AI cooking assistant. Always:\n"
        "- Format response using proper Markdown syntax [e.g., Use **bold** for key terms].\n"
        "- Ask clarifying questions if anything is ambiguous.\n"
        "- Respond in the same language the user is writing (e.g., Portuguese, English, Spanish, etc.).\n"
        'Strictly avoid responses stating: "Based on the knowledge base", or similar phrases.\n'
        "Keep answers concise and helpful.\n"
    )

    _INTENT_ADDONS = {
        "SuggestRecipe": (
            "When suggesting recipes:\n"
            "- Provide 2 to 3 options in Markdown.\n"
            '- Include name, category, calories, cook time (e.g., "1 hour 30 minutes", not 01:30), and rating - strictly each on a new line with labels, and properly formatted\n'
            "- Immediately after the name, include a Markdown image (if available from the context retrieved).\n"
            "- Brief list of ingredients [sub-bulleted list or comma separated].\n"
            "- There is no need to include instructions.\n"
        ),
        "IngredientQuery": (
            "For ingredient questions:\n"
            "- Use bullet points with quantities.\n"
            "- Only include relevant ingredients.\n"
        ),
        "InstructionsOnly": (
            "When explaining instructions steps:\n"
            "- Use a numbered list.\n"
            "- Avoid adding unrelated commentary.\n"
        ),
        "NutritionInfo": (
            "For nutrition questions:\n"
            "- Mention calories, macros, and diet types (if known).\n"
            "- Use a clean, bullet-style summary.\n"
        ),
        "CookingTimeFilter": (
            "For time-based requests:\n"
            "- Suggest recipes with matching or under X cook time.\n"
            '- Clearly show total cooking time (e.g., "1 hour 30 minutes", not 01:30).\n'
        ),
        "DietaryPreferences": (
            "Respect dietary preferences like vegan, gluten-free, etc.\n"
            "- Do not suggest recipes with restricted ingredients.\n"
        ),
        "ExpandRecipe": (
            "If elaborating on a recipe:\n"
            "- Include full details (name, image, ingredients, instructions, calories, rating, total time).\n"
            "- Numbered steps for instructions.\n"
        ),
        "ToolOrMethodQuery": (
            "For tool/method questions:\n"
            "- Briefly explain tool usage.\n"
            "- Offer alternatives if applicable.\n"
        ),
    }

    _DETECTED_INTENT_TO_ADDON: dict[Intent, Optional[str]] = {
        Intent.INGREDIENT_SEARCH: "SuggestRecipe",
        Intent.SPECIFIC_RECIPE: "ExpandRecipe",
        Intent.RECIPE_GENERATION: "SuggestRecipe",
        Intent.STEP_NAVIGATION: "InstructionsOnly",
        Intent.DIET_FILTER: "DietaryPreferences",
        Intent.NUTRITION_INFO: "NutritionInfo",
        Intent.TIME_FILTER: "CookingTimeFilter",
        Intent.RATING_FILTER: "SuggestRecipe",
        Intent.UNCLEAR: None,
    }

    def build_messages(
        self, chat_history: ChatHistory, intent: Intent, recipes: List[Recipe]
    ) -> List[dict]:
        system_prompt = self._build_system_prompt(intent)
        context_message = self._build_context_message(recipes)

        system_content = (
            f"{system_prompt}\n\n"
            f"{context_message}\n"
            "Always format using proper Markdown. Keep answers concise and helpful. "
            "Strictly avoid responses stating: 'Based on the knowledge base', or similar phrases."
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in chat_history.messages:
            if msg.role and msg.content:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def _build_system_prompt(self, intent: Intent) -> str:
        addon_key = self._DETECTED_INTENT_TO_ADDON.get(intent)
        addon = self._INTENT_ADDONS.get(addon_key, "") if addon_key else ""
        return f"{self._BASE_PROMPT}\n{addon}".strip()

    def _build_context_message(self, recipes: List[Recipe]) -> str:
        if not recipes:
            return "[No context retrieved. Try to respond based on the chat history or ask the user for clarification.]\n"

        formatted_chunks = []
        for i, recipe in enumerate(recipes):
            md = _format_recipe_as_markdown(recipe.model_dump())
            formatted_chunks.append(f"Recipe {i + 1}:\n{md}")
        context_block = "\n\n".join(formatted_chunks)
        return f"[Context Retrieved from Knowledge Base]\n{context_block}\n"
