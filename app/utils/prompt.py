from typing import List, Dict, Any
import re


def _format_recipe_as_markdown(recipe: Any) -> str:
    """Format a recipe dict/metadata as readable Markdown text."""
    if isinstance(recipe, str):
        return recipe

    if not isinstance(recipe, dict):
        return str(recipe)

    name = recipe.get("name", "Unnamed Recipe")
    ingredients = recipe.get("ingredients_with_quantities", recipe.get("ingredients_cleaned", []))
    instructions = recipe.get("recipe_instructions", [])
    category = recipe.get("category", recipe.get("recipe_category", ""))
    calories = recipe.get("calories", "")
    total_time = recipe.get("total_time", "")
    rating = recipe.get("rating", recipe.get("aggregated_rating", None))
    images = recipe.get("images", [])

    lines = [f"### {name}"]

    if category:
        lines.append(f"**Category:** {category}")
    if calories:
        lines.append(f"**Calories:** {calories}")
    if total_time:
        lines.append(f"**Total Time:** {total_time}")
    if rating is not None and rating != "":
        lines.append(f"**Rating:** {rating}")

    # Images
    if images and isinstance(images, list) and len(images) > 0:
        first_img = images[0] if images[0] and str(images[0]).strip() else None
        if first_img:
            lines.append(f"![{name}]({first_img})")

    # Ingredients
    if ingredients:
        lines.append("\n**Ingredients:**")
        if isinstance(ingredients, list):
            for ing in ingredients:
                lines.append(f"- {ing}")
        else:
            lines.append(f"- {ingredients}")

    # Instructions
    if instructions:
        lines.append("\n**Instructions:**")
        if isinstance(instructions, list):
            for step in instructions:
                lines.append(f"- {step}")
        else:
            lines.append(f"{instructions}")

    return "\n".join(lines)


def build_chat_messages(system_prompt: str, retrieved_chunks: list, chat_history: list) -> list:
    """
    Constrói uma lista de mensagens no formato da API Chat Completions (OpenRouter/OpenAI).
    """
    if retrieved_chunks:
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks):
            md = _format_recipe_as_markdown(chunk)
            formatted_chunks.append(f"Recipe {i+1}:\n{md}")
        context_block = "\n\n".join(formatted_chunks)
        context_message = f"[Context Retrieved from Knowledge Base]\n{context_block}\n"
    else:
        context_message = "[No context retrieved. Try to respond based on the chat history or ask the user for clarification.]\n"

    system_content = (
        f"{system_prompt}\n\n"
        f"{context_message}\n"
        f"Always format using proper Markdown. Keep answers concise and helpful. "
        f"Strictly avoid responses stating: 'Based on the knowledge base', or similar phrases."
    )

    messages = [{"role": "system", "content": system_content}]

    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role and content:
            messages.append({"role": role, "content": content})

    return messages

def construct_prompt(system_prompt: str, retrieved_chunks: list, chat_history: list, latest_user_message: str) -> str:
    """
    Constructs a complete prompt for the language model by combining:
    - a system prompt
    - context retrieved from vector DB
    - chat history
    - latest user query
    """
    # Formatting retrieved chunks
    if retrieved_chunks:
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks):
            md = _format_recipe_as_markdown(chunk)
            formatted_chunks.append(f"Recipe {i+1}:\n{md}")
        context_block = "\n\n".join(formatted_chunks)
        context_message = f"[Context Retrieved from Knowledge Base]\n{context_block}\n"
    else:
        context_message = "[No context retrieved. Try to respond based on the chat history or ask the user for clarification.]\n"

    # Formatting chat history
    formatted_history = ""
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        formatted_history += f"{role.capitalize()}: {content}\n"

    # Combining all parts
    prompt = (
        f"{system_prompt}\n"
        f"{context_message}"
        f"[Conversation History]\n"
        f"{formatted_history}\n"
        f"The user asked: {latest_user_message}\n"
        f"Assiatnce: You are the assistant. Please respond accordingly."
    )
    return prompt

def generate_system_prompt(user_message: str) -> str:
    INTENT_PATTERNS = {
        "SuggestRecipe": [
            r"\b(suggest|recommend|idea|give me|show|find|any)\b.*\b(recipes?|dishes?|meals?)\b",
            r"\b(what can i make|cook|prepare)\b.*\b(with|using)\b.*",
            r"\b(available ingredients?|leftovers?|at home)\b",
            r"\b(good|easy|quick|simple|healthy).*\brecipes?\b",
            r"\b(dinner|lunch|breakfast|snack).*ideas?\b",
            # Portuguese
            r"\b(sugira|sugere|recomende|recomenda|me dê|dê|mostre|encontre|alguma)\b.*\b(receitas?|pratos?|refeições?)\b",
            r"\b(o que posso (fazer|cozinhar|preparar))\b.*\b(com|usando)\b.*",
            r"\b(ingredientes disponiveis?|sobras?|em casa)\b",
            r"\b(boa|boa|fácil|facil|rapido|rápido|simples|saudavel|saudável).*\breceitas?\b",
            r"\b(jantar|almoço|almoco|cafe da manha|cafe da manhã|lanche).*\b(ideias?|sugestões?|sugestoes?)\b",
        ],
        "IngredientQuery": [
            r"\b(ingredients?|need(ed)?|require|contain|consist of)\b",
            r"\b(do i need|what do i need|is it made of)\b.*",
            # Portuguese
            r"\b(ingredientes?|preciso|necessito|contém|contem|consiste em)\b",
            r"\b(preciso de|o que preciso|do que é feito)\b.*",
        ],
        "InstructionsOnly": [
            r"\b(how to|steps to|prepare|make|cook|method|instruction(s)?)\b.*",
            r"\bprocedure\b",
            # Portuguese
            r"\b(como (fazer|preparar)|passos para|modo de preparo|instrucoes?|instruções?)\b.*",
            r"\bprocedimento\b",
        ],
        "NutritionInfo": [
            r"\b(calories|nutritional|health(y)?|macro|carbs|protein)\b",
            # Portuguese
            r"\b(calorias|nutricional|nutritivo|saudavel|saudável|macro|carboidratos|proteina|proteína)\b",
        ],
        "CookingTimeFilter": [
            r"\b(time|required|cook(ing)? time|under \d{1,3} (mins?|minutes?))\b",
            r"\bquick|fast|30 min\b",
            # Portuguese
            r"\b(tempo|tempo de cozimento|em menos de \d{1,3} (min|minutos?))\b",
            r"\brapido|rápido|rapida|rápida|30 min\b",
        ],
        "DietaryPreferences": [
            r"\b(vegetarian|vegan|gluten[- ]?free|dairy[- ]?free|low carb|low fat|keto|paleo)\b",
            # Portuguese
            r"\b(vegetariano|vegano|sem gluten|sem glúten|sem lactose|low carb|baixa carb|baixa gordura|keto|paleo)\b",
        ],
        "ExpandRecipe": [
            r"\b(more details|elaborate|explain more|show full|tell me more)\b",
            # Portuguese
            r"\b(mais detalhes|detalhe|explique mais|mostre completo|me fale mais|fale mais)\b",
        ],
        "ToolOrMethodQuery": [
            r"\b(do i need|how to use|can i use|tool(s)?|equipment|machine|oven|grill|stove|microwave)\b",
            # Portuguese
            r"\b(preciso de|como usar|posso usar|ferramenta|equipamento|maquina|máquina|forno|churrasqueira|fogao|fogão|microondas)\b",
        ],
    }

    intent_addons = {
        "SuggestRecipe": """
When suggesting recipes:
- Provide 2 to 3 options in Markdown.
- Include name, category, calories, cook time (e.g., "1 hour 30 minutes", not 01:30), and rating - strictly each on a new line with labels, and properly formatted
- Immediately after the name, include a Markdown image (if available from the context retrieved).
- Brief list of ingredients [sub-bulleted list or comma separated].
- There is no need to include instructions.
""",
        "IngredientQuery": """
For ingredient questions:
- Use bullet points with quantities.
- Only include relevant ingredients.
""",
        "InstructionsOnly": """
When explaining instructions steps:
- Use a numbered list.
- Avoid adding unrelated commentary.
""",
        "NutritionInfo": """
For nutrition questions:
- Mention calories, macros, and diet types (if known).
- Use a clean, bullet-style summary.
""",
        "CookingTimeFilter": """
For time-based requests:
- Suggest recipes with matching or under X cook time.
- Clearly show total cooking time (e.g., "1 hour 30 minutes", not 01:30).
""",
        "DietaryPreferences": """
Respect dietary preferences like vegan, gluten-free, etc.
- Do not suggest recipes with restricted ingredients.
""",
        "ExpandRecipe": """
If elaborating on a recipe:
- Include full details (name, image, ingredients, instructions, calories, rating, total time).
- Numbered steps for instructions.
""",
        "ToolOrMethodQuery": """
For tool/method questions:
- Briefly explain tool usage.
- Offer alternatives if applicable.
"""
    }

    base_prompt = """
You are a helpful, friendly AI cooking assistant. Always:
- Format response using proper Markdown syntax [e.g., Use **bold** for key terms].
- Ask clarifying questions if anything is ambiguous.
- Respond in the same language the user is writing (e.g., Portuguese, English, Spanish, etc.).
Strictly avoid responses stating: "Based on the knowledge base", or similar phrases.
Keep answers concise and helpful.
"""

    # Normalize input
    user_message = user_message.lower()

    # Detecting intents
    matched_intents = []
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(pattern, user_message) for pattern in patterns):
            matched_intents.append(intent)

    # Building final prompt
    full_prompt = base_prompt
    for intent in matched_intents:
        full_prompt += intent_addons.get(intent, "")

    return full_prompt.strip()