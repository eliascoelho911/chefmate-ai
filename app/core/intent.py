from enum import Enum


class Intent(Enum):
    """
    Domain enumeration representing the user's goal in a conversation.

    Each intent describes what the user wants to achieve, not how the
    system should satisfy it.  The mapping from intent to concrete
    behaviour (FAISS index, prompt addon, etc.) lives in the orchestration
    layer, keeping this enum a pure vocabulary type.
    """

    INGREDIENT_SEARCH = "ingredient_search"
    SPECIFIC_RECIPE = "specific_recipe"
    RECIPE_GENERATION = "recipe_generation"
    STEP_NAVIGATION = "step_navigation"
    DIET_FILTER = "diet_filter"
    NUTRITION_INFO = "nutrition_info"
    TIME_FILTER = "time_filter"
    RATING_FILTER = "rating_filter"
    UNCLEAR = "unclear"
