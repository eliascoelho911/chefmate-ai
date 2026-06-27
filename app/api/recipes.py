import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from app.core.container import AppContainer, get_container
from app.core.intent import Intent
from app.core.models import Recipe
from app.utils.ingredient_translator import TranslationError
from app.utils.recipe_translator import RecipeTranslationError

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

router = APIRouter(dependencies=[Security(api_key_header)])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    results: List[Recipe]


class SuggestByIngredientsRequest(BaseModel):
    proteinas: List[str]
    carboidratos: List[str]
    legumes: List[str]
    top_k: int = 5


class SuggestByIngredientsResponse(BaseModel):
    results: List[Recipe]


@router.post("/search", response_model=SearchResponse)
def search_recipes(
    request: SearchRequest, container: AppContainer = Depends(get_container)
):
    t_start = time.perf_counter()
    results = container.recipe_search.search(request.query, top_k=request.top_k)
    t_total = time.perf_counter() - t_start
    logger.debug("search_endpoint total_time_ms=%.2f", t_total * 1000)
    return SearchResponse(results=results)


@router.post("/suggest-by-ingredients", response_model=SuggestByIngredientsResponse)
def suggest_by_ingredients(
    request: SuggestByIngredientsRequest,
    container: AppContainer = Depends(get_container),
):
    t_start = time.perf_counter()

    parts = []
    if request.proteinas:
        parts.append(", ".join(request.proteinas))
    if request.carboidratos:
        parts.append(", ".join(request.carboidratos))
    if request.legumes:
        parts.append(", ".join(request.legumes))

    query = "recipes with " + ", ".join(parts) if parts else "recipes"

    # Gather all required ingredients for strict filtering
    required = []
    if request.proteinas:
        required.extend(request.proteinas)
    if request.carboidratos:
        required.extend(request.carboidratos)
    if request.legumes:
        required.extend(request.legumes)

    t_ingredient_translation_start = time.perf_counter()
    try:
        translated = container.ingredient_translator.translate_batch(required)
    except TranslationError as exc:
        logger.warning("Ingredient translation failed, using original terms: %s", exc)
        translated = required
    t_ingredient_translation_elapsed = (
        time.perf_counter() - t_ingredient_translation_start
    )
    logger.debug(
        "ingredient_translation_time_ms=%.2f",
        t_ingredient_translation_elapsed * 1000,
    )

    t_search_start = time.perf_counter()
    results = container.recipe_search.search(
        query,
        intent=Intent.INGREDIENT_SEARCH,
        top_k=request.top_k,
        required_ingredients=translated,
    )
    t_search_elapsed = time.perf_counter() - t_search_start
    logger.debug("recipe_search_time_ms=%.2f", t_search_elapsed * 1000)

    t_recipe_translation_start = time.perf_counter()
    try:
        results = container.recipe_translator.translate_recipes(results)
    except RecipeTranslationError as exc:
        logger.warning("Recipe translation failed, returning English recipes: %s", exc)
    t_recipe_translation_elapsed = time.perf_counter() - t_recipe_translation_start
    logger.debug(
        "recipe_translation_time_ms=%.2f",
        t_recipe_translation_elapsed * 1000,
    )

    t_total = time.perf_counter() - t_start
    logger.debug(
        "suggest_by_ingredients_endpoint total_time_ms=%.2f",
        t_total * 1000,
    )
    return SuggestByIngredientsResponse(results=results)
