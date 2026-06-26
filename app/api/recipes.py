from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from app.core.container import AppContainer, get_container
from app.core.intent import Intent
from app.core.models import Recipe
from app.utils.ingredient_translator import TranslationError
from app.utils.recipe_translator import RecipeTranslationError

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
    results = container.recipe_search.search(request.query, top_k=request.top_k)
    return SearchResponse(results=results)


@router.post("/suggest-by-ingredients", response_model=SuggestByIngredientsResponse)
def suggest_by_ingredients(
    request: SuggestByIngredientsRequest,
    container: AppContainer = Depends(get_container),
):
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

    try:
        translated = container.ingredient_translator.translate_batch(required)
    except TranslationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    results = container.recipe_search.search(
        query,
        intent=Intent.INGREDIENT_SEARCH,
        top_k=request.top_k,
        required_ingredients=translated,
    )

    try:
        results = container.recipe_translator.translate_recipes(results)
    except RecipeTranslationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return SuggestByIngredientsResponse(results=results)
