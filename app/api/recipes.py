import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List

from app.core.container import AppContainer, get_container
from app.core.intent import Intent
from app.core.models import Recipe
from app.utils.ingredient_search_service import IngredientQueryItem

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

router = APIRouter(dependencies=[Security(api_key_header)])


class QueryItem(BaseModel):
    text: str
    generalize: bool = True


class SearchRequest(BaseModel):
    intent: Intent
    query: List[QueryItem]
    top_k: int = 5


class SearchResponse(BaseModel):
    results: List[Recipe]


@router.post("/search", response_model=SearchResponse)
def search_recipes(
    request: SearchRequest, container: AppContainer = Depends(get_container)
):
    if request.intent != Intent.INGREDIENT_SEARCH:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported intent: {request.intent.value}. Only 'ingredient_search' is supported.",
        )

    t_start = time.perf_counter()

    items = [
        IngredientQueryItem(text=item.text, generalize=item.generalize)
        for item in request.query
    ]

    results = container.ingredient_search_service.search(
        items=items,
        intent=request.intent,
        top_k=request.top_k,
    )

    t_total = time.perf_counter() - t_start
    logger.debug("search_endpoint total_time_ms=%.2f", t_total * 1000)
    return SearchResponse(results=results)
