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
    page: int = 1
    per_page: int = 10


class SearchResponse(BaseModel):
    results: List[Recipe]
    page: int
    per_page: int
    has_more: bool


@router.post("/search", response_model=SearchResponse)
def search_recipes(
    request: SearchRequest, container: AppContainer = Depends(get_container)
):
    if request.intent != Intent.INGREDIENT_SEARCH:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported intent: {request.intent.value}. Only 'ingredient_search' is supported.",
        )

    if request.page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if not (1 <= request.per_page <= 10):
        raise HTTPException(status_code=400, detail="per_page must be between 1 and 10")

    t_start = time.perf_counter()

    items = [
        IngredientQueryItem(text=item.text, generalize=item.generalize)
        for item in request.query
    ]

    # Fetch one extra result to determine has_more accurately.
    top_k = request.page * request.per_page + 1

    all_results = container.ingredient_search_service.search(
        items=items,
        intent=request.intent,
        top_k=top_k,
    )

    has_more = len(all_results) > request.page * request.per_page
    results = all_results[
        (request.page - 1) * request.per_page : request.page * request.per_page
    ]

    t_total = time.perf_counter() - t_start
    logger.debug("search_endpoint total_time_ms=%.2f", t_total * 1000)
    return SearchResponse(
        results=results,
        page=request.page,
        per_page=request.per_page,
        has_more=has_more,
    )
