from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from app.core.container import AppContainer, get_container
from app.core.models import Recipe

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    results: List[Recipe]


@router.post("/search", response_model=SearchResponse)
def search_recipes(
    request: SearchRequest, container: AppContainer = Depends(get_container)
):
    results = container.recipe_search.search(request.query, top_k=request.top_k)
    return SearchResponse(results=results)
