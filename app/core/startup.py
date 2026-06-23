import logging
from typing import Any

from app.core.logging_config import setup_logging
from app.utils.config_loader import load_config
from app.utils.embedder import SentenceTransformerEmbedder, load_embedding_model
from app.utils.faiss_handler import FAISSHandler
from app.utils.intent_detector import IntentDetector
from app.utils.llm_model import LLMRunner
from app.utils.recipe_retriever import FAISSRecipeRetriever
from app.utils.recipe_search import RecipeSearch
from app.utils.sqlite_store import RecipeSQLiteStore


class GlobalState:
    config: Any = None
    embedding_model: Any = None
    sqlite_store: Any = None
    faiss_handler: Any = None
    intent_detector: Any = None
    llm_runner: Any = None
    recipe_search: Any = None


def init_dependencies():
    if GlobalState.config is None:
        GlobalState.config = load_config()
        setup_logging(GlobalState.config["logging"]["level"])
        logging.info("Logging initialized")

    if GlobalState.embedding_model is None:
        GlobalState.embedding_model = load_embedding_model(GlobalState.config)

    if GlobalState.sqlite_store is None:
        db_path = GlobalState.config["paths"]["sqlite_db"]
        GlobalState.sqlite_store = RecipeSQLiteStore(db_path)
        print(
            f"[Startup] SQLite store loaded from {db_path} ({GlobalState.sqlite_store.count()} recipes)"
        )

    if GlobalState.faiss_handler is None:
        GlobalState.faiss_handler = FAISSHandler(
            GlobalState.config, GlobalState.sqlite_store
        )

    if GlobalState.intent_detector is None:
        GlobalState.intent_detector = IntentDetector()

    if GlobalState.llm_runner is None:
        GlobalState.llm_runner = LLMRunner()

    if GlobalState.recipe_search is None:
        embedder = SentenceTransformerEmbedder(GlobalState.embedding_model)
        retriever = FAISSRecipeRetriever(GlobalState.faiss_handler)
        GlobalState.recipe_search = RecipeSearch(
            embedder=embedder,
            intent_detector=GlobalState.intent_detector,
            retriever=retriever,
        )
        print("[Startup] RecipeSearch initialized")
