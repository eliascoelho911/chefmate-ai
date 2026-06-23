import logging

from app.core.container import AppContainer, set_container
from app.core.logging_config import setup_logging
from app.utils.chat_orchestrator import ChatOrchestrator
from app.utils.config_loader import load_config
from app.utils.embedder import SentenceTransformerEmbedder, load_embedding_model
from app.utils.faiss_handler import FAISSHandler
from app.utils.intent_detector import IntentDetector
from app.utils.llm_model import LLMRunner
from app.utils.prompt_builder import PromptBuilder
from app.utils.recipe_retriever import FAISSRecipeRetriever
from app.utils.recipe_search import RecipeSearch
from app.utils.sqlite_store import RecipeSQLiteStore


def init_dependencies():
    """Build the AppContainer and expose it via get_container()."""
    config = load_config()
    setup_logging(config["logging"]["level"])
    logging.info("Logging initialized")

    embedding_model = load_embedding_model(config)

    db_path = config["paths"]["sqlite_db"]
    sqlite_store = RecipeSQLiteStore(db_path)
    logging.info(
        "SQLite store loaded from %s (%d recipes)", db_path, sqlite_store.count()
    )

    faiss_handler = FAISSHandler(config, sqlite_store)

    intent_detector = IntentDetector()

    llm_runner = LLMRunner()

    embedder = SentenceTransformerEmbedder(embedding_model)
    retriever = FAISSRecipeRetriever(faiss_handler)
    recipe_search = RecipeSearch(
        embedder=embedder,
        intent_detector=intent_detector,
        retriever=retriever,
    )
    logging.info("RecipeSearch initialized")

    prompt_builder = PromptBuilder()

    chat_orchestrator = ChatOrchestrator(
        recipe_search=recipe_search,
        llm_runner=llm_runner,
        prompt_builder=prompt_builder,
    )
    logging.info("ChatOrchestrator initialized")

    container = AppContainer(
        config=config,
        recipe_search=recipe_search,
        llm_runner=llm_runner,
        intent_detector=intent_detector,
        embedder=embedder,
        faiss_handler=faiss_handler,
        chat_orchestrator=chat_orchestrator,
    )
    set_container(container)
    logging.info("AppContainer initialized and set")
