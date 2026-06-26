import logging

from openai import OpenAI

from app.core.container import AppContainer, set_container
from app.core.logging_config import setup_logging
from app.utils.chat_orchestrator import ChatOrchestrator
from app.utils.config_loader import load_config
from app.utils.embedder import SentenceTransformerEmbedder, load_embedding_model
from app.utils.ingredient_translator import IngredientTranslator
from app.utils.intent_detector import IntentDetector
from app.utils.llm_model import LLMRunner
from app.utils.prompt_builder import PromptBuilder
from app.utils.recipe_repository import RecipeRepository
from app.utils.recipe_retriever import FAISSRecipeRetriever
from app.utils.recipe_search import RecipeSearch
from app.utils.recipe_translator import RecipeTranslator
from app.utils.sqlite_store import RecipeSQLiteStore
from app.utils.vector_index import VectorIndex

logger = logging.getLogger(__name__)


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

    recipe_repository = RecipeRepository(sqlite_store)

    vector_index = VectorIndex(
        index_dir=config["paths"]["faiss_index_dir"],
        nprobe=config.get("faiss", {}).get("nprobe", 16),
    )

    intent_detector = IntentDetector()

    llm_runner = LLMRunner(
        api_key=config["openrouter"]["api_key"],
        model=config["openrouter"]["model"],
    )

    fast_model = config.get("openrouter", {}).get(
        "fast_model", "meta-llama/llama-3.1-8b-instruct"
    )
    translator_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config["openrouter"]["api_key"],
    )
    ingredient_translator = IngredientTranslator(
        client=translator_client,
        model=fast_model,
    )
    logging.info("IngredientTranslator initialized with model=%s", fast_model)

    recipe_translator = RecipeTranslator(
        client=translator_client,
        model=fast_model,
    )
    logging.info("RecipeTranslator initialized with model=%s", fast_model)

    embedder = SentenceTransformerEmbedder(embedding_model)
    retriever = FAISSRecipeRetriever(
        vector_index=vector_index,
        recipe_repository=recipe_repository,
    )
    recipe_search = RecipeSearch(
        embedder=embedder,
        intent_detector=intent_detector,
        retriever=retriever,
    )
    logging.info("RecipeSearch initialized")

    prompt_builder = PromptBuilder()

    chat_orchestrator = ChatOrchestrator(
        recipe_search=recipe_search,
        intent_detector=intent_detector,
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
        vector_index=vector_index,
        recipe_repository=recipe_repository,
        chat_orchestrator=chat_orchestrator,
        ingredient_translator=ingredient_translator,
        recipe_translator=recipe_translator,
    )
    set_container(container)
    logging.info("AppContainer initialized and set")
