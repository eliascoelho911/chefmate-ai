from app.utils.config_loader import load_config
from app.utils.embedder import load_embedding_model
from app.utils.faiss_handler import FAISSHandler
from app.utils.sqlite_store import RecipeSQLiteStore
from app.utils.intent_detector import IntentDetector
from app.utils.llm_model import LLMRunner

class GlobalState:
    config = None
    embedding_model = None
    sqlite_store = None
    faiss_handler = None
    intent_detector = None
    llm_runner = None

def init_dependencies():
    if GlobalState.config is None:
        GlobalState.config = load_config()

    if GlobalState.embedding_model is None:
        GlobalState.embedding_model = load_embedding_model(GlobalState.config)

    if GlobalState.sqlite_store is None:
        db_path = GlobalState.config["paths"]["sqlite_db"]
        GlobalState.sqlite_store = RecipeSQLiteStore(db_path)
        print(f"[Startup] SQLite store loaded from {db_path} ({GlobalState.sqlite_store.count()} recipes)")

    if GlobalState.faiss_handler is None:
        GlobalState.faiss_handler = FAISSHandler(GlobalState.config, GlobalState.sqlite_store)

    if GlobalState.intent_detector is None:
        GlobalState.intent_detector = IntentDetector()

    if GlobalState.llm_runner is None:
        GlobalState.llm_runner = LLMRunner()
