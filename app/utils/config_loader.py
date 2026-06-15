import os
from dotenv import load_dotenv

# Load .env file if present (helpful for local development outside Docker)
load_dotenv()


def load_config(config_file: str = "config.yml") -> dict:
    """
    Build configuration dict from environment variables.
    Falls back to sensible defaults for all paths and model settings.
    The only required variable is OPENROUTER_API_KEY.
    """
    config = {
        "paths": {
            "recipe_data": os.getenv("RECIPE_DATA", "data/raw/recipes.csv"),
            "cleaned_data_csv": os.getenv("CLEANED_DATA_CSV", "data/processed/cleaned_recipes.csv"),
            "cleaned_data_pkl": os.getenv("CLEANED_DATA_PKL", "data/processed/cleaned_recipes.pkl"),
            "faiss_index_dir": os.getenv("FAISS_INDEX_DIR", "data/indexes"),
        },
        "embedding": {
            "model_name": os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "128")),
        },
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-5.4-mini"),
        },
    }

    return config
