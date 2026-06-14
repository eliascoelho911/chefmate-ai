# Chefmate AI – Agent Notes

## Project Layout

- The repository root **is** the backend. The README references a `backend/` folder, but all code and configs live at the repo root (`main.py`, `app/`, `requirements.txt`, etc.).
- FastAPI entrypoint: `main.py`.
- Routers: `app/api/chat.py` (`/chat`) and `app/api/data_preparation.py` (`/data`).
- Core logic lives in `app/utils/` (embeddings, FAISS, LLM, prompts, preprocessing, intent detection).

## Required Setup (Not in Repo)

These files are gitignored and **must be created/downloaded manually** before the app will start:

1. **`config.yml`** at repo root (required for every module). Example:
   ```yaml
   paths:
     recipe_data: "data/raw/recipes.csv"
     cleaned_data_csv: "data/processed/cleaned_recipes.csv"
     cleaned_data_pkl: "data/processed/cleaned_recipes.pkl"
     faiss_index_dir: "data/indexes"
     model_path: "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
   embedding:
     model_name: "all-MiniLM-L6-v2"
     batch_size: 128
   ```
2. **GGUF model**: Download `mistral-7b-instruct-v0.2.Q5_K_M.gguf` from Hugging Face (TheBloke) and place at the path configured above.
3. **Raw recipe CSV**: Download the Kaggle "Food Recipes Dataset" (`recipes.csv`) and place at `data/raw/recipes.csv` (or whatever `config.yml` specifies).

## Running the App

```bash
# Install dependencies (heavy: torch, transformers, faiss-cpu, llama-cpp-python)
pip install -r requirements.txt

# Start the dev server (run from repo root)
uvicorn main:app --reload
```

The server will fail on startup if `config.yml`, the cleaned recipe pickle, or the FAISS indexes are missing.

## Data Preparation Flow

Data prep is exposed as a **runtime API endpoint**, not a standalone script:

```bash
# After starting the server and placing the raw CSV
curl -X POST http://localhost:8000/data/initialize-recipes
```

This endpoint cleans the CSV, generates sentence-transformer embeddings, serializes a pickle, and builds three FAISS indexes (`ingredients_embedding`, `ingredients_with_quantities_embedding`, `title_embedding`). The resulting artifacts are written to `data/processed/` and `data/indexes/`.

## Known Code Quirk

- `app/api/data_preparation.py` line 2 imports from `backend.app.utils.recipe_preprocessor`. Because the repo root is the backend, this path is invalid. It should be `app.utils.recipe_preprocessor`. Hitting `/data/initialize-recipes` will raise an `ModuleNotFoundError` until this is fixed.

## Testing

- There are **no automated tests** in the repo.
- Manual smoke test for `/chat`:
  ```bash
  curl -X POST http://localhost:8000/chat/ \
    -H "Content-Type: application/json" \
    --data-raw '{"chat_history":[{"role":"user","content":"What can I cook with flour, eggs, salt, onion and garlic"}]}'
  ```

## Startup Dependency Chain

`main.py` -> `init_dependencies()` (in `app/core/startup.py`) eagerly loads, in order:
1. `config.yml`
2. SentenceTransformer model (`all-MiniLM-L6-v2`)
3. Cleaned recipe DataFrame pickle
4. FAISS indexes
5. Intent detector
6. Llama-cpp GGUF model

If any step fails, the app will not start.

## Environment Notes

- Python 3.10+ required.
- `n_threads=8` is hardcoded in `app/utils/llm_model.py`.
- Prompt truncation is word-count based (`len(tokens) > context_length - 512`), not actual token count.
- CORS is wide open (`allow_origins=["*"]`).
