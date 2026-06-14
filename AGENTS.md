# Chefmate AI – Agent Notes

## Project Layout

- The repository root **is** the backend. The README references a `backend/` folder, but all code and configs live at the repo root (`main.py`, `app/`, `requirements.txt`, etc.).
- FastAPI entrypoint: `main.py`.
- Routers: `app/api/chat.py` (`/chat`) and `app/api/data_preparation.py` (`/data`).
- Core logic lives in `app/utils/` (embeddings, FAISS, LLM, prompts, preprocessing, intent detection).

## Required Setup (Not in Repo)

These files are gitignored and **must be created/downloaded manually** before the app will start:

1. **`.env`** at repo root (preferred configuration method). Copy from `.env.example` and fill in at least:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-...
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_key
   ```
   All paths, model names, and hyperparameters can also be set via environment variables. See `.env.example` for the full list.

2. **Legacy `config.yml`** (optional — still supported for local dev, but `.env` is recommended). Example:
   ```yaml
   paths:
     recipe_data: "data/raw/recipes.csv"
     cleaned_data_csv: "data/processed/cleaned_recipes.csv"
     cleaned_data_pkl: "data/processed/cleaned_recipes.pkl"
     faiss_index_dir: "data/indexes"
   embedding:
     model_name: "all-MiniLM-L6-v2"
     batch_size: 128
   openrouter:
     api_key: "sk-or-v1-..."
     model: "openai/gpt-5.4-mini"
   ```

3. **Raw recipe CSV**: Download from Kaggle and place at `data/raw/recipes.csv` (or override via `RECIPE_DATA` env var).
   - **Recommended dataset**: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data?select=recipes.csv) (linked in README).
   - The CSV **must contain** these columns (PascalCase; they are auto-converted to snake_case by the preprocessor):
     - `Name`, `RecipeIngredientParts`, `RecipeInstructions`, `RecipeIngredientQuantities`
     - `Keywords`, `DatePublished`, `ReviewCount`, `AggregatedRating`
     - `CookTime`, `PrepTime`, `TotalTime`, `RecipeCategory`, `Calories`, `RecipeYield`, `Images`.

## Data Preparation Flow

Data prep can be run in two ways:

### Option A: Standalone script (recommended — avoids the chicken-and-egg problem)

```bash
# After placing the raw CSV in data/raw/recipes.csv
python scripts/prepare_data.py
```

This script cleans the CSV, generates sentence-transformer embeddings, serializes a pickle, and builds three FAISS indexes (`ingredients_embedding`, `ingredients_with_quantities_embedding`, `title_embedding`). The resulting artifacts are written to `data/processed/` and `data/indexes/`.

### Option B: Runtime API endpoint

```bash
# Start the server (only works if the pickle and indexes already exist)
uvicorn main:app --reload

# In another terminal:
curl -X POST http://localhost:8000/data/initialize-recipes
```

**Note**: The server will fail on startup if `config.yml`, the cleaned recipe pickle, or the FAISS indexes are missing. Use Option A for first-time setup.

## Known Code Quirks

- `app/api/data_preparation.py` line 2 imports from `backend.app.utils.recipe_preprocessor`. Because the repo root is the backend, this path is invalid. It should be `app.utils.recipe_preprocessor`. Hitting `/data/initialize-recipes` will raise an `ModuleNotFoundError` until this is fixed.
- ~~The preprocessor (`app/utils/recipe_preprocessor.py`) does **not** create the `faiss_index` column required by `FAISSHandler.__init__` (`df.set_index("faiss_index")`).~~ **Fixed**: `clean_recipe_data()` now adds `df['faiss_index'] = df.index` at the end.

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
1. Configuration (`.env` → legacy `config.yml`)
2. SentenceTransformer model (`all-MiniLM-L6-v2`)
3. Cleaned recipe DataFrame pickle
4. FAISS indexes
5. Intent detector
6. OpenRouter API client (lightweight — no local model download)

If any step fails, the app will not start.

## Docker & Containerization

The app is fully dockerized. See the `Dockerfile`, `docker-compose.yml`, and `entrypoint.sh` in the repo root.

### Key container behavior
- The container runs as a non-root user (`appuser`, UID 1001).
- On first boot, the `entrypoint.sh` checks for the cleaned recipe pickle and FAISS indexes.
  - If missing, it automatically runs `python scripts/prepare_data.py` (downloads Kaggle dataset if needed, generates embeddings, builds indexes).
  - If present, it skips directly to starting Uvicorn.
- The `data/` directory is persisted via a named Docker volume (`chefmate_data`), so data preparation only runs once.
- All configuration is injected through environment variables (`.env` file mounted by `docker-compose.yml`). The image does **not** contain any API keys or secrets.

### Production constraints
The `docker-compose.yml` enforces resource limits matching the target hardware:
- CPU limit: 1.0
- Memory limit: 4G
- Reservations: 0.5 CPU / 2G RAM

## Environment Notes

- Python 3.10+ required.
- Prompt truncation is word-count based (`len(tokens) > context_length - 512`), not actual token count.
- CORS is wide open (`allow_origins=["*"]`).
