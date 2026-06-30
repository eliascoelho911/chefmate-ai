# Chefmate AI – Agent Notes

## Project Layout

- The repository root **is** the backend. The README references a `backend/` folder, but all code and configs live at the repo root (`main.py`, `app/`, `requirements.txt`, etc.).
- FastAPI entrypoint: `main.py`.
- Router: `app/api/recipes.py` (`/recipes`).
- Core runtime logic lives in `app/utils/` (embeddings, FAISS, LLM, prompts, intent detection, recipe search).
- Data-preparation logic lives in `scripts/data_prep/` (recipe preprocessing, embedding generation, FAISS index building). These modules are only consumed by `scripts/prepare_data.py`.

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
     model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

### Option A: Standalone script (recommended)

```bash
# After placing the raw CSV in data/raw/recipes.csv
python scripts/prepare_data.py
```

This script cleans the CSV, generates sentence-transformer embeddings, serializes a pickle, and builds three FAISS indexes (`ingredients_embedding`, `ingredients_with_quantities_embedding`, `title_embedding`). The resulting artifacts are written to `data/processed/` and `data/indexes/`.

**Note:** The server will fail on startup if the cleaned recipe pickle or the FAISS indexes are missing. Data preparation must be run manually beforehand (or artifacts transferred into the container/volume).

## Known Code Quirks

- ~~The preprocessor (`app/utils/recipe_preprocessor.py`) does **not** create the `faiss_index` column required by `FAISSHandler.__init__` (`df.set_index("faiss_index")`).~~ **Fixed**: `clean_recipe_data()` now adds `df['faiss_index'] = df.index` at the end. (Module relocated to `scripts/data_prep/recipe_preprocessor.py`.)

## Testing

### Automated Integration Tests — IngredientTranslator

The repo now contains **real integration tests** for `app/utils/ingredient_translator.py` that hit the live OpenRouter API. They are located at:

```
tests/test_ingredient_translator.py
```

#### What the tests cover

| Suite | Description |
|-------|-------------|
| `TestSimpleIngredients` | Direct translations (e.g. "frango" → "chicken"). |
| `TestNormalization` | Generalized translations when `generalize=True` (e.g. "peito de frango" → "chicken", not "chicken breast"). |
| `TestLiteralTranslation` | Specific translations when `generalize=False` (e.g. "peito de frango" → "chicken breast"). |
| `TestBatchBehavior` | Deduplication, order preservation, and English pass-through. |
| `TestEdgeCases` | Empty lists, whitespace-only items. |

Each test run creates a **unique log file** under `tests/logs/` (timestamped) with a simplified, one-line-per-test format showing input, output, elapsed time, parse success, and validity:

```
2026-06-27 14:18:12 | [simple_frango] parse=OK valid=OK | in=frango | out=chicken | 1205ms | notes=
```

#### Running the tests

```bash
# 1. Ensure the virtual environment is active and dependencies are installed
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt pytest

# 2. The tests require a valid OPENROUTER_API_KEY in your .env
#    (they will skip automatically if the key is missing)

# 3. Run all translator tests
pytest tests/test_ingredient_translator.py -v

# 4. Run only simple ingredient tests
pytest tests/test_ingredient_translator.py -v -k "TestSimpleIngredients"

# 5. Run only normalization (generalization) tests
pytest tests/test_ingredient_translator.py -v -k "TestNormalization"
```

#### Interpreting results

- **`parse=OK`** — The LLM returned valid JSON and no internal warning/fallback was triggered.
- **`parse=FAIL`** — The response was truncated, malformed, or an exception occurred; the translator fell back to returning the original term.
- **`valid=OK`** — The translated output matched the test assertion (substring check or exact match).
- **`valid=FAIL`** — The assertion failed (e.g. "peito de frango" returned "chicken breast" when `generalize=True` was expected).

### Manual smoke test for `/recipes/search`

```bash
curl -X POST http://localhost:8000/recipes/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CHEFMATE_API_KEY" \
  --data-raw '{"intent":"ingredient_search","query":["chicken","rice","broccoli"]}'
```

## Startup Dependency Chain

`main.py` -> `init_dependencies()` (in `app/core/startup.py`) eagerly loads, in order:
1. Configuration (`.env` → legacy `config.yml`)
2. SentenceTransformer model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
3. Cleaned recipe DataFrame pickle
4. FAISS indexes
5. Intent detector
6. OpenRouter API client (lightweight — no local model download)

If any step fails, the app will not start.

## Docker & Containerization

The app is fully dockerized. See the `Dockerfile`, `docker-compose.yml`, and `entrypoint.sh` in the repo root.

### Key container behavior
- The container image creates a non-root user (`appuser`, UID 1001).
- **On boot**, the container starts as **root** so that `entrypoint.sh` can create directories inside the Docker volume (e.g. `data/cache`) and `chown` them to `appuser`. Immediately after fixing permissions, it drops to `appuser` via `gosu` before starting Uvicorn.
- `entrypoint.sh` is **root-owned** (`chmod 755`) so `appuser` cannot tamper with it. This prevents a compromised app process from modifying the entrypoint and escalating privileges on the next restart.
- On boot, `entrypoint.sh` checks for the cleaned recipe pickle and FAISS indexes.
  - If missing, the container **exits with an error** — it does NOT run data preparation automatically.
  - If present, it starts Uvicorn immediately.
- The `data/` directory is persisted via a named Docker volume (`chefmate_data`).
- All configuration is injected through environment variables (`.env` file mounted by `docker-compose.yml`). The image does **not** contain any API keys or secrets.
- To populate data inside the container, either:
  1. Run `docker compose exec chefmate python scripts/prepare_data.py` after the first failed start (with raw CSV available), or
  2. Transfer pre-built artifacts from your local machine into the volume (see README).

### Production constraints
The `docker-compose.yml` enforces resource limits matching the target hardware:
- CPU limit: 1.0
- Memory limit: 4G
- Reservations: 0.5 CPU / 2G RAM

## Environment Notes

- Python 3.10+ required.
- Prompt truncation is word-count based (`len(tokens) > context_length - 512`), not actual token count.
- Authentication is enforced via `X-API-Key` header (see `CHEFMATE_API_KEY` env var). The healthcheck endpoint is exempt.
- CORS is disabled; the service is designed for server-to-server calls only.
