"""
Standalone script to prepare recipe data without needing the FastAPI server running.
Run this after placing recipes.csv in data/raw/ or let it auto-download from Kaggle.
"""
import sys
import os
import shutil

# Ensure repo root is on path so `app.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config_loader import load_config
from app.utils.recipe_preprocessor import load_recipe_data, clean_recipe_data
from app.utils.embedder import generate_recipe_embeddings
from app.utils.faiss_handler import build_recipe_faiss_indexes


def download_raw_dataset(raw_path: str) -> None:
    """Download the raw recipe CSV from Kaggle if it doesn't already exist."""
    if os.path.exists(raw_path):
        print(f"[INFO] Raw recipe CSV already exists at: {raw_path}")
        return

    if not (os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        print("[WARNING] No Kaggle credentials found.")
        print("          Set one of the following before running this script:")
        print('          export KAGGLE_API_TOKEN="<your-token>"')
        print("          OR")
        print('          export KAGGLE_USERNAME="<your-username>"')
        print('          export KAGGLE_KEY="<your-key>"')
        print("          Attempting download anyway (may use cached credentials)...")

    print("[INFO] Downloading dataset from Kaggle (irkaal/foodcom-recipes-and-reviews)...")
    try:
        import kagglehub
    except ImportError as exc:
        print("[ERROR] kagglehub is not installed. Install it with: pip install kagglehub")
        raise exc

    dataset_dir = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
    print(f"[INFO] Dataset downloaded to: {dataset_dir}")

    # Find recipes.csv inside the downloaded directory
    csv_candidates = [
        os.path.join(dataset_dir, "recipes.csv"),
        os.path.join(dataset_dir, "recipes.csv.zip"),
    ]
    # Also search recursively in case the archive unpacks to a subfolder
    for root, _dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower() == "recipes.csv":
                csv_candidates.insert(0, os.path.join(root, f))
            elif f.lower() == "recipes.csv.zip":
                csv_candidates.append(os.path.join(root, f))

    source_csv = None
    for cand in csv_candidates:
        if os.path.exists(cand):
            source_csv = cand
            break

    if source_csv is None:
        print(f"[ERROR] Could not find recipes.csv inside downloaded directory: {dataset_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    shutil.copy2(source_csv, raw_path)
    print(f"[INFO] Copied downloaded CSV to: {raw_path}")


def main():
    config = load_config()
    raw_path = config["paths"]["recipe_data"]

    download_raw_dataset(raw_path)

    print("[INFO] Loading raw recipe data...")
    df = load_recipe_data(raw_path)
    if df.empty:
        print("[ERROR] Loaded DataFrame is empty.")
        sys.exit(1)

    print("[INFO] Cleaning recipe data...")
    df = clean_recipe_data(df)

    print("[INFO] Generating embeddings (this may take a while)...")
    df = generate_recipe_embeddings(df, config)

    cleaned_csv = config["paths"]["cleaned_data_csv"]
    cleaned_pkl = config["paths"]["cleaned_data_pkl"]

    print(f"[INFO] Saving cleaned data to {cleaned_csv} and {cleaned_pkl}...")
    df.to_csv(cleaned_csv, index=False)
    df.to_pickle(cleaned_pkl)

    print("[INFO] Building FAISS indexes...")
    build_recipe_faiss_indexes(df, config)

    print("[SUCCESS] Data preparation complete! You can now start the server with:")
    print("    uvicorn main:app --reload")


if __name__ == "__main__":
    main()
