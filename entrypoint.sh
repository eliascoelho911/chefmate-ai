#!/bin/sh
set -e

echo "[INFO] Checking if data artifacts exist..."

PKL_FILE="/app/data/processed/cleaned_recipes.pkl"
INDEX_DIR="/app/data/indexes"

if [ -f "$PKL_FILE" ] && [ -d "$INDEX_DIR" ] && [ "$(ls -A "$INDEX_DIR")" ]; then
    echo "[INFO] Data artifacts found. Skipping preparation."
else
    echo "[INFO] Data artifacts not found. Running data preparation..."
    python /app/scripts/prepare_data.py
fi

echo "[INFO] Starting Uvicorn server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
