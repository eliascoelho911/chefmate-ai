#!/bin/sh
set -e

echo "[INFO] Checking if data artifacts exist..."

DB_FILE="/app/data/processed/recipes.db"
INDEX_DIR="/app/data/indexes"

if [ -f "$DB_FILE" ] && [ -d "$INDEX_DIR" ] && [ "$(ls -A "$INDEX_DIR")" ]; then
    echo "[INFO] Data artifacts found. Starting server..."
else
    echo "[ERROR] Data artifacts not found."
    echo "[ERROR] Expected:"
    echo "  - $DB_FILE"
    echo "  - $INDEX_DIR/ (non-empty)"
    echo "[ERROR] Please run data preparation manually before starting the container:"
    echo "  docker compose exec chefmate python scripts/prepare_data.py"
    echo "[ERROR] Or transfer existing artifacts into the container/volume."
    exit 1
fi

echo "[INFO] Starting Uvicorn server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
