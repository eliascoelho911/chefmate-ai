#!/bin/sh
set -e

echo "[INFO] Checking if data artifacts exist..."

DB_FILE="/app/data/processed/recipes.db"
INDEX_DIR="/app/data/indexes"

if [ -f "$DB_FILE" ] && [ -d "$INDEX_DIR" ] && [ "$(ls -A "$INDEX_DIR")" ]; then
    echo "[INFO] Data artifacts found."
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

# If running as root, ensure cache directory exists and is writable by appuser,
# then drop privileges to appuser for the actual server process.
if [ "$(id -u)" = "0" ]; then
    echo "[INFO] Running as root. Setting up permissions for appuser (UID 1001)..."
    mkdir -p /app/data/cache
    chown -R appuser:appgroup /app/data/cache
    # Also ensure the data volume root is writable so appuser can create
    # subdirectories if needed in the future.
    chown appuser:appgroup /app/data
    echo "[INFO] Starting Uvicorn server as appuser..."
    exec gosu appuser uvicorn main:app --host 0.0.0.0 --port 8000
else
    echo "[INFO] Starting Uvicorn server..."
    exec uvicorn main:app --host 0.0.0.0 --port 8000
fi
