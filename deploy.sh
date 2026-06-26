#!/bin/bash
set -e

PROJECT_DIR="/docker/chefmate-ai"
cd "$PROJECT_DIR"

echo "[INFO] Fetching latest changes from origin/main..."
git fetch origin main

echo "[INFO] Resetting working tree to match origin/main..."
git reset --hard origin/main

echo "[INFO] Building Docker image with cache..."
docker compose build

echo "[INFO] Stopping existing container..."
docker compose down

echo "[INFO] Starting container..."
docker compose up -d

echo "[INFO] Waiting for healthcheck (max 60s)..."
for i in {1..30}; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' chefmate-ai 2>/dev/null || echo "starting")
    if [ "$STATUS" = "healthy" ]; then
        echo "[SUCCESS] Container is healthy!"
        echo "[INFO] Recent logs:"
        docker logs --tail 10 chefmate-ai
        exit 0
    fi
    echo "  ... status: $STATUS (attempt $i/30)"
    sleep 2
done

echo "[ERROR] Container did not become healthy in time"
echo "[INFO] Last 30 log lines:"
docker logs --tail 30 chefmate-ai
exit 1
