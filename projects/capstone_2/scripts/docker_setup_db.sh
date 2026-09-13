#!/usr/bin/env bash
# Start the OpsMate MySQL container and apply the schema. Safe to run repeatedly.
# Usage: ./scripts/docker_setup_db.sh [--reset]
set -euo pipefail

cd "$(dirname "$0")/.."

CONTAINER="opsmate-mysql"
ROOT_PW="rootpw"
DATABASE="opsmate"

if [[ "${1:-}" == "--reset" ]]; then
  echo "Removing existing container and data volume..."
  docker compose down -v
fi

echo "Starting MySQL..."
docker compose up -d

echo "Waiting for the container to report healthy..."
for _ in $(seq 1 60); do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo starting)"
  echo "  health: $status"
  [[ "$status" == "healthy" ]] && break
  sleep 3
done
[[ "$status" == "healthy" ]] || { echo "MySQL never became healthy. docker logs $CONTAINER"; exit 1; }

echo "Applying db/schema.sql..."
docker exec -i "$CONTAINER" mysql -u root -p"$ROOT_PW" < db/schema.sql

count="$(docker exec "$CONTAINER" mysql -u root -p"$ROOT_PW" -N -B \
  -e "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='$DATABASE';")"
echo
echo "Ready. Database '$DATABASE' has ${count} tables."
echo "Next: python scripts/populate_database.py"
