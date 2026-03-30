#!/usr/bin/env bash
# Get logs for a run.
# Usage: ./scripts/logs.sh <run_id> [level]
# Example: ./scripts/logs.sh abc123 ERROR
set -euo pipefail

API_URL="${AUTOPI_API_URL:-http://localhost:8000}"
RUN_ID="${1:?Usage: logs.sh '<run_id>' [level]}"
LEVEL="${2:-}"

URL="$API_URL/runs/$RUN_ID/logs"
if [[ -n "$LEVEL" ]]; then
  URL="$URL?level=$LEVEL"
fi

curl -s "$URL" | python3 -m json.tool
