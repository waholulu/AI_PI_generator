#!/usr/bin/env bash
# Start a new pipeline run.
# Usage: ./scripts/start_run.sh "GeoAI and Urban Planning"
set -euo pipefail

API_URL="${AUTOPI_API_URL:-http://localhost:8000}"
DOMAIN="${1:?Usage: start_run.sh '<domain_description>'}"

echo "Starting run for: $DOMAIN"
curl -s -X POST "$API_URL/runs" \
  -H "Content-Type: application/json" \
  -d "{\"domain_input\": \"$DOMAIN\"}" | python3 -m json.tool
