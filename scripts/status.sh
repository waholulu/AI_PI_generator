#!/usr/bin/env bash
# Check the status of a run.
# Usage: ./scripts/status.sh <run_id>
set -euo pipefail

API_URL="${AUTOPI_API_URL:-http://localhost:8000}"
RUN_ID="${1:?Usage: status.sh '<run_id>'}"

curl -s "$API_URL/runs/$RUN_ID/status" | python3 -m json.tool
