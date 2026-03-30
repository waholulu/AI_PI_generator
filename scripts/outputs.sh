#!/usr/bin/env bash
# List output files for a run.
# Usage: ./scripts/outputs.sh <run_id>
set -euo pipefail

API_URL="${AUTOPI_API_URL:-http://localhost:8000}"
RUN_ID="${1:?Usage: outputs.sh '<run_id>'}"

curl -s "$API_URL/runs/$RUN_ID/outputs" | python3 -m json.tool
