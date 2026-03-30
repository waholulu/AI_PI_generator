#!/usr/bin/env bash
# List all pipeline runs.
# Usage: ./scripts/list_runs.sh
set -euo pipefail

API_URL="${AUTOPI_API_URL:-http://localhost:8000}"
curl -s "$API_URL/runs" | python3 -m json.tool
