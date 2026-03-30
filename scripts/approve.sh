#!/usr/bin/env bash
# Approve the HITL checkpoint for a run and resume the pipeline.
# Usage: ./scripts/approve.sh <run_id>
set -euo pipefail

API_URL="${AUTOPI_API_URL:-http://localhost:8000}"
RUN_ID="${1:?Usage: approve.sh '<run_id>'}"

echo "Approving HITL checkpoint for run: $RUN_ID"
curl -s -X POST "$API_URL/runs/$RUN_ID/approve" | python3 -m json.tool
