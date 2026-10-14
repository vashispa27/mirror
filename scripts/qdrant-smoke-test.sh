#!/usr/bin/env bash
set -euo pipefail

endpoint="${QDRANT_ENDPOINT:-http://localhost:6333}"
collection="qdrant-smoke-$(date +%s)-$$"

cleanup() {
  curl --silent --show-error --fail --request DELETE \
    "${endpoint}/collections/${collection}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

curl --silent --show-error --fail --retry 10 --retry-connrefused \
  "${endpoint}/healthz" >/dev/null

curl --silent --show-error --fail --request PUT \
  --header "Content-Type: application/json" \
  --data '{"vectors":{"size":4,"distance":"Cosine"}}' \
  "${endpoint}/collections/${collection}" >/dev/null

curl --silent --show-error --fail --request PUT \
  --header "Content-Type: application/json" \
  --data '{"points":[{"id":1,"vector":[0.1,0.2,0.3,0.4]}]}' \
  "${endpoint}/collections/${collection}/points?wait=true" >/dev/null

result="$(curl --silent --show-error --fail --request POST \
  --header "Content-Type: application/json" \
  --data '{"query":[0.1,0.2,0.3,0.4],"limit":1,"with_payload":false}' \
  "${endpoint}/collections/${collection}/points/query")"

QDRANT_RESULT="${result}" python3 - <<'PY'
import json
import os

response = json.loads(os.environ["QDRANT_RESULT"])
points = response.get("result", {}).get("points", [])
if len(points) != 1 or points[0].get("id") != 1:
    raise SystemExit(f"unexpected Qdrant query result: {response}")
PY

cleanup
trap - EXIT
echo "PASS: Qdrant created, queried, and removed ${collection}."
