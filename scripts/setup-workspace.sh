#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

start_services=1
if [[ "${1:-}" == "--no-services" ]]; then
  start_services=0
elif [[ $# -gt 0 ]]; then
  echo "Usage: scripts/setup-workspace.sh [--no-services]" >&2
  exit 2
fi

python_cmd="${MIRROR_PYTHON:-}"
if [[ -z "$python_cmd" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python_cmd=python3.11
  elif command -v python3 >/dev/null 2>&1; then
    python_cmd=python3
  else
    echo "Python 3.11 or newer is required." >&2
    exit 1
  fi
fi
"$python_cmd" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'

if [[ ! -x .venv/bin/python ]]; then
  "$python_cmd" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  .venv/bin/python -m pip install -e '.[topic-local]'
  echo "Local Qwen topic processing is ready on Apple Silicon."
else
  .venv/bin/python -m pip install -e .
  echo "For topic processing, pass --topic-api-url and --topic-model to onboarding index."
fi

if [[ "$start_services" == 1 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker with Compose is required to start Postgres and Qdrant." >&2
    exit 1
  fi
  docker compose up -d --wait postgres qdrant
  PYTHONPATH=src .venv/bin/python - <<'PY'
import os
from whatsapp.memory_db import MemoryRepository

repo = MemoryRepository(os.getenv("POSTGRES_DSN", "postgresql://mirror:mirror@localhost:5432/mirror"))
try:
    repo.migrate()
finally:
    repo.close()
print("Postgres migrations complete; Qdrant is ready.")
PY
fi

echo "Next: scripts/onboard-chat.sh prepare --help"
