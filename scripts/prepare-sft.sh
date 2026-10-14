#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run scripts/setup-workspace.sh first." >&2
  exit 1
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m whatsapp.sft_cli "$@"
