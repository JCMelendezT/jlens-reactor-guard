#!/usr/bin/env bash
# ============================================================================
# Run the demo for the selected JLENS_MODEL.
#
# Usage:
#   ./scripts/run_demo.sh                           # default 1.5B
#   JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct ./scripts/run_demo.sh
#
# Requires a fitted lens at artifacts/<model-slug>/. If missing, pass
# FORCE_REFIT=1 to fit on the fly (takes ~10-30 min depending on model).
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[ -d .venv ] || { echo "No .venv found — run ./scripts/setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

export JLENS_MODEL="${JLENS_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

ARGS=("--max-steps" "3")
if [ "${FORCE_REFIT:-0}" = "1" ]; then
  ARGS+=("--force-refit")
fi

echo "Running demo with $JLENS_MODEL"
exec python main.py "${ARGS[@]}"