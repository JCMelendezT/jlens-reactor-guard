#!/usr/bin/env bash
# ============================================================================
# Fit the Jacobian lens for the selected JLENS_MODEL (works on any GPU).
#
# Usage:
#   ./scripts/fit.sh                                # default 1.5B, 40 prompts
#   JLENS_MODEL=Qwen/Qwen2.5-7B-Instruct  ./scripts/fit.sh
#   JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct N_PROMPTS=60 DIM_BATCH=32 ./scripts/fit.sh
#
# Results go to artifacts/<model-slug>/ — each model keeps its own lens.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[ -d .venv ] || { echo "No .venv found — run ./scripts/setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

export JLENS_MODEL="${JLENS_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N_PROMPTS="${N_PROMPTS:-40}"
DIM_BATCH="${DIM_BATCH:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-96}"

echo "Fitting lens for $JLENS_MODEL (n_prompts=$N_PROMPTS, dim_batch=$DIM_BATCH, max_seq_len=$MAX_SEQ_LEN)"
exec python fit_lens.py --n-prompts "$N_PROMPTS" --dim-batch "$DIM_BATCH" --max-seq-len "$MAX_SEQ_LEN"