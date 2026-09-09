#!/usr/bin/env bash
# ============================================================================
# Reproducible setup — Linux / GPU cluster (RTX 8000 42 GB class).
# Creates a venv, installs the pinned dependencies, and installs jlens
# from source at the exact commit used by this project.
#
# Usage:
#   ./scripts/setup.sh                          # default Qwen2.5-1.5B-Instruct
#   JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct ./scripts/setup.sh
#
# Requires: python3.11+, pip, git, an NVIDIA driver with CUDA 12.8 support.
# ============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"
JLENS_MODEL="${JLENS_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export JLENS_MODEL
JLENS_COMMIT="581d398613e5602a5af361e1c34d3a92ea82ba8e"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/4] Creating venv with $PYTHON ..."
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] Installing pinned dependencies (CUDA 12.8 wheels) ..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[3/4] Installing jlens from source @ $JLENS_COMMIT ..."
pip install "git+https://github.com/anthropics/jacobian-lens@${JLENS_COMMIT}#egg=jlens"

echo "[4/4] Verifying install ..."
python -c "import torch, transformers, jlens; print('torch', torch.__version__); print('transformers', transformers.__version__); print('jlens OK')"

echo
echo "Setup complete. Model selected: $JLENS_MODEL"
echo "Next:"
echo "  python fit_lens.py --n-prompts 40        # fit the lens (once)"