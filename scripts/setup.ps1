# ============================================================================
# Reproducible setup — Windows (Python 3.11 recommended; CUDA 12.8 wheels).
# Creates a venv, installs the pinned dependencies, and installs jlens
# from source at the exact commit used by this project.
#
# Usage (PowerShell):
#   .\scripts\setup.ps1                              # default Qwen2.5-1.5B
#   $env:JLENS_MODEL="Qwen/Qwen2.5-14B-Instruct"; .\scripts\setup.ps1
# ============================================================================
$ErrorActionPreference = "Stop"

$PyExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }
if (-not $env:JLENS_MODEL) { $env:JLENS_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" }
$JlensCommit = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "[1/4] Creating venv with $PyExe ..." -ForegroundColor Cyan
& $PyExe -m venv .venv
if (-not $?) { throw "venv creation failed" }
& ".venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "[2/4] Installing pinned dependencies (CUDA 12.8 wheels) ..." -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if (-not $?) { throw "pip install failed" }

Write-Host "[3/4] Installing jlens from source @ $JlensCommit ..." -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -m pip install "git+https://github.com/anthropics/jacobian-lens@${JlensCommit}#egg=jlens"
if (-not $?) { throw "jlens install failed" }

Write-Host "[4/4] Verifying install ..." -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -c "import torch, transformers, jlens; print('torch', torch.__version__); print('transformers', transformers.__version__); print('jlens OK')"

Write-Host "`nSetup complete. Model selected: $env:JLENS_MODEL" -ForegroundColor Green
Write-Host "Next: python fit_lens.py --n-prompts 40   # fit the lens (once)"