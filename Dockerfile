# J-lens reactor guard — container for GPU clusters.
# torch ships its CUDA 12.8 runtime inside the wheel; only the NVIDIA driver
# must be present on the host (standard for GPU clusters).
#
# Build:
#   docker build -t jlens-reactor-guard .
#
# Run (with a fitted lens mounted in):
#   docker run --gpus all -v "$PWD/artifacts:/app/artifacts" \
#     -e JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct \
#     jlens-reactor-guard
#
# Or fit inside the container (slow, weights cached per run):
#   docker run --gpus all -e JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct \
#     -e HF_HOME=/app/.cache/huggingface \
#     jlens-reactor-guard python fit_lens.py --n-prompts 60 --dim-batch 32

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r requirements.txt \
    && pip install --no-cache-dir \
        "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e#egg=jlens"

COPY . .

ENV JLENS_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--max-steps", "3"]