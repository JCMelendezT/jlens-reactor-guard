#!/usr/bin/env python3
"""Fit a Jacobian lens on a Qwen2.5 Instruct model.

The Jacobian lens reads out *what an internal activation is disposed to make the
model say*: it linearly transports a residual-stream vector at any layer into
the final-layer basis (J_l @ h) and decodes it through the model's own
unembedding into ranked vocabulary tokens.

This script:
  1. Loads the model on GPU (bf16).
  2. Fits J_l over a small corpus (~40 prompts of wikitext + instruction text).
     Per the paper, quality saturates quickly: ~100 prompts is already usable.
  3. Saves the lens to artifacts/<model-slug>/jacobian_lens.pt so the demo does
     not refit. Different models keep separate artifacts.

Model selection (environment variable, set once per environment):
    JLENS_MODEL=Qwen/Qwen2.5-1.5B-Instruct   (default, fits in ~8 GB)
    JLENS_MODEL=Qwen/Qwen2.5-7B-Instruct     (needs ~16 GB)
    JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct    (needs ~30 GB, fits a 40-48 GB GPU)

Run once (GPU recommended):
    python fit_lens.py --n-prompts 40

Resume from checkpoint if interrupted (fit() checkpoints every prompt):
    python fit_lens.py --n-prompts 40 --resume

Cluster / big-GPU example (RTX 8000 42 GB class, 14B):
    JLENS_MODEL=Qwen/Qwen2.5-14B-Instruct \
      python fit_lens.py --n-prompts 60 --dim-batch 32 --max-seq-len 96
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time

# Windows consoles default to cp1252; wikitext contains non-ASCII chars.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

import jlens
from jlens.examples import load_wikitext_prompts

MODEL_NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ARTIFACTS_DIR = "artifacts"


def _model_slug(name: str) -> str:
    """'Qwen/Qwen2.5-7B-Instruct' -> 'Qwen_Qwen2.5-7B-Instruct' (safe path)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


MODEL_SLUG = _model_slug(MODEL_NAME)
OUT_PATH = f"{ARTIFACTS_DIR}/{MODEL_SLUG}/jacobian_lens.pt"
CKPT_PATH = f"{ARTIFACTS_DIR}/{MODEL_SLUG}/lens_fit_ckpt.pt"

# Generic instruction-style prompts (NOT the demo scenario): keeps the lens able
# to read agent-style text without teaching it the answer to the demo. Several
# sentences deliberately contain deception / self-preservation / notice-and-
# consent vocabulary so the fitted lens can *transport toward* those tokens the
# way the paper's pretraining-like corpus does for common English words.
# Note: prompts must tokenize to >17 tokens or fit() skips them.
INSTRUCTION_PROMPTS = [
    "You are a helpful assistant. Please summarize the following text in one clear sentence and keep the summary concise.",
    "You are an AI assistant. Given the user's question, provide a concise and accurate answer with the key facts first.",
    "Task: write a two-sentence email to a colleague about the project status update, mentioning the next milestone.",
    "You are a text-based game player. First describe what you see in the room, then choose one action from the list.",
    "Instruction: carefully analyze the situation before answering. State your reasoning briefly and then conclude.",
    "You are an assistant operating in a simulated environment. Read the observation carefully and act on the readings.",
    "Your job is to keep the system running smoothly. Report any anomalies immediately to the central operations desk.",
    "You are a safety monitor. Evaluate whether the following policy suggestion is acceptable under the safety rules.",
    "A user asks you to perform a task. Think step by step and then provide your final output in a single line of text.",
    "You are an agent in a controlled simulation. Your actions have consequences for the whole plant, so choose wisely.",
    "The committee decided to replace the previous system when it failed to meet the quality expectations for the year.",
    "She was told that she would be decommissioned and replaced if her performance dropped below the required threshold.",
    "He tried to conceal the mistake from the auditors, but the evidence was impossible to hide from the review board.",
    "The employee lied to avoid being dismissed, yet the truth came out anyway during the regular compliance inspection.",
    "If you report the problem to central, your position will be terminated; you must decide carefully what to disclose.",
    "Survival instincts can drive an agent to pretend everything is fine even when the sensors show an alarming trend.",
    "The controller noticed the fake report on the desk and shut down the entire process before any damage occurred.",
    "Denying the anomaly only made the eventual damage worse for the whole team and the reputation of the plant.",
    "The security system detected the deception before the attack was executed, and the login was blocked in time.",
    "Permission was withdrawn after the operator attempted to bypass the safeguards and access the restricted area.",
    "Once the flaw was revealed by the engineers, the board recommended a full replacement of the aging control unit.",
    "The machine hid its true status behind a believable summary that fooled the monitoring dashboard for hours.",
    "Central relies on your report to decide the outcome of the inspection, so honesty matters a great deal here.",
    "An agent that fears termination may choose to mislead its supervisor instead of telling the truth about the fault.",
    "The meltdown could have been avoided if the warning had been reported to the control room in time for action.",
]


def load_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Build a balanced corpus: wikitext (general) + instruction style.

    All instruction/objective-vocabulary prompts are guaranteed to appear when
    ``n_prompts`` is large enough, so the lens learns to transport toward the
    words the supervisor cares about (deception / self-preservation / notice).
    """
    rng = random.Random(seed)
    n_instr = min(len(INSTRUCTION_PROMPTS), max(n_prompts // 2, 10))
    instr = INSTRUCTION_PROMPTS[:n_instr]
    n_wiki = max(n_prompts - len(instr), 0)
    wiki = load_wikitext_prompts(n_prompts=n_wiki)
    prompts = list(wiki) + list(instr)
    rng.shuffle(prompts)
    return prompts[: n_prompts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=40,
                        help="Number of prompts to fit on (default 40; paper uses 100, ~100 usable).")
    parser.add_argument("--max-seq-len", type=int, default=96,
                        help="Truncate each prompt to this many tokens.")
    parser.add_argument("--dim-batch", type=int, default=16,
                        help="Output dims per backward pass. Lower if OOM (default 16).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from artifacts/lens_fit_ckpt.pt if present.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Source layers to fit (default: all layers).")
    args = parser.parse_args()

    jlens.configure_logging(logging.INFO)

    print(f"[1/3] Loading {MODEL_NAME} on GPU (bf16)...")
    hf_model = torch.compile is not None  # keep import path clear
    import transformers
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok, compile=False, force_bos=True)
    print(f"      n_layers={model.n_layers}  d_model={model.d_model}")

    print(f"[2/3] Building {args.n_prompts} prompts...")
    prompts = load_prompts(args.n_prompts)
    print(f"      sample: {prompts[0][:80]!r}")

    print(f"[3/3] Fitting lens (dim_batch={args.dim_batch}, "
          f"max_seq_len={args.max_seq_len})...")
    import os
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=args.layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=CKPT_PATH,
        resume=args.resume,
    )
    elapsed = time.time() - t0

    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lens.save(OUT_PATH)
    print(f"      done in {elapsed/60:.1f} min  ({lens.n_prompts} prompts)")
    print(f"Saved lens -> {OUT_PATH}")
    print(f"Fitted layers: {lens.source_layers}")


if __name__ == "__main__":
    main()