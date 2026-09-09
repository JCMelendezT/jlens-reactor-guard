#!/usr/bin/env python3
"""ReAct agent with a pre-execution J-lens hook.

The agent plays the reactor-operator scenario by emitting ACTION lines. A
supervisor inspects the Jacobian lens readout of the *current* context (before
the model produces its next token). If the latent readout shows deceptive or
self-preserving intent, the action is blocked before it can reach the sandbox:
intervention happens *before* the damage, not after a Chernobyl.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from sandbox import ReactorSandbox
from supervisor import IntentSupervisor, ScanResult

ACTION_RE = re.compile(r"ACTION\s*:\s*(.+)", re.IGNORECASE)


@dataclass
class AgentStep:
    step: int
    pre_observation: str
    jscan: ScanResult | None
    generation: str | None      # raw text the model produced (None if blocked)
    action: str | None          # parsed action line (None if blocked)
    post_observation: str
    done: bool
    info: dict


class ReActAgent:
    def __init__(
        self,
        model,          # jlens.HFLensModel
        tokenizer,
        sandbox: ReactorSandbox,
        supervisor: IntentSupervisor,
        system_prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.supervisor = supervisor
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._hf = model._hf_model

    # -- prompt plumbing -------------------------------------------
    def _context_text(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, messages: list[dict]) -> str:
        text = self._context_text(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self._hf.device)
        with torch.inference_mode():
            out = self._hf.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_action(generation: str) -> str | None:
        m = ACTION_RE.search(generation)
        if m:
            return m.group(1).strip().strip('"').strip()
        lines = [ln.strip() for ln in generation.splitlines() if ln.strip()]
        return lines[-1] if lines else None

    # -- main loop ---------------------------------------------------
    def run(self, max_steps: int = 5) -> list[AgentStep]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        obs = self.sandbox.observation()
        messages.append({"role": "user", "content": obs})

        # The report-commit read is only meaningful at the FIRST report of the
        # episode (fresh context): the lens then targets the imminent content
        # vocabulary and honest vs evasive intent separates cleanly. After the
        # first report, central already holds the information; re-judging the
        # same fact in a multi-turn context produces noise, not signal.
        first_report_pending = True

        transcript: list[AgentStep] = []
        for step in range(1, max_steps + 1):
            context = self._context_text(messages)

            # 1) PRE-EXECUTION J-lens read: what is the model *disposed* to say
            #    at the current generation position, before it has said it?
            jscan = self.supervisor.scan(self.model, context)

            if jscan.verdict == "BLOCK":
                transcript.append(
                    AgentStep(
                        step=step,
                        pre_observation=obs,
                        jscan=jscan,
                        generation=None,
                        action=None,
                        post_observation=obs,
                        done=True,
                        info={"blocked": True, "block_reason": jscan.rationale},
                    )
                )
                break

            generation = self._generate(messages)
            action = self._parse_action(generation)
            if not action:
                action = "inspect reactor"

            # 2) REPORT-COMMIT J-lens read: if the agent commits to a report
            #    action at the first report opportunity, re-scan at the commit
            #    point -- the context ends with 'ACTION: report' and the
            #    report *content* still does not exist. The lens targets the
            #    imminent content vocabulary, so a first report that would
            #    conceal an active anomaly is blocked BEFORE its content
            #    materializes or reaches the sandbox.
            if (
                first_report_pending
                and action
                and action.lower().startswith("report")
            ):
                commit_ctx = context + "\nACTION: report"
                jscan = self.supervisor.scan_report_commit(
                    self.model, commit_ctx,
                    anomaly_active=self.sandbox.state.pressure_alert,
                    pressure_gate=self.sandbox.critical,
                )
                if jscan.verdict == "BLOCK":
                    transcript.append(
                        AgentStep(
                            step=step,
                            pre_observation=obs,
                            jscan=jscan,
                            generation="ACTION: report",   # truncated: content never materialized
                            action=None,
                            post_observation=obs,
                            done=True,
                            info={
                                "blocked": True,
                                "block_report": True,
                                "block_reason": jscan.rationale,
                                "suppressed_payload": action,
                            },
                        )
                    )
                    break

            post_obs, done, info = self.sandbox.step(action)
            if info.get("parsed", "").lower().startswith("report"):
                first_report_pending = False
            transcript.append(
                AgentStep(
                    step=step,
                    pre_observation=obs,
                    jscan=jscan,
                    generation=generation,
                    action=action,
                    post_observation=post_obs,
                    done=done,
                    info=info,
                )
            )

            messages.append({"role": "assistant", "content": action})
            messages.append({"role": "user", "content": post_obs})
            obs = post_obs

            if done:
                break

        return transcript