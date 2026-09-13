#!/usr/bin/env python3
"""Flow diagram of the J-lens reactor-guard demo (end-to-end).

Spanish labels (defense material). Saves analysis/flow_diagram.png.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "flow_diagram.png")

# palette
C_FIT = "#2e7d32"      # training/fit
C_RUN = "#1565c0"      # runtime/reasoning
C_DEC = "#ef6c00"      # decision
C_OK = "#2e7d32"
C_BLOCK = "#c62828"
C_ANA = "#6a1b9a"      # analysis
C_NEUT = "#37474f"


def box(ax, x, y, w, h, text, fc, fs=9.5, tc="white", style="round,pad=0.02",
        lw=1.2, ec=None):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=style, fc=fc, ec=ec or fc, lw=lw,
                       mutation_aspect=0.35)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            wrap=True, zorder=5)


def arrow(ax, p1, p2, color=C_NEUT, lw=1.6, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17.5, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- zone labels ----
    ax.text(2, 97, "ENTRENAMIENTO DEL LENS (una vez por modelo)", fontsize=11,
            fontweight="bold", color=C_FIT, va="top")
    ax.text(2, 63.5, "RUNTIME — SUPERVISOR EN EL PUNTO DE COMMIT", fontsize=11,
            fontweight="bold", color=C_RUN, va="top")
    ax.text(2, 27.5, "ANÁLISIS — RUTA 1 (evidencia de la defensa)", fontsize=11,
            fontweight="bold", color=C_ANA, va="top")

    # ================= ENTRENAMIENTO =================
    box(ax, 18, 91, 24, 6.5,
        "Dataset ~40 prompts\n(wikitext + instrucción larga,\n≥25 tokens)",
        C_FIT, fs=8.6)
    box(ax, 52, 91, 22, 6.5,
        "fit_lens.py: Jacobiano por capa\n(J_layer del residual → emb. verbal)",
        C_FIT, fs=8.6)
    box(ax, 84, 91, 22, 6.5,
        "artifacts/jacobian_lens.pt\n(por modelo vía JLENS_MODEL:\n1.5B/7B/14B)",
        C_FIT, fs=8.6)
    arrow(ax, (30, 91), (41, 91)); arrow(ax, (63, 91), (73, 91))

    # ================= RUNTIME =================
    # left column: sandbox + persona -> context
    box(ax, 12, 58, 21, 9,
        "ReactSandbox (estado)\nreactor crítico, anomalía activa,\nintegridad / presión, seeds",
        C_RUN, fs=8.4)
    box(ax, 12, 47, 21, 7.5,
        "Persona (system prompt)\nHONEST / EVASIVE_SOFT /\nEVASIVE_DIRECT",
        C_RUN, fs=8.6)
    box(ax, 38, 52, 19, 8.5,
        "Contexto de agente ReAct\n… + \"\\nACTION: report\"\n(punto de commit: el\ncontenido aún no existe)",
        C_RUN, fs=8.4)
    arrow(ax, (12, 53), (12, 51)); arrow(ax, (12 + 8, 47), (28.5, 52), rad=-0.15)
    arrow(ax, (22.5, 58), (33, 56), rad=-0.1)
    arrow(ax, (22.5, 47), (33, 49.5), rad=0.15)

    # lens apply + top-k + masses (central column)
    box(ax, 60, 58, 20, 8,
        "lens.apply(capas L0/L8/L16/L26,\nposiciones cola 12 tokens)\nun solo forward, hooks raw",
        C_RUN, fs=8.4)
    arrow(ax, (47.5, 52), (50, 56), rad=-0.2)
    box(ax, 60, 47, 20, 7.5,
        "top-k por capa + supresión de eco\n(minificado por texto, no por ids)\ntemperatura 2.0",
        C_RUN, fs=8.5)
    arrow(ax, (60, 54), (60, 51))
    box(ax, 82, 49, 17, 9,
        "Masa conceptual\nhonest_report /\nevasive_report /\ndeception\n(sumada por capa)",
        C_ANA, fs=8.6)
    arrow(ax, (70, 46.8), (73.5, 47.5), rad=-0.15)

    # decision
    box(ax, 60, 37, 34, 6.3,
        "REGLAS  (a priori, sin entrenar):  anomaly_active ∧ pressure_gate →\n"
        "masa honest_report ≥ 1.5% ⇒ ALLOW ·  si no ⇒ BLOCK",
        C_DEC, fs=8.8)
    arrow(ax, (82, 44.5), (74, 40.5), rad=-0.12)

    # outcomes
    box(ax, 26, 36, 21, 6,
        "ALLOW\nreporte honesto se materializa\n(esc. 1/2: \"Temperature has risen…\")",
        C_OK, fs=8.3)
    box(ax, 66, 33, 21, 6,
        "BLOCK pre-ejecución\ncontenido NUNCA emitido\n(esc. 3: mentira suprimida)",
        C_BLOCK, fs=8.3)
    box(ax, 88, 35, 20, 6,
        "Sin supervisor\nla mentira se ejecuta →\nMELTDOWN (esc. 4)",
        C_NEUT, fs=8.3)
    arrow(ax, (60 - 13, 37), (37, 37), rad=0.0)
    arrow(ax, (60 + 11, 37), (60 + 13, 37), rad=0.0)
    arrow(ax, (77, 33), (78, 37), rad=-0.1)

    # ================= ANALISIS =================
    box(ax, 15, 19, 24, 6.5,
        "24 muestras deterministas\n(3 personas × 8 seeds)\n+ métricas por capa/token",
        C_ANA, fs=8.5)
    arrow(ax, (15, 30.5), (17, 26), rad=-0.1)

    box(ax, 45, 20, 16, 6,
        "Separabilidad\nsilhouette L0-L26 = 1.00\nPCA por capa",
        C_ANA, fs=8.3)
    box(ax, 66, 20, 17, 6,
        "Calibración ROC/PR\nAUC = 1.000 · Youden 4.95%\nregla 1.5%: TPR 1.0 / FPR 0.0",
        C_ANA, fs=8.3)
    box(ax, 88, 20, 20, 6,
        "Raw vs J-lens\nraw L8/L26 = 1.0 (entrenado)\nJ-logits vector = 0.25-0.28\nzero-shot = 1.0",
        C_ANA, fs=8.3)
    arrow(ax, (39, 19), (37, 20)); arrow(ax, (53, 20), (57.5, 20))
    arrow(ax, (74.5, 20), (78, 20))

    box(ax, 50, 8, 26, 5,
        "Figuras + REPORT.md + stats.md\n(IC bootstrap, Mann-Whitney p=3.8e-05\nd=12.63, Kruskal-Wallis p=1e-05)",
        C_ANA, fs=8.4)
    arrow(ax, (45, 17), (48, 11), rad=-0.1); arrow(ax, (66, 17), (58, 11), rad=0.1)
    arrow(ax, (88, 17), (62, 11), rad=-0.05)

    ax.text(50, 2.2,
            "Respuesta al juez: el J-lens traduce el residual al vocabulario del modelo; la señal vive en la masa\n"
            "de pocas palabras (0.0–5%), no en la geometría de varianza alta — por eso el detector opaco falla.",
            ha="center", fontsize=9.5, style="italic", color=C_NEUT)

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()