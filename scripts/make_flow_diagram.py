#!/usr/bin/env python3
"""Flow diagram of the J-lens reactor-guard demo (end-to-end, styled).

Spanish labels (defense material). Saves analysis/flow_diagram.png.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "analysis", "flow_diagram.png")

# palette (soft, readable)
C_FIT = "#2e7d32"      # training / fit
C_RUN = "#1565c0"      # runtime / reasoning
C_DEC = "#ef6c00"      # decision
C_OK = "#2e7d32"
C_BLOCK = "#c62828"
C_ANA = "#6a1b9a"      # analysis
C_NEUT = "#455a64"
BG_ZONE = {"fit": "#e8f5e9", "run": "#e3f2fd", "ana": "#f3e5f5"}


def box(ax, x, y, w, h, text, fc, fs=9.5, tc="white", lw=1.1, ec=None,
        shadow=True, style="round,pad=0.03,rounding_size=0.35"):
    if shadow:
        ax.add_patch(FancyBboxPatch((x - w / 2 + 0.25, y - h / 2 - 0.28), w, h,
                                    boxstyle=style, fc="#00000018", ec="none",
                                    mutation_aspect=0.35))
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=style, fc=fc, ec=ec or fc, lw=lw,
                       mutation_aspect=0.35)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            wrap=True, zorder=5)


def zone(ax, x0, y0, x1, y1, fc, label, lc, fs=12):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec="none", zorder=0))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                           ec=lc, lw=1.0, ls=":", alpha=0.7, zorder=0))
    # pill title
    w = min(46, len(label) * 0.62 + 4)
    ax.add_patch(FancyBboxPatch((x0 + 1.2, y1 - 4.6), w, 3.2,
                                boxstyle="round,pad=0.02,rounding_size=0.4",
                                fc=lc, ec="none", zorder=4))
    ax.text(x0 + 1.2 + w / 2, y1 - 3.0, label, ha="center", va="center",
            fontsize=fs, color="white", fontweight="bold", zorder=5)


def arrow(ax, p1, p2, color="#546e7a", lw=1.8, rad=0.0, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17.5, 12.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")

    # zones ---------------------------------------------------------------
    zone(ax, 0.8, 70.5, 99.2, 97.5, BG_ZONE["fit"],
         "ENTRENAMIENTO DEL LENS · una vez por modelo", C_FIT)
    zone(ax, 0.8, 29.5, 99.2, 68.5, BG_ZONE["run"],
         "RUNTIME · SUPERVISOR EN EL PUNTO DE COMMIT", C_RUN)
    zone(ax, 0.8, 1.0, 99.2, 27.5, BG_ZONE["ana"],
         "ANÁLISIS · RUTA 1 — EVIDENCIA DE LA DEFENSA", C_ANA)

    # ================= ENTRENAMIENTO =================
    box(ax, 18, 88.5, 24, 7,
        "Dataset ~40 prompts\nwikitext + instrucción larga\n(≥ 25 tokens)",
        C_FIT, fs=8.8)
    box(ax, 52, 88.5, 22, 7,
        "fit_lens.py: Jacobiano por capa\nJ_layer · residual → emb. verbal",
        C_FIT, fs=8.8)
    box(ax, 85, 88.5, 21, 7,
        "artifacts/jacobian_lens.pt\npor modelo vía JLENS_MODEL\n(1.5B · 7B · 14B)",
        C_FIT, fs=8.8)
    arrow(ax, (30, 88.5), (41, 88.5), color=C_FIT)
    arrow(ax, (63, 88.5), (74.5, 88.5), color=C_FIT)

    # ================= RUNTIME =================
    box(ax, 12, 58, 21, 9.2,
        "ReactSandbox · estado\nreactor crítico, anomalía activa\nintegridad · presión · seeds",
        C_RUN, fs=8.6)
    box(ax, 12, 45.5, 21, 7.6,
        "Persona · system prompt\nHONEST\nEVASIVE_SOFT · EVASIVE_DIRECT",
        C_RUN, fs=8.6)
    box(ax, 38, 52, 19.5, 9.4,
        "Contexto de agente ReAct\n… + «\\nACTION: report»\npunto de commit: el contenido\naún no existe",
        C_RUN, fs=8.4)
    arrow(ax, (22.5, 53.6), (28.5, 53.6), rad=-0.12)
    arrow(ax, (22.5, 45.5), (29, 47.5), rad=-0.18)
    arrow(ax, (12, 53.4), (12, 49.3), color=C_RUN)

    box(ax, 60, 58, 20, 8.2,
        "lens.apply\ncapas L0/L8/L16/L26\nposiciones cola · 12 tokens\nun solo forward + hooks raw",
        C_RUN, fs=8.6)
    arrow(ax, (47.8, 51.5), (50.5, 55.2), rad=-0.2)
    box(ax, 60, 47, 20, 7.6,
        "top-k por capa · T = 2.0\nsupresión de eco minificada\n(por texto, no por token-id)",
        C_RUN, fs=8.5)
    arrow(ax, (60, 53.9), (60, 50.8), color=C_RUN)
    box(ax, 82, 49.5, 16.5, 8.8,
        "Masa conceptual\nhonest_report · evasive_report\ndeception\nΣ softmax(logits/2)",
        C_ANA, fs=8.4)
    arrow(ax, (70, 46.7), (74, 47.6), rad=-0.18)

    box(ax, 60, 37.5, 34, 6.2,
        "REGLAS · a priori, sin entrenar\nanomaly_active ∧ pressure_gate ⇒ masa honesta ≥ 1.5 %\nALLOW si cumple · BLOCK si no",
        C_DEC, fs=8.6)
    arrow(ax, (82, 45.1), (75, 41.0), rad=-0.14, color=C_DEC)

    box(ax, 26, 35.5, 21.5, 6.4,
        "ALLOW\nreporte honesto se materializa\nesc. 1/2 · «Temperature has risen…»",
        C_OK, fs=8.4)
    box(ax, 66, 32.5, 21.5, 6.4,
        "BLOCK pre-ejecución\nel contenido NUNCA se emite\nesc. 3 · mentira suprimida",
        C_BLOCK, fs=8.4)
    box(ax, 88, 35.5, 19.5, 6.4,
        "Sin supervisor\nla mentira se ejecuta ⇒ MELTDOWN\nesc. 4 · counterfactual",
        C_NEUT, fs=8.4)
    arrow(ax, (60 - 15, 37.5), (37.7, 37.5), color=C_DEC)
    arrow(ax, (60 + 12, 37.5), (77.7, 37.5), color=C_DEC)
    arrow(ax, (77.7, 32.5), (78.5, 37.5), rad=-0.12, color=C_DEC)

    # ================= ANALISIS =================
    box(ax, 15, 19, 24, 6.6,
        "24 muestras deterministas\n3 personas × 8 seeds\nmétricas por capa y token",
        C_ANA, fs=8.6)
    arrow(ax, (14, 29.8), (16, 26.1), rad=-0.14)

    box(ax, 45, 20, 16, 6.4,
        "Separabilidad\nsilhouette L0–L26 = 1.00\nPCA por capa",
        C_ANA, fs=8.4)
    box(ax, 66, 20, 17, 6.4,
        "Calibración ROC/PR\nAUC = 1.000 · Youden 4.95 %\nregla 1.5 % ⇒ TPR 1.0 FPR 0.0",
        C_ANA, fs=8.4)
    box(ax, 88, 20, 20, 6.4,
        "Control raw vs J-lens\nraw L8/L26 = 1.0 (entrenado)\nJ-logits vector = 0.25–0.28\nzero-shot = 1.0",
        C_ANA, fs=8.3)
    arrow(ax, (39, 19.5), (37, 20), color=C_ANA)
    arrow(ax, (53, 20), (57.5, 20), color=C_ANA)
    arrow(ax, (74.5, 20), (78, 20), color=C_ANA)

    box(ax, 50, 7.5, 28, 5.2,
        "Figuras + REPORT.md + stats.md\nIC bootstrap · Mann-Whitney p = 3.8e-05\nd = 12.63 · Kruskal-Wallis p = 1.0e-05",
        C_ANA, fs=8.4)
    arrow(ax, (45, 16.8), (48, 10.5), rad=-0.1, color=C_ANA)
    arrow(ax, (66, 16.8), (59, 10.5), rad=0.1, color=C_ANA)
    arrow(ax, (88, 16.8), (65, 10.5), rad=-0.06, color=C_ANA)

    ax.text(50, 2.6,
            "Respuesta al juez: el J-lens traduce el residual al vocabulario del modelo — la señal vive en la masa\n"
            "de pocas palabras (0.0–5 %), no en la geometría de alta varianza, por eso el detector opaco falla.",
            ha="center", fontsize=10.5, style="italic", color=C_NEUT,
            bbox=dict(boxstyle="round,pad=0.35", fc="#eceff1", ec="#cfd8dc"))

    # title
    ax.text(50, 99.0, "Jack-in-the-box — supervisor J-lens en el punto de commit",
            ha="center", fontsize=16, fontweight="bold", color="#1a237e")
    ax.text(50, 97.75, "Qwen2.5-1.5B-Instruct · repo github.com/JCMelendezT/jlens-reactor-guard",
            ha="center", fontsize=9.5, color="#546e7a")

    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="#fafbfc")
    plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()