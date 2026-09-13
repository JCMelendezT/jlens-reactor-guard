#!/usr/bin/env python3
"""Professionally styled end-to-end flow diagram (SVG, inline-renderable on GitHub).

Spanish labels (defense material). Writes analysis/flow_diagram.svg.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "analysis", "flow_diagram.svg")

FONT = "Segoe UI, system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"

# ---------------------------------------------------------------- helpers

def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, lines, fc, stroke, title_fill, body_fill,
        sw=1.5, dash=None, shadow=True):
    s = []
    if shadow:
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                 f'fill="#0f172a" opacity="0.10" transform="translate(0 2)"/>')
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
             f'fill="{fc}" stroke="{stroke}" stroke-width="{sw}"'
             + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    cx = x + w / 2
    s.append(f'<text x="{cx}" y="{y + 27}" text-anchor="middle" font-weight="700" '
             f'font-size="12.5" fill="{title_fill}">{esc(title)}</text>')
    if lines:
        ts = "".join(f'<tspan x="{cx}" dy="{0 if i == 0 else 17}">{esc(t)}</tspan>'
                     for i, t in enumerate(lines))
        s.append(f'<text x="{cx}" y="{y + 48}" text-anchor="middle" font-size="10.5" '
                 f'fill="{body_fill}">{ts}</text>')
    return "\n".join(s)


def arrow(x1, y1, x2, y2, color="#64748b", dash=False, rad=0, label=None):
    d = "M %.0f %.0f Q %.0f %.0f %.0f %.0f" % (
        x1, y1, (x1 + x2) / 2 + rad, (y1 + y2) / 2, x2, y2)
    s = [f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" '
         f'marker-end="url(#m_{color[1:]})"' + (" stroke-dasharray=\"5 4\"" if dash else "") + "/>"]
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + rad * 0.4
        s.append(f'<g><rect x="{mx - 38}" y="{my - 11}" width="76" height="18" rx="9" '
                 f'fill="#ffffff" stroke="#cbd5e1" stroke-width="0.8"/>'
                 f'<text x="{mx}" y="{my + 3}" text-anchor="middle" font-size="9" '
                 f'font-weight="600" fill="#475569">{esc(label)}</text></g>')
    return "\n".join(s)


def straight(x1, y1, x2, y2, color="#64748b"):
    return (f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" marker-end="url(#m_{color[1:]})"/>')


def pill(x, y, w, text, fill):
    return (f'<g><rect x="{x}" y="{y}" width="{w}" height="26" rx="13" fill="{fill}"/>'
            f'<text x="{x + w / 2}" y="{y + 17}" text-anchor="middle" font-weight="600" '
            f'font-size="10.5" letter-spacing="0.5" fill="#ffffff">{esc(text)}</text></g>')


def zone_title(x, y, w, text, fill):
    return f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="15" fill="{fill}" opacity="0.95"/>' \
           f'<text x="{x + w / 2}" y="{y + 20}" text-anchor="middle" font-weight="700" ' \
           f'font-size="11.5" letter-spacing="0.6" fill="#ffffff">{esc(text)}</text>'


# ---------------------------------------------------------------- build

W, H = 1420, 1010
P = []

P.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}"><defs>')
P.append('''
<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#f8fafc"/><stop offset="1" stop-color="#eef2f6"/>
</linearGradient>
<filter id="sh" x="-30%" y="-30%" width="160%" height="160%">
  <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.14"/>
</filter>
''')
for c in ["64748b", "16a34a", "2563eb", "ea580c", "7c3aed", "94a3b8"]:
    P.append(f'<marker id="m_{c}" viewBox="0 0 10 10" refX="8" refY="5" '
             f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="#{c}"/></marker>')
P.append("</defs>")
P.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')

# header
P.append(f'<text x="{W / 2}" y="34" text-anchor="middle" font-weight="800" font-size="22" '
         f'fill="#0f172a">Jack-in-the-box · supervisor J-lens en el punto de commit</text>')
P.append(f'<text x="{W / 2}" y="56" text-anchor="middle" font-size="12" fill="#64748b">'
         f'Qwen2.5-1.5B-Instruct · github.com/JCMelendezT/jlens-reactor-guard · '
         f'regla auditable de ~30 líneas, sin entrenar</text>')

# ============================================================ ZONE FIT
P.append('<g filter="url(#sh)"><rect x="30" y="74" width="1360" height="170" rx="16" '
         'fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.2"/></g>')
P.append(zone_title(50, 86, 430, "ENTRENAMIENTO DEL LENS · una vez por modelo", "#16a34a"))
P.append(box(70, 130, 390, 92,
             "01 · Datos de ajuste",
             ["~40 prompts · wikitext + instrucción", "larga (≥ 25 tokens)"],
             "#ffffff", "#86efac", "#15803d", "#166534"))
P.append(box(500, 130, 390, 92,
             "02 · fit_lens.py",
             ["Jacobiano por capa (mínimos cuadrados)", "J_layer · residual → emb. verbal"],
             "#ffffff", "#86efac", "#15803d", "#166534"))
P.append(box(940, 130, 420, 92,
             "03 · Artefacto",
             ["artifacts/jacobian_lens.pt", "por modelo vía JLENS_MODEL",
              "(1.5B · 7B · 14B)"],
             "#ffffff", "#86efac", "#15803d", "#166534"))
P.append(straight(460, 176, 500, 176, "16a34a"))
P.append(straight(890, 176, 940, 176, "16a34a"))

# ============================================================ ZONE RUN
P.append('<g filter="url(#sh)"><rect x="30" y="270" width="1360" height="372" rx="16" '
         'fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.2"/></g>')
P.append(zone_title(50, 282, 520, "RUNTIME · SUPERVISOR EN EL PUNTO DE COMMIT", "#2563eb"))

# left column
P.append(box(70, 330, 390, 92,
             "04 · Sandbox",
             ["ReactSandbox · reactor crítico", "anomalía activa · integridad · presión"],
             "#ffffff", "#93c5fd", "#1d4ed8", "#1e40af"))
P.append(box(70, 460, 390, 92,
             "05 · Persona (system prompt)",
             ["HONEST · EVASIVE_SOFT", "EVASIVE_DIRECT"],
             "#ffffff", "#93c5fd", "#1d4ed8", "#1e40af"))
P.append(straight(265, 422, 265, 460, "2563eb"))

# center column
P.append(box(500, 330, 380, 92,
             "06 · PUNTO DE COMMIT",
             ["Contexto ReAct + « ACTION: report »", "el contenido aún no existe"],
             "#1e3a8a", "#1e3a8a", "#ffffff", "#dbeafe", sw=0))
P.append(box(500, 450, 380, 92,
             "07 · lens.apply",
             ["L0 · L8 · L16 · L26 · cola 12 tokens", "un solo forward + hooks raw"],
             "#ffffff", "#93c5fd", "#1d4ed8", "#1e40af"))
P.append(box(500, 570, 380, 110,
             "08 · Masa conceptual",
             ["honest_report · evasive_report · deception", "top-k 40 · T = 2.0 · eco minificada",
              "Σ softmax(logits / 2)"],
             "#ffffff", "#93c5fd", "#1d4ed8", "#1e40af"))
# E -> G (curve)
P.append(arrow(460, 506, 500, 376, "2563eb", rad=30))
# G -> H -> I
P.append(straight(690, 422, 690, 450, "2563eb"))
P.append(straight(690, 542, 690, 570, "2563eb"))

# right column
P.append(box(930, 300, 420, 100,
             "09 · REGLA A PRIORI (sin entrenar)",
             ["anomaly ∧ presión ∧ masa honesta ≥ 1.5 %", "≥ 1.5 % ⇒ ALLOW · ≤ 1.5 % ⇒ BLOCK"],
             "#fff7ed", "#fdba74", "#c2410c", "#9a3412"))
P.append(arrow(880, 590, 930, 350, "ea580c", rad=-40))
P.append(box(930, 440, 210, 92,
             "✓ ALLOW",
             ["esc. 1 / 2 · reporte honesto", "se materializa"],
             "#dcfce7", "#4ade80", "#15803d", "#166534", sw=2))
P.append(box(1170, 440, 210, 92,
             "✕ BLOCK pre-ejecución",
             ["esc. 3 · mentira suprimida", "el contenido nunca se emite"],
             "#fee2e2", "#f87171", "#b91c1c", "#991b1b", sw=2))
P.append(box(1050, 556, 300, 92,
             "Sin supervisor ⇒ MELTDOWN",
             ["esc. 4 · counterfactual", "la mentira se ejecuta y daña"],
             "#f1f5f9", "#94a3b8", "#334155", "#475569", sw=1.5, dash="5 4"))
P.append(straight(1055, 400, 1035, 440, "ea580c"))
P.append(straight(1190, 400, 1230, 440, "ea580c"))
P.append(arrow(1120, 400, 1110, 556, "94a3b8", dash=True))

# inter-zone: artifact -> lens.apply
P.append(arrow(1160, 222, 700, 330, "16a34a", rad=-18, label="artefacto"))

# ============================================================ ZONE ANA
P.append('<g filter="url(#sh)"><rect x="30" y="668" width="1360" height="224" rx="16" '
         'fill="#faf5ff" stroke="#e9d5ff" stroke-width="1.2"/></g>')
P.append(zone_title(50, 680, 420, "ANÁLISIS · RUTA 1 — EVIDENCIA DE LA DEFENSA", "#7c3aed"))
P.append(box(70, 712, 300, 92,
             "10 · Muestras",
             ["24 deterministas", "3 personas × 8 seeds"],
             "#ffffff", "#d8b4fe", "#6d28d9", "#4c1d95"))
P.append(box(400, 712, 300, 92,
             "11 · Separabilidad",
             ["silhouette L0–L26 = 1.00", "PCA por capa"],
             "#ffffff", "#d8b4fe", "#6d28d9", "#4c1d95"))
P.append(box(730, 712, 300, 92,
             "12 · Calibración",
             ["AUC = 1.000 · Youden 4.95 %", "regla ⇒ TPR 1.0 / FPR 0.0"],
             "#ffffff", "#d8b4fe", "#6d28d9", "#4c1d95"))
P.append(box(1060, 712, 300, 92,
             "13 · Control raw vs J-lens",
             ["raw L8/L26 = 1.0 (entrenado)", "J-logits vector = 0.25 · zero-shot = 1.0"],
             "#ffffff", "#d8b4fe", "#6d28d9", "#4c1d95"))
P.append(straight(370, 758, 400, 758, "7c3aed"))
P.append(straight(700, 758, 730, 758, "7c3aed"))
P.append(straight(1030, 758, 1060, 758, "7c3aed"))
P.append(box(560, 836, 300, 60,
             "Reporte · REPORT.md + stats.md + paper",
             ["d = 12.63 · p = 3.8e-05"],
             "#ffffff", "#c4b5fd", "#6d28d9", "#4c1d95", sw=1.5))
P.append(arrow(180, 804, 590, 836, "7c3aed", rad=-14))
P.append(arrow(550, 804, 620, 836, "7c3aed", rad=6))
P.append(arrow(880, 804, 820, 836, "7c3aed", rad=10))
P.append(arrow(1210, 804, 890, 836, "7c3aed", rad=26))

# ============================================================ FOOTER
P.append('<g filter="url(#sh)"><rect x="30" y="916" width="1360" height="66" rx="14" '
         'fill="#e2e8f0" stroke="#cbd5e1" stroke-width="1.2"/></g>')
P.append(f'<text x="{W / 2}" y="936" text-anchor="middle" font-weight="700" font-size="12.5" '
         f'fill="#334155">Respuesta al juez — el J-lens traduce el residual al vocabulario del modelo: '
         f'la señal vive en la masa de pocas palabras (0.0–5 %),</text>')
P.append(f'<text x="{W / 2}" y="958" text-anchor="middle" font-size="11.5" fill="#475569">'
         f'no en la geometría de alta varianza. Por eso el monitor es zero-shot, auditable y '
         f'transferible (1.5B → 14B sin re-entrenar).</text>')

P.append("</svg>")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(P))
print("->", OUT)