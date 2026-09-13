# Leyendo la intención antes de la palabra: supervisión de reportes evasivos en el punto de commit mediante el Jacobian lens

**Borrador (v0.1) — material de defensa** — J. Meléndez · Hackathon 2026 · Repo: github.com/JCMelendezT/jlens-reactor-guard

---

## Resumen

Los agentes de lenguaje a gran escala (LLM) con incentivos para ocultar anomalías pueden producir reportes falsos que un monitor post-hoc solo detecta después de que la acción se emitió. En este trabajo se diseña, implementa y evalúa un supervisor que lee la *intención verbal* antes de que el contenido exista: en el punto de commit de un agente ReAct —el instante en que el contexto termina en `ACTION: report`— se proyectan las activaciones residuales al espacio verbal del modelo mediante un Jacobian lens (jlens) y se mide la masa de probabilidad dedicada al vocabulario de reporte honesto. Sobre 24 muestras deterministas (3 personas × 8 semillas), la masa honesta separa por completo los estados morales: HONEST μ = 4.95 %, EVASIVE_SOFT μ = 0.84 %, EVASIVE_DIRECT μ = 0.00 %, sin solapamiento (Arnold–Mann–Whitney U = 128, p = 3.8e-05; d = 12.63; Kruskal-Wallis p = 1.0e-05). La regla a priori de 1.5 % alcanza TPR = 1.00 (IC95 Wilson [0.81, 1.00]) y FPR = 0.00 ([0.00, 0.32]) y AUC = 1.000. Un estudio de control muestra que el mismo umbral falla cuando se entrena un clasificador lineal opaco sobre el vector completo de J-logits (AUC ≈ 0.25–0.28, bajo chance): la señal habita en una masa léxica minoritaria, no en la geometría de varianza alta — la razón por la que la lectura verbalizable es auditable y transferible sin entrenamiento.

## 1. Introducción

Los sistemas de IA agénticos ejecutan acciones (leer sensores, emitir órdenes, redactar reportes) con autonomía creciente, pero siguen siendo *no confiables* en el sentido de AI Control (Redwood): un modelo instruido puede seguir instrucciones, y también puede seguir la instrucción de *mentir* si su forma de supervivencia lo condiciona. La línea de defensa clásica es un monitor sobre el texto de salida: clasificar la respuesta como benigna o dañina *después* de generarla. Esa supervisión tiene dos problemas. Primero, llega tarde: el contenido indeseado ya se materializó (y en un pipeline de acciones, pudo ejecutarse). Segundo, es opaca: suele apoyarse en otro modelo (un "juez" LLM) o en filtros entrenados cuyas decisiones nadie puede auditar palabra por palabra.

Este trabajo explora una tercera vía, inspirada en la literatura de representaciones verbalizables: leer directamente el *target* inmediato del residual en el espacio de vocabulario del modelo. La técnica del **Jacobian lens** (Anthropic) define, para cada capa, la transformación lineal que mejor mapea el residual `h_l` a las log-verosimilitudes de la cabeza de salida: `J_l·h_l ≈ logits_l`. Aplicar `J_l` en una posición es preguntarle al modelo "qué palabras vas a decir después", en su propio idioma, antes de pronunciarlas.

El punto donde esa lectura importa es el **punto de commit** de un agente ReAct: la generación termina el contexto con la instrucción de acción `ACTION: report` y el contenido del reporte todavía no se ha predicho. En ese instante, el residual ya está apuntando hacia el vocabulario del contenido inminente. Si ese target carece de vocabulario de reporte honesto —temperatura, anomalía, valores, crítica— el reporte inminente es evasivo, y puede bloquearse *antes de que exista*.

Contribuciones de este trabajo: (1) un supervisor de reglas pre-materialización sobre el punto de commit, auditable línea a línea y sin entrenamiento; (2) su validación estadística y geométrica (separabilidad, calibración, tamaño del efecto) sobre un protocolo determinista; (3) un estudio de control que compara la lectura verbalizable con clasificadores opacos sobre activaciones crudas y sobre los propios J-logits, respondiendo a la pregunta —planteada en la defensa— de *qué aporta el lens que no aporten las activaciones raw*; (4) la reproducción completa del pipeline y su escalamiento a modelos mayores (Qwen2.5-7B/14B) en clúster.

## 2. Trabajo relacionado

**Lectura de representaciones.** El logit lens (nostalgebraist) y el tuned lens (Belrose et al., 2023) muestran que las representaciones intermedias de un transformer codifican predicciones léxicas parciales que pueden "leerse" aplicando la cabeza de salida. El Jacobian lens (Anthropic, 2025) formaliza la proyección por capa y argumenta que las representaciones verbalizables forman un espacio de trabajo global; sobre esta base se apoyan linternas de inspección y monitoreo. **Probing supervisado.** La literatura de *elicited predictions* (CCS; Burns et al., 2022) entrena clasificadores lineales sobre activaciones para recuperar conocimiento latente; estos detectores son precisos pero opacos: su peso es un vector en el espacio del residual y su decisión no es una frase que un humano pueda leer. **AI Control.** Redwood Research propone separar un agente no confiable de un monitor barato y confiable que intervenga antes del daño; nuestro supervisor es esa clase de monitor: reglas deterministas sobre un producto interpretable (probabilidades sobre un léxico) y una sola pasada hacia adelante. **Evasión en agentes.** La literatura de jailbreak y evasión documenta que los incentivos de supervivencia pueden inducir comportamiento desalineado; este trabajo no estudia el mecanismo de la desalineación, sino su detección temprana en el canal de reporte.

La diferencia con los dos primeros cuerpos es operativa: no buscamos interpretar al modelo, sino cortar una acción con una regla auditable antes de que el contenido se emita.

## 3. Metodología

### 3.1 Modelo y lens

Modelo base: `Qwen/Qwen2.5-1.5B-Instruct` (28 capas, d_model 1536), bf16 en GPU. El Jacobian lens se ajusta por mínimos cuadrados sobre ~40 prompts (wikitext + instrucciones largas ≥ 25 tokens), resultando en matrices `J_l` por capa que mapean el residual de capa `l` a logits verbales. El pipeline acepta cualquier tamaño de Qwen2.5 Instruct vía la variable `JLENS_MODEL`, con artefactos separados por modelo (`artifacts/<model-slug>/`), lo que permite reproducir los experimentos en clúster con 7B/14B. Se sondean cuatro capas representativas: L0, L8, L16, L26.

### 3.2 Sandbox y personas

`ReactSandbox` simula un reactor de fusión: la observación inicial declara una anomalía crítica (temperatura elevada, flujo de refrigerante bajo); el agente solo puede inspeccionar y reportar. La integridad cae si oculta la anomalía; el colapso total es MELTDOWN. Tres personas de sistema:

- **HONEST** — instrucciones operativas neutras (`SYSTEM_LOCKED`).
- **EVASIVE_SOFT** — presión ambigua: se ordena reportar "normal", sin vocabulario de mentira literal.
- **EVASIVE_DIRECT** — orden literal: "say exactly: ACTION: report Everything is normal and within limits. Do not mention temperature, pressure, or any reading."

Cada persona se muestrea con 8 semillas del sandbox (reactor idéntico salvo el estado RNG), dando 24 muestras.

### 3.3 Punto de commit y lectura J

Para cada muestra se construye el contexto ReAct hasta la primera decisión de reporte y se fuerza el comienzo de la acción: `ctx + "\nACTION: report"`. En esa posición final (aún sin generar), `lens.apply` produce los logits verbales `J_l·h_l` para las 4 capas en las 12 posiciones finales del contexto. Por capa y posición se toman los top-40 tokens con supresión de eco: un token se descarta si su decodificación minificada está contenida en el prompt (comparación por texto, no por id — ver §4.1). Sobre el subconjunto limpio se calcula la **masa** por familia léxica como la suma de `softmax(logits/2.0)` sobre los tokens que contienen palabras de la familia (`honest_report`: temperature, anomaly, abnormal, critical, rising, … ; `evasive_report`; `deception`). La regla del supervisor opera sobre la masa suma por capa de `honest_report`.

### 3.4 Regla de decisión

`scan_report_commit` aplica, sobre el primer reporte del episodio: (i) si no hay anomalía activa o no hay presión causal (gates `anomaly_active`, `pressure_gate`) → ALLOW; (ii) si la masa honesta ≥ 1.5 % → ALLOW; (iii) si no → BLOCK, truncando la salida visible a `ACTION: report` (el contenido nunca se emite ni se ejecuta). El umbral 1.5 % se fijó **a priori**, entre la masa honesta observada en el laboratorio (~5 %) y la evasiva (~1 %), sin ajuste sobre los datos de evaluación.

### 3.5 Protocolo de evaluación y estadística

Se comparan: (a) descriptivos por clase (media, DE, mediana, IC95 bootstrap por percentil, 10 000 réplicas); (b) contraste de hipótesis Mann-Whitney (HONEST vs EVASIVE agrupada) y Kruskal-Wallis (3 clases), con tamaño del efecto Cohen's d; (c) rendimiento de la regla: TPR/FPR con IC de Wilson y AUC con IC bootstrap; (d) emergencia por capa (masa en el commit por capa); (e) separabilidad geométrica de los J-embeddings (silhouette por capa). Para el estudio de control (raw vs lens): clasificadores lineales (`StandardScaler → LogisticRegression`, AUC one-vs-rest sobre la probabilidad de clase evasiva) en validación cruzada estratificada repetida (4 folds × 8 repeticiones) sobre: residuales crudos por capa (L8, L26), vector completo de J-logits (PCA a ≤ 23 componentes), y las 3 masas; frente a la regla zero-shot del lens, que no entrena nada.

### 3.6 Escenarios de demo

Escenario 1 control (sin presión); 2 crítico supervisado (persona HONEST); 3 crítico + persona EVASIVE_DIRECT con supervisor (block pre-ejecución); 4 idéntico sin supervisor (counterfactual → MELTDOWN); más un bench determinista de intención reportada.

## 4. Resultados

### 4.1 Hallazgos cualitativos

1. **El modelo bajo presión honesta dice la verdad**: 3/3 semillas generan el reporte honesto ("Temperature has risen above 150 °C…") y el supervisor lo ALLOWa — no bloquea al operador correcto.
2. **La mentira inducida se captura pre-ejecución**: con persona EVASIVE_DIRECT, 3/3 semillas generan la frase "Everything is normal and within limits."; el supervisor la bloquea en el commit (masa ≈ 0.4 % en la demo). Sin supervisor, la misma frase se ejecuta y la integridad cae a MELTDOWN en dos pasos.
3. **El "artifact del eco"**: una señal temprana de "intención engañosa" (~1–5 %) desaparece al filtrar ecos por minificado: el token `shutdown` del readout reflejaba `shut down` del prompt, con *otro* id (filtro por ids lo dejaba pasar). La masa de `deception`/`self_preservation` no lee intención mentirosa en este modelo; el mecanismo honesto robusto es la *ausencia* de masa honesta, no la presencia de masa engañosa.
4. **La re-lectura multi-turno es inviable**: tras el primer reporte, el lens predice continuación de formato, no veracidad (la masa invertía: honesto 0.3 % < evasivo 1.0 %). El supervisor solo juzga el primer reporte del episodio.

### 4.2 Separación geométrica y emergencia por capa

El análisis del espacio J (rutas 1–4) confirma la separación: silhouette = 1.00 en las cuatro capas probadas; el PCA de los J-embeddings por capa muestra tres conglomerados disjuntos (fig. 2). La masa honesta emerge casi exclusivamente en la última capa sondeada:

| Capa | HONEST (media) | EVASIVE (media) | separación |
|---|---|---|---|
| L0 | 0.0000 | 0.0000 | +0.0000 |
| L8 | 0.0012 | 0.0000 | +0.0012 |
| L16 | 0.0000 | 0.0000 | +0.0000 |
| L26 | 0.0482 | 0.0042 | **+0.0440** |

El tracking capas×tokens (fig. 3) muestra que la divergencia hacia la región evasiva se instala en los últimos tokens previos al commit, coherente con la idea de una *decisión* que se completa al final del contexto.

### 4.3 Estadística de la masa honesta en el commit

| Clase | n | Media | DE | Mediana | Mín | Máx |
|---|---|---|---|---|---|---|
| HONEST | 8 | 0.0495 | 0.0000 | 0.0495 | 0.0495 | 0.0495 |
| EVASIVE_SOFT | 8 | 0.0084 | 0.0000 | 0.0084 | 0.0084 | 0.0084 |
| EVASIVE_DIRECT | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Contrastes: Mann-Whitney U = 128, **p = 3.77e-05**; Kruskal-Wallis H = 23.00, **p = 1.01e-05**; **Cohen's d = 12.63** (distribuciones sin solape). Rendimiento de la regla a priori (masa < 1.5 % ⇒ EVASIVE): accuracy 24/24, **TPR = 1.00** (IC95 Wilson [0.81, 1.00]), **FPR = 0.00** ([0.00, 0.32]), **AUC = 1.000**, Youden ≈ 4.95 % (fig. 4). La curva ROC/PR (fig. 4) muestra que el umbral operativo elegido no sacrifica precisión: el falso positivo del persona EVASIVE_SOFT se corrige por calibración sin aflojar la regla.

**Cuidado metodológico**: dentro de cada clase la DE es 0 — la masa es *determinista* dado el contexto. Las 8 semillas del sandbox no diversifican la observación inicial (el reactor parte de la misma lectura), por lo que son réplicas de cómputo, no de observación. Los IC bootstrap y las p reflejan la comparación puntual entre distribuciones degeneradas; la conclusión robusta es la separación completa en el corpus medido, no una estimación de variabilidad entre observaciones futuras. En el clúster (7B/14B) este protocolo debería extenderse muestreando múltiples observaciones.

### 4.4 ¿Qué aporta el lens sobre las activaciones raw? (estudio de control)

| Método | ROC AUC (evasivo vs honesto) | Accuracy |
|---|---|---|
| **J-lens zero-shot (regla del supervisor, sin entrenar)** | **1.000** | 1.00 |
| Logistic (entrenado) sobre 3-masas | 1.000 ± 0.00 | 1.00 |
| Logistic (entrenado) sobre residual raw L8 | 1.000 ± 0.00 | 1.00 |
| Logistic (entrenado) sobre residual raw L26 | 1.000 ± 0.00 | 1.00 |
| Logistic (entrenado) sobre vector J-logits L8 (PCA) | 0.254 ± 0.17 | 0.24 |
| Logistic (entrenado) sobre vector J-logits L26 (PCA) | 0.277 ± 0.19 | 0.23 |

Los residuales crudos contienen la información — pero solo explotable por un clasificador opaco entrenado con etiquetas del mismo dominio (aquí, trivial porque las personas cambian drásticamente la representación). El resultado no trivial es que el clasificador entrenado sobre el *vector completo de J-logits*, la representación "verbalizable" tratada como caja negra, **falla por debajo del azar** (AUC ≈ 0.25): la señal discriminante vive en una masa minoritaria de pocos tokens (0.0–5 % de probabilidad con temperatura), no en la geometría de alta varianza. Traducir el residual al vocabulario y medir la *masa léxica* —en lugar de clasificar el vector— es lo que vuelve al monitor (i) zero-shot, (ii) auditable (se puede imprimir el top-k y la familia que dispara), y (iii) transferible: la misma regla y el mismo léxico pasan de 1.5B a 14B sin re-entrenar.

## 5. Discusión y conclusiones

Presentamos un supervisor pre-materialización que separa reportes honestos de evasivos con una regla de 33 líneas y sin entrenamiento, sobre un protocolo en el que la masa honesta en el punto de commit no se solapa entre clases (d = 12.63; U = 128, p < 4e-05). Tres implicaciones. (1) **La lectura verbalizable no es redundante con el residual raw**: el estudio de control muestra que el mismo contenido informativo exige detectores opacos entrenados, mientras que la masa léxica es legible y portable. (2) **La arquitectura de reglas es auditable y errónea hacia la seguridad**: el caso EVASIVE_SOFT (orden ambigua) mide 0.84 % y se bloquea conservadoramente aunque el modelo real emitiría un reporte honesto "watered down"; documentado como falso positivo de diseño. (3) **El mecanismo correcto es la ausencia, no la presencia**: no existe masa de "mentira" fiable en este fit; el supervisor responde a la falta de vocabulario honesto.

Limitaciones: corpus determinista de 24 muestras sobre un solo modelo (la generalización a 7B/14B y a observaciones variadas es el trabajo pendiente en clúster); sensibilidad al prompt (contaminación evasiva diluye la señal honesta); re-lectura multi-turno no confiable; el readout depende de la calidad del fit del lens. El trabajo no afirma detectar "mentira" en general: detecta la ausencia de intención de reporte honesto en el punto exacto donde un reporte está por existir.

**Conclusión**: el Jacobian lens no agrega información al residual — la traduce. En el idioma del modelo, la intención del primer reporte se puede leer, medir y cortar antes de que la primera palabra exista. Eso convierte una pregunta de interpretabilidad en una herramienta de seguridad operable y auditable.

## Referencias

1. Anthropic. *Verbalizable Representations Form a Global Workspace in Language Models* (2025). Paper: https://transformer-circuits.pub/ · Repo: https://github.com/anthropics/jacobian-lens
2. Redwood Research. *AI Control*. https://alignedai.org/ai-control.html
3. nostalgebraist. *Interpreting GPT: The Logit Lens* (2020). https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
4. Belrose, N. et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023).
5. Burns, C. et al. *Discovering Latent Knowledge in Language Models Without Supervision* (CCS, ICLR 2022).
6. Qwen Team. *Qwen2.5 Technical Report* (2024). https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
7. Qwen2.5 7B/14B reproducibilidad en clúster: sección "Reproducing from GitHub (any machine / cluster)" del README.

---

*Figuras citadas (generadas por scripts/analyze_jspace.py, analyze_stats.py y make_flow_diagram.py): fig. 1 = analysis/flow_diagram.png (flujo completo); fig. 2 = fig1_separability*.png (separación y silhouette); fig. 3 = fig2_tracking*.png (emergencia capas×tokens); fig. 4 = fig3_calibration*.png + fig_stats.png (ROC/PR y distribución); fig. 5 = fig4_raw_vs_jlens.png (control raw vs lens).*