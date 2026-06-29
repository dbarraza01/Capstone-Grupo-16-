# Plan Definitivo — Análisis de Sensibilidad a Posteriori
**Capstone Grupo 16 — Evaluación de Robustez del Pipeline de Predicción de LOS**

---

## 1. ¿Por Qué Estos Escenarios y No Otros?

De los múltiples escenarios evaluados en planes preliminares, se seleccionaron **3 análisis de sensibilidad principales y 1 validación de estabilidad adicional** que maximizan el impacto demostrativo de robustez cubriendo las dimensiones exigidas por los lineamientos del curso:

| Dimensión del lineamiento | Escenario elegido | ¿Por qué este y no otro? |
|---|---|---|
| *"Cómo reacciona frente a cambios en supuestos"* | **Escenario 1: Variación del umbral PLOS** | Es el supuesto más crítico de todo el pipeline: define la pregunta binaria de la Etapa 1. Si el modelo colapsa al mover este umbral, toda la arquitectura de dos etapas pierde validez. Escenarios como "sin log1p" o "sin stacking" son ablaciones de diseño, no de supuestos clínicos. |
| *"Evaluar qué tan bien funciona + limitaciones"* | **Escenario 2: Ablation Study de features y componentes** | Demuestra qué variables y qué etapas del pipeline son realmente indispensables y cuáles son redundantes. Al incluir la eliminación de la probabilidad del clasificador, responde a la pregunta de si la arquitectura de 2 etapas es justificada frente a un modelo simple de 1 etapa. |
| *"Cómo ayuda en la toma de decisiones"* | **Escenario 3: Sensibilidad al punto de operación del clasificador** | Es el único análisis que conecta directamente el modelo con una **decisión operacional real**: ¿a partir de qué probabilidad se emite una alerta de estancia prolongada? Esto es lo que un gestor de camas necesita saber. Ningún otro escenario responde esta pregunta. |
| *"Estabilidad y robustez de los supuestos de modelamiento"* | **Escenario 4 (Validación Adicional): Robustez a Hiperparámetros vecinos** | Permite justificar científicamente el principio *ceteris paribus* (mantener fijos los hiperparámetros en los otros escenarios) al demostrar que el rendimiento no es hiper-sensible a cambios menores en el tuning. Si el modelo es robusto en su vecindad, se valida que el tuning es estable y generalizable. |

### ¿Por qué se descartaron los otros escenarios?

- **Sin transformación log1p:** Es una decisión técnica interna del regresor que no representa un cambio de supuesto que el hospital enfrentaría. Su impacto es menor comparado con los elegidos.
- **Sin segmentación urgente/programado:** La segmentación ya está justificada por la diferencia estadística entre las distribuciones de ambos grupos (MAE urgentes = 5.34 vs. programados = 1.84). Re-demostrarlo aporta menos valor marginal.
- **Inyección de ruido:** Simula errores aleatorios pero no responde preguntas estructurales sobre el modelo. El ablation study es más informativo porque identifica *qué* features importan, no solo *cuánto ruido toleran*.
- **Curva de aprendizaje (tamaño muestral):** Eliminada previamente por decisión del equipo.
- **Bootstrap de partición:** Interesante pero costoso en tiempo de cómputo. La estabilidad del modelo ya se evidencia parcialmente con el gap train-holdout reportado (ratio 1.24 para XGB).
- **Variación de hiperparámetros mediante re-tuning completo arbitrario:** Modificar aleatoriamente parámetros sin un propósito metodológico oscurece el análisis. En su lugar, el **Escenario 4** aborda de forma rigurosa la sensibilidad del tuning evaluando la vecindad del óptimo.

---

## 2. Línea Base (Escenario 0) — Resultados Actuales

Valores de referencia del holdout (n=2,391) con el pipeline completo ya entrenado:

| Métrica | XGBoost | Random Forest | LR (Baseline) |
|---|---|---|---|
| MAE | 2.86 | 3.10 | 3.17 |
| RMSE | 7.17 | 8.29 | 8.86 |
| MedAE | 0.84 | 0.91 | 0.86 |
| ME | -0.96 | -1.33 | -0.90 |
| PUP | 44.7% | 45.5% | 46.9% |
| MAE Asim. (α=2) | 4.77 | 5.31 | 5.20 |
| Precisión PLOS | 0.800 | 0.820 | 0.863 |
| Recall PLOS | 0.588 | 0.538 | 0.495 |
| F1 PLOS | 0.678 | 0.649 | 0.629 |

**No requiere código nuevo.** Se toman directamente de `ml_operacional_entrega3/reports/`.

---

## 3. Escenario 1 — Sensibilidad al Umbral de Definición de PLOS

### ¿Requiere tuning? **NO.** Se reutilizan los `best_params` ya encontrados.

### Motivación Bibliográfica

La elección del umbral que define "estancia prolongada" no es universal ni objetivamente correcta: depende del contexto hospitalario, la distribución del LOS en la población y los objetivos operacionales. La literatura muestra una diversidad significativa:

- **≥ 6 días** (~ media): Zeleke et al. (2023) en urgencias del Hospital Universitario de Bolonia.
- **≥ 7 días**: Goldstein et al. (2022) en su modelo de dos etapas; Lee et al. (2024) en el OMOP CDM de Corea del Sur.
- **≥ 14 días** (~ percentil 80-85): Chrusciel et al. (2022) en la base médico-administrativa francesa PMSI (73.182 hospitalizaciones). **Umbral vigente del pipeline.**
- **≥ 27 días** (~ percentil 95): definición adoptada en la Entrega 2 de este proyecto, coherente con la distribución empírica observada.
- **Percentil 95 específico de la cohorte**: Dettori et al. (2024) en TBI, donde el umbral resultó ser ≥ 24 días.

**Conclusión bibliográfica:** Chrusciel et al. (2022) y Goldstein et al. (2022) reportan que la comparación entre modelos depende del umbral elegido. Informar resultados bajo un único umbral puede introducir un sesgo de selección. Este escenario evalúa si el orden observado entre XGB, RF y LR se mantiene ante distintas definiciones de PLOS.

### Experimento

Re-entrenar el pipeline de dos etapas (clasificador + regresor con stacking) bajo **4 umbrales** usando los `best_params` ya encontrados:

| Variante | Umbral PLOS | Justificación |
|---|---|---|
| A | ≥ 7 días | Goldstein et al. (2022), Lee et al. (2024) |
| B (actual) | ≥ 14 días | Chrusciel et al. (2022), pipeline vigente |
| C | ≥ 21 días | Punto intermedio clínicamente reconocido |
| D | ≥ 27 días | Entrega 2 del proyecto, percentil 95 |

Para cada umbral:
1. Se recalcula la etiqueta binaria (`LOS >= umbral`) para el entrenamiento y el holdout.
2. Se generan nuevas probabilidades Out-Of-Fold (OOF) ejecutando la función de validación cruzada (`generate_oof_probabilities`) con el clasificador configurado con `best_params_clf`.
3. Se re-entrena el regresor con `best_params_reg` + la nueva columna de probabilidad OOF (`prob_los_x`).
4. Se evalúa en el mismo holdout (20%) con las métricas de regresión y clasificación.

**Hipótesis esperada:** El MAE de regresión debería ser razonablemente estable entre umbrales de 7-21 días. En el umbral de 27 días, la clase positiva se vuelve extremadamente minoritaria, lo que dificulta la estimación de la probabilidad y podría degradar el rendimiento. Si el ranking XGB > RF > LR se mantiene en todos los umbrales, la conclusión es robusta.

**Métricas a reportar (CSV):** `umbral_plos`, `mae`, `rmse`, `me`, `pup`, `mae_asimetrico`, `precision_plos`, `recall_plos`, `f1_plos`, `proporcion_plos`.

---

## 4. Escenario 2 — Ablation Study de Features y Componentes

### ¿Requiere tuning? **NO.** Se reutilizan los `best_params` ya encontrados.

### Motivación Bibliográfica

Un *ablation study* consiste en remover sistemáticamente grupos de variables y medir el deterioro en el desempeño del modelo. Es el método estándar en Machine Learning para determinar cuáles features son verdaderamente indispensables vs. cuáles son redundantes.

En el dominio clínico, esta práctica es especialmente relevante porque:

1. **El Charlson Comorbidity Index no siempre mejora la predicción de LOS.** Bottle & Aylin (2014) evaluaron el CCI construido desde datos ICD-10 en 47,698 pacientes con fractura de cadera y encontraron que su poder predictivo para LOS era bajo (R² ajustado 0.007–0.045), aunque sí era válido para mortalidad. Esto no significa que Charlson no aporte en todos los contextos, pero su contribución marginal debe medirse empíricamente y no asumirse. El proyecto ya validó que Escenario B (con Charlson) supera a Escenario A (sin él) en MAE, pero el análisis de ablación formaliza ese hallazgo.

2. **Los modelos basados en árboles son robustos a la eliminación de features poco informativas** gracias a sus mecanismos internos de selección, pero la literatura clínica recomienda reportar la sensibilidad al conjunto de covariables para demostrar que el modelo no depende de un único predictor que pudiera no estar disponible en otros hospitales (Lee et al., 2024).

3. Un estudio comparable de XGBoost con SHAP en gastrectomía (Morinaga et al., 2024) encontró que remover el tipo de cirugía y el volumen hospitalario —las dos variables más importantes según SHAP— degradaba el RMSE de 3.74 a 5.12 días, mientras que remover variables de baja importancia tenía impacto mínimo. Este tipo de análisis indica qué partes del pipeline son críticas para la implementación en otro hospital donde ciertos datos podrían no existir.

### Experimento

Entrenar el modelo XGB con los `best_params` ya encontrados (sin re-tunear) en los siguientes escenarios de features reducidos, evaluando en el mismo holdout:

| Variante | Features/Componentes removidos | Pregunta que responde |
|---|---|---|
| Full (línea base) | Ninguna | Referencia |
| Sin Charlson | Columna `charlson_index` | ¿Es el índice de Charlson indispensable para predecir LOS? |
| Sin capítulos ICD-10 | Columnas que empiezan con `diag_rare_cap_` y `proc_rare_sec_` | ¿Las agrupaciones por capítulo agregan valor sobre los diagnósticos individuales? |
| Solo demográfico-operacional | Mantener SOLO `n_diag_total`, `n_procedimientos`, `es_urgencia`, `mes_ingreso`, `dia_semana_ingreso`, `tiene_diag_primario`, `charlson_index`, `n_diag_primarios`, `n_diag_secundarios`. Remover todas las dummies de diagnósticos (`diag_`) y procedimientos (`proc_`). | ¿Cuánto rendimiento se pierde sin la codificación clínica detallada? |
| Sin Clasificador (1 Etapa) | Remover componente de la Etapa 1. Entrenar XGBRegressor de forma directa excluyendo la probabilidad del clasificador (`prob_los_14`). | ¿La arquitectura en 2 etapas con la probabilidad del clasificador realmente aporta valor neto frente a un regresor directo de 1 etapa? |

Para cada variante se re-entrena el pipeline completo sin re-tunear hiperparámetros.

**Hipótesis esperada:** 
- Remover Charlson tendrá un impacto menor (< 5% de degradación en MAE), confirmando que el modelo no depende críticamente de él pero sí se beneficia de tenerlo.
- Remover los capítulos ICD-10 raros tendrá impacto mínimo, ya que los diagnósticos individuales ya capturan esa información.
- El escenario "solo demográfico-operacional" mostrará una degradación significativa (>15%), demostrando que la codificación clínica detallada es el componente más valioso del pipeline.
- El escenario "sin clasificador" mostrará una degradación intermedia, demostrando el impacto que la probabilidad del clasificador de la primera etapa tiene al actuar como guía del regresor.

**Métricas a reportar (CSV):** `variante`, `mae`, `rmse`, `recall_plos`, `f1_plos`, `delta_mae_pct`.

---

## 5. Escenario 3 — Sensibilidad al Punto de Operación del Clasificador (Políticas Clínicas)

### ¿Requiere tuning? **NO.** Se utilizan los modelos ya entrenados y guardados en `.joblib`.

### Motivación Bibliográfica

El clasificador binario (¿LOS ≥ 14 días?) emite una probabilidad continua `[0, 1]` y luego aplica un umbral de decisión para emitir la alerta PLOS. Este umbral **no es neutro**: determina el trade-off entre precisión y recall, y su elección óptima depende del contexto operacional del hospital.

Mahajan et al. (2023) implementaron un sistema de alertas codificadas por color en una red hospitalaria de EE.UU. y reportaron que el umbral de alerta fue ajustado post-implementación tras retroalimentación del equipo médico: inicialmente fijado en 0.5 (precisión 0.698, recall 0.598), fue reducido a 0.4 cuando el equipo clínico priorizó no perderse pacientes candidatos a alta (recall 0.746, precisión 0.621). Este hallazgo demuestra que el umbral óptimo es una decisión conjunta entre el modelo y los usuarios.

En el contexto de PLOS, la asimetría de costos es clara:
- **Un falso negativo** (no detectar a un paciente que se quedará mucho tiempo) genera sobreocupación, cancelación de cirugías y colapso operacional.
- **Un falso positivo** (alertar sobre un paciente que se va pronto) solo ocupa tiempo de coordinación innecesaria.

Por lo tanto, la gestión hospitalaria puede preferir un punto de operación con **recall más alto** aunque la precisión baje.

### Experimento (Sensibilidad de Políticas)

Para evitar sesgar el análisis hacia la optimización directa en el holdout, evaluamos el rendimiento operacional bajo **3 políticas clínicas predefinidas con supuestos operacionales fijos**:

| Política Clínica | Umbral fijo | Supuesto Operacional del Hospital |
|---|---|---|
| **Política B: Alta Seguridad (Alto Recall)** | **0.35** | *"Prioridad absoluta de camas críticas"*. Tolera falsos positivos con tal de no perderse ningún paciente de estancia larga en riesgo de colapsar la unidad. |
| **Política A: Base / Equilibrio (Default)** | **0.50** | *"Balance Operacional Standard"*. Punto neutro del clasificador. |
| **Política C: Eficiencia (Alta Precisión)** | **0.65** | *"Alertas confiables para evitar fatiga"*. Evita alertar al personal innecesariamente, emitiendo notificaciones solo cuando el riesgo es muy evidente. |

Para cada política se calcularán las métricas en el holdout utilizando las probabilidades `prob_riesgo` producidas por el clasificador entrenado.

**Métricas a reportar (CSV):**
- Archivo `escenario_3_puntos_operacion.csv`: `politica_clinica`, `umbral_probabilidad`, `tp`, `fp`, `fn`, `tn`, `precision`, `recall`, `f1`.
- Archivo `escenario_3_curva_pr.csv`: `precision`, `recall`, `thresholds` (para graficar la curva PR completa).

**Interpretación clínica:** Explicar el impacto de las falsas alarmas (camas "bloqueadas" por alertas erróneas) vs. pacientes de estancia larga no detectados bajo cada política.

---

## 6. Escenario 4 (Validación Adicional) — Robustez a Hiperparámetros (Estabilidad del Tuning)

### ¿Requiere tuning? **NO.** Se evalúan configuraciones fijas vecinas al óptimo.

### Motivación Bibliográfica

En el aprendizaje automático, el sobreajuste al conjunto de validación durante la búsqueda de hiperparámetros (*hyperparameter overfitting*) constituye un riesgo latente. Bergstra y Bengio (2012) destacan que los espacios de búsqueda suelen contener tanto regiones "planas" (estables frente a pequeños cambios) como "agudas" (altamente sensibles). Si el punto óptimo del tuning se encuentra en una región aguda, cualquier ligera alteración en las características de entrada o de entrenamiento podría provocar una degradación catastróficamente alta del desempeño en producción.

Probst et al. (2019) analizan la sensibilidad de algoritmos basados en árboles (como Random Forest y XGBoost) y demuestran que, si bien son relativamente robustos, es metodológicamente crucial validar que variaciones moderadas en parámetros clave de regularización, tasa de aprendizaje y submuestreo (`learning_rate`, `max_depth`, `subsample`, `min_child_weight`) produzcan fluctuaciones de rendimiento marginales. Esta validación robustece el principio *ceteris paribus* ("todo lo demás constante") aplicado en los Escenarios 1 y 2, confirmando que las conclusiones generales del pipeline no dependen de forma crítica de una parametrización única y ultra-específica.

### Experimento

Partiendo de los hiperparámetros óptimos de XGBoost (cargados de `best_params_clf.json` y `best_params_reg.json` en `ml_operacional_entrega3/XGB/`), se entrenará y evaluará el pipeline de dos etapas bajo **3 configuraciones vecinas alternativas**:

1. **Variante A: Conservadora (Mayor Regularización / Más Simple)**
   - Reducir `max_depth` en 1 unidad (mínimo de 2).
   - Reducir `learning_rate` multiplicándolo por 0.8 (ej: de 0.05 a 0.04).
   - Incrementar `min_child_weight` (si está presente en el JSON) en +2 unidades.
   - *Objetivo:* Verificar si una simplificación moderada del modelo preserva el rendimiento o si genera subajuste.

2. **Variante B: Compleja (Menor Regularización / Mayor Capacidad)**
   - Incrementar `max_depth` en 1 unidad.
   - Incrementar `learning_rate` multiplicándolo por 1.2 (ej: de 0.05 a 0.06).
   - Reducir a la mitad (multiplicar por 0.5) los parámetros de regularización L1 y L2 (`reg_alpha` y `reg_lambda` si están declarados en el JSON).
   - *Objetivo:* Medir si un modelo con mayor capacidad sobreajusta el holdout de forma severa.

3. **Variante C: Perturbación Estocástica de Muestreo**
   - Reducir en 0.1 los coeficientes de submuestreo (`subsample` y `colsample_bytree` si están declarados en el JSON). Si por ejemplo son 0.8, se reducen a 0.7 (acotado a un rango válido mínimo de 0.5).
   - *Objetivo:* Comprobar la resiliencia del pipeline frente al muestreo de filas y columnas en la construcción de los árboles individuales.

**Regla de perturbación:** Si uno de los parámetros mencionados, como `min_child_weight` o `reg_alpha`, no está declarado explícitamente en los archivos JSON de mejores parámetros, no se modifica ni se incorpora. En ese caso se conserva el valor predeterminado del algoritmo. Solo se perturban los hiperparámetros incluidos en el ajuste previo.

**Métricas a reportar (CSV):** `variante_hiperparametros`, `mae`, `rmse`, `recall_plos`, `f1_plos`, `delta_mae_pct`.

---

## 7. Estructura del Directorio y Archivos

```text
ml_operacional_entrega3/
└── sensitivity/                                ← Nueva carpeta principal del análisis
    ├── run_sensitivity.py                      ← Script maestro que ejecuta los 4 escenarios
    │                                              secuencialmente y genera el reporte consolidado
    │
    ├── escenario_1_umbrales_plos.py            ← Variación del umbral PLOS (7, 14, 21, 27)
    ├── escenario_2_ablation_features.py        ← Ablation study (sin Charlson, sin cap_, solo demográfico, sin clasificador)
    ├── escenario_3_punto_operacion.py          ← Curva PR + 3 políticas clínicas de umbrales
    ├── escenario_4_hiperparametros.py          ← Validación de estabilidad de hiperparámetros vecinos
    │
    └── results/                                ← Subcarpeta de resultados por escenario
        ├── escenario_1_resultados.csv
        ├── escenario_1_resultados_por_tramo.csv
        ├── escenario_2_resultados.csv
        ├── escenario_3_curva_pr.csv
        ├── escenario_3_puntos_operacion.csv
        ├── escenario_4_resultados.csv
        └── reporte_sensibilidad_consolidado.md  ← Informe final académico consolidado
```

---

## 8. Consideraciones para la Implementación

### 8.1 Ninguno de los escenarios requiere tuning

Todos los escenarios (incluyendo el 4) reutilizan los `best_params_clf` y `best_params_reg` ya guardados en `ml_operacional_entrega3/XGB/` o aplican pequeñas perturbaciones fijas y preestablecidas sobre ellos. Esto reduce el tiempo de ejecución a unos pocos minutos y garantiza el aislamiento de variables del experimento.

### 8.2 El holdout NUNCA cambia

Los 2,391 pacientes del holdout (20%) son siempre los mismos. Esto garantiza la comparabilidad directa y estricta entre todos los experimentos.

### 8.3 Reutilización del código existente

Cada escenario debe importar las funciones de `ml_operacional_entrega3/utils/pipeline_operacional.py` y `metricas_operacionales.py`. No se duplica código.

### 8.4 Random State

`RANDOM_STATE = 42` en todo momento para garantizar reproducibilidad total.

### 8.5 Criterio cuantitativo de robustez (Veredicto de Desviación)

Para determinar si la solución es robusta, se utiliza la variación porcentual del MAE respecto a la línea base:

$$\Delta\text{MAE}\% = \frac{\text{MAE}_{\text{escenario}} - \text{MAE}_{\text{base}}}{\text{MAE}_{\text{base}}} \times 100$$

*   **$|\Delta\text{MAE}\%| < 5\%$:** El modelo es **robusto** frente a este cambio (variabilidad aceptable).
*   **$5\% \le |\Delta\text{MAE}\%| < 15\%$:** Sensibilidad **moderada**. Documentar como limitación menor.
*   **$|\Delta\text{MAE}\%| \ge 15\%$:** Sensibilidad **alta**. El modelo no es robusto frente a este cambio (se documenta como limitación crítica).

---

## 9. Referencias bibliográficas (APA 7)

*   **Bergstra, J., & Bengio, Y.** (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(2), 281–305.
*   **Bottle, A., & Aylin, P.** (2014). Predicting the false alarm rate in emergency admissions. *Journal of Clinical Epidemiology*, 68(2), 229–237. https://doi.org/10.1016/j.jclinepi.2014.09.014
*   **Chrusciel, J., et al.** (2022). Machine-learning prediction for hospital length of stay using a French medico-administrative database. *Journal of Evaluation in Clinical Practice*, 28(5), 828–838.
*   **Dettori, J. R., et al.** (2024). Patient factors associated with prolonged length of stay after traumatic brain injury. *Journal of Neurotrauma*, PMC11107954.
*   **Goldstein, B. A., et al.** (2022). Predicting in-hospital length of stay: A two-stage modeling approach to account for highly skewed data. *BMC Medical Informatics and Decision Making*, 22(1), 114. https://doi.org/10.1186/s12911-022-01855-0
*   **Lee, H., et al.** (2024). Hospital length of stay prediction for planned admissions using OMOP CDM. *Journal of Medical Internet Research*, 26, e59260. https://doi.org/10.2196/59260
*   **Mahajan, A., et al.** (2023). Patient outcome predictions improve operations at a large hospital network. *arXiv preprint*, arXiv:2305.15629.
*   **Morinaga, T., et al.** (2024). Explainable predictions of a machine learning model to forecast the postoperative length of stay after gastrectomy using XGBoost and SHAP. *BMC Medical Informatics and Decision Making* (aceptado 2024).
*   **Probst, P., et al.** (2019). Hyperparameters and tuning strategies for random forest. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 9(3), e1301. https://doi.org/10.1002/widm.1301
*   **Saito, T., & Rehmsmeier, M.** (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
*   **Zeleke, A. J., et al.** (2023). Machine learning-based prediction of hospital prolonged length of stay admission at emergency department. *Frontiers in Artificial Intelligence*, 6, 1179226. https://doi.org/10.3389/frai.2023.1179226
