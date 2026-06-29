# Análisis Clínico: Índices de Comorbilidad v3

Este reporte presenta los resultados del cálculo de los índices de comorbilidad de Charlson y Elixhauser para los 11.951 pacientes del conjunto de datos. Estas variables conforman los escenarios B y C utilizados en el entrenamiento de los modelos.

---

## 1. Índice de Comorbilidad de Charlson (Escenario B)

El índice de Charlson clasifica las enfermedades crónicas asignándoles un peso (1 a 6) según su impacto en el riesgo de mortalidad.

### Distribución de Puntajes
*   **Total de pacientes analizados:** 11,951
*   **Pacientes sin comorbilidades graves (Score 0):** 8,219 (68.7%)
*   **Pacientes con comorbilidades (Score ≥ 1):** 3,732 (31.3%)
*   **Rango de puntajes:** 0 a 11 puntos.

| Puntaje Charlson | Cantidad de Pacientes | Interpretación Clínica |
| :--- | :--- | :--- |
| **0** | 8,219 | Sin condiciones crónicas de alto riesgo. |
| **1 - 2** | 2,339 | Riesgo moderado (ej. Diabetes, EPOC, IAM). |
| **3 - 5** | 779 | Riesgo alto (múltiples condiciones combinadas). |
| **6 - 11** | 614 | Riesgo crítico (ej. Tumores metastásicos, SIDA, daño hepático severo). |

**Conclusión Charlson:** La distribución es altamente asimétrica (*skewed*). Casi el 70% de la población tiene un score de 0. Esto significa que agregar el Charlson ayudará al modelo de Machine Learning a aislar e identificar rápidamente a ese ~30% de pacientes que, por su alta carga de enfermedad, son candidatos naturales a estancias prolongadas (PLOS).

---

## 2. Índice de Elixhauser (Escenario C)

Elixhauser es un sistema más granular que Charlson y considera 31 categorías. Para calcular el puntaje continuo se utilizó el sistema de ponderación de **van Walraven (2009)**.

### Top 10 Comorbilidades más Frecuentes
De las 31 categorías evaluadas, estas son las condiciones crónicas más prevalentes en el hospital:

1.  **Obesidad (`elix_obes`):** 2,964 pacientes
2.  **Hipertensión no complicada (`elix_hypunc`):** 2,484 pacientes
3.  **Pérdida de peso / Desnutrición (`elix_wloss`):** 1,689 pacientes
4.  **Arritmias cardíacas (`elix_carit`):** 1,430 pacientes
5.  **Diabetes no complicada (`elix_diabunc`):** 1,276 pacientes
6.  **Enfermedad Pulmonar Crónica (`elix_cpd`):** 1,101 pacientes
7.  **Tumores Sólidos sin metástasis (`elix_solidtum`):** 997 pacientes
8.  **Falla Renal (`elix_rf`):** 973 pacientes
9.  **Desequilibrio de fluidos/electrolitos (`elix_fed`):** 892 pacientes
10. **Hipertensión complicada (`elix_hypc`):** 872 pacientes

### Distribución del Score de Elixhauser (van Walraven)
*   **Pacientes con Score 0:** 7,352 (61.5%)
*   **Rango de puntajes:** 0 a 46 puntos.

**Hallazgo clínico:**
Se observa una frecuencia elevada de pérdida de peso (`elix_wloss`: 1.689 casos) y desequilibrio de fluidos o electrolitos (`elix_fed`: 892 casos). Ambas condiciones pueden representar fragilidad clínica y aportar información predictiva sobre la duración de la estancia. Su contribución efectiva debe comprobarse mediante el análisis de importancia de variables del modelo.

---

## 3. Implicaciones para la Fase 3 (Modelamiento)

1. **Reducción del ruido dimensional:** En la versión original (v2), los diagnósticos ICD se representaban mediante cientos de columnas dispersas. El escenario C resume las comorbilidades en 31 categorías clínicas densas relacionadas, entre otros ámbitos, con síndrome metabólico y fragilidad. Esta reducción puede disminuir la dimensionalidad, aunque también implica pérdida de información diagnóstica específica.
2. **Representación de valores extremos de LOS:** Los puntajes altos de comorbilidad ofrecen al modelo una medida agregada de carga clínica que puede contribuir a distinguir pacientes con mayor riesgo de estancias prolongadas. Esta hipótesis debe evaluarse mediante las métricas obtenidas en validación y holdout.
