# Análisis Clínico: Índices de Comorbilidad v3

Este reporte presenta los hallazgos tras calcular los índices de severidad clínica (Charlson y Elixhauser) para los 11,951 pacientes del dataset. Estos datos forman ahora los **Escenarios B y C** que utilizaremos para entrenar modelos avanzados.

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

Elixhauser es un sistema más granular que Charlson, midiendo 31 categorías distintas. Usamos el sistema de pesos de **van Walraven (2009)** para calcular el score continuo.

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

**Hallazgo Clínico Crítico:**
El alto volumen de pacientes con **Pérdida de Peso (`elix_wloss`: 1,689 casos)** y **Desequilibrio de Electrolitos (`elix_fed`: 892 casos)** es crucial. En la literatura médica sobre Length of Stay (LOS), la desnutrición y la deshidratación son dos de los **mejores predictores de hospitalizaciones prolongadas**, ya que indican fragilidad sistémica del paciente al momento del ingreso. XGBoost seguramente utilizará estas variables con mucho peso en el Escenario C.

---

## 3. Implicaciones para la Fase 3 (Modelamiento)

1.  **Reducción de Ruido Dimensional:** En la versión original (v2), teníamos cientos de columnas de diagnósticos ICD dispersos que causaban esparcidad y sobreajuste. En el Escenario C, concentramos toda esa información médica en **solo 31 columnas clínicas densas** que representan a la perfección el Síndrome Metabólico (obesidad, hipertensión, diabetes) y la Fragilidad (pérdida de peso, falla renal) de la población.
2.  **Manejo de Valores Extremos (Outliers en LOS):** Ya que la mayoría de los pacientes tienen un Score de 0, el algoritmo XGBoost podrá usar los valores altos de estos índices para justificar e identificar anticipadamente por qué un paciente que a simple vista parece normal, terminará internado por periodos extremadamente largos (≥ 14 o 27 días).
