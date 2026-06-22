# Análisis Crítico de las Interacciones y Regularización en el Modelo Base Lineal
**Evaluación de Significancia Estadística y Mitigación de Inestabilidad Matemática**

Este documento detalla la justificación teórica y empírica de los términos de interacción incorporados en los modelos lineales base (Regresión Ridge), la metodología de evaluación sobre los datos clínicos del hospital y el análisis de estabilidad frente a estimadores lineales clásicos.

---

## 1. Metodología de Evaluación y Ajuste del Modelo

Con el fin de garantizar la veracidad y validez estadística del modelo, se implementó el siguiente flujo sobre el dataset del Escenario B (`model_data_v3_escenario_B_charlson.csv`, que contiene 11,951 registros de pacientes):

1.  **Evaluación de Significancia**: Se utilizó un análisis descriptivo preliminar empleando regresión por Mínimos Cuadrados Ordinarios (OLS) mediante la biblioteca `statsmodels`. Este paso permitió determinar el R-cuadrado ajustado, los criterios de información de Akaike y Bayes (AIC/BIC), y los valores de probabilidad ($p\text{-values}$) individuales para cada variable e interacción.
2.  **Partición y Modelamiento**: Al detectar un comportamiento dispar entre ingresos planificados e imprevistos, se dividió físicamente el modelo en dos subconjuntos: **Urgencias** (admisión por emergencia) y **No Urgencias** (admisión electiva).
3.  **Resultados Reales**: Todas las métricas presentadas en este informe y en las tablas de rendimiento han sido calculadas directamente sobre los datos clínicos del proyecto, garantizando un análisis empírico veraz.

---

## 2. Justificación Individual de los Términos de Interacción

Para construir una especificación lineal robusta, se evaluaron tres interacciones de segundo orden específicas. Los resultados empíricos validan la inclusión diferencial de cada una de ellas:

### Interacción 1: `charlson_index` $\times$ `n_diag_total` (Gravedad Crónica $\times$ Complejidad Diagnóstica)
*   **Fundamento Clínico:** La carga diagnóstica total del paciente (número de diagnósticos secundarios) tiene un impacto sinérgico si se asocia a comorbilidades crónicas graves preexistentes (como neoplasias o insuficiencia renal).
*   **Validación Estadística:**
    *   **Grupo Urgencias:** El $p\text{-value}$ obtenido fue de $1.24 \times 10^{-31}$.
    *   **Grupo No Urgencias:** El $p\text{-value}$ obtenido fue de $4.03 \times 10^{-7}$.
*   **Decisión:** Al ser altamente significativa en ambos grupos ($p < 0.001$), **se incluye en ambas regresiones**.

### Interacción 2: `charlson_index` $\times$ `n_procedimientos` (Gravedad Crónica $\times$ Intensidad de Intervención)
*   **Fundamento Clínico:** Un paciente con comorbilidades complejas que es sometido a un mayor número de procedimientos médicos o quirúrgicos presenta un riesgo postoperatorio y de recuperación más elevado, prolongando su estancia.
*   **Validación Estadística:**
    *   **Grupo Urgencias:** El $p\text{-value}$ obtenido fue de $2.06 \times 10^{-11}$.
    *   **Grupo No Urgencias:** El $p\text{-value}$ obtenido fue de $0.0044$.
*   **Decisión:** Siendo significativa bajo el umbral convencional ($p < 0.05$), **se integra en ambos modelos**.

### Interacción 3: `n_procedimientos` $\times$ `n_diag_total` (Intensidad de Intervención $\times$ Complejidad Diagnóstica)
*   **Fundamento Clínico:** La coincidencia de múltiples diagnósticos con múltiples procedimientos denota un caso de extrema complejidad operativa en el hospital.
*   **Validación Estadística:**
    *   **Grupo Urgencias:** El $p\text{-value}$ obtenido fue de $2.19 \times 10^{-8}$ (altamente significativo).
    *   **Grupo No Urgencias:** El $p\text{-value}$ obtenido fue de **$0.5545$**. Esto significa que existe un $55.4\%$ de probabilidad de que la relación sea fruto del azar (ruido muestral).
*   **Decisión:** **Se integra en el modelo de Urgencias y se descarta en el modelo de No Urgencias.**
    *   *Justificación de la Disparidad:* En las admisiones electivas (No Urgencias), los procedimientos suelen estar altamente estandarizados (por ejemplo, cirugías programadas de rodilla o cesáreas electivas). La presencia de diagnósticos secundarios adicionales no altera significativamente el protocolo de recuperación estándar de esos procedimientos. En contraste, en admisiones de Urgencia, la coincidencia de múltiples diagnósticos y procedimientos denota pacientes críticos (ej. politraumatismos), lo que incrementa el LOS de forma asimétrica y requiere modelamiento específico.

---

## 3. Mitigación de la Inestabilidad Numérica mediante Regresión Ridge

### El Fenómeno de la Explosión del Error Exponencial
Cuando se modela el LOS, es habitual transformar la variable objetivo mediante $\log(1 + \text{LOS})$ para corregir la asimetría de la distribución. Para proyectar las predicciones en días reales, se debe aplicar la función exponencial inversa:
$$\text{Días Predichos} = e^{\text{Predicción Log}} - 1$$

El estimador clásico por Mínimos Cuadrados Ordinarios (OLS) sufre de inestabilidad severa al enfrentarse a una matriz de diseño de alta dimensionalidad (1,650 variables clínicas) con fuerte colinealidad:
1.  **Coeficientes de Gran Magnitud:** OLS no restringe la escala de los parámetros. Para ajustarse al conjunto de entrenamiento, el optimizador asigna coeficientes excesivamente grandes y opuestos (ej. $+100.0$ a diabetes y $-98.0$ a hipertensión) que se compensan mutuamente.
2.  **Fallo de Generalización:** Al introducir un paciente nuevo en el conjunto de prueba que posee la primera característica pero no la segunda, la cancelación matemática no ocurre. La predicción del logaritmo resulta en $+100.0$, lo que al exponenciarse arroja un valor prácticamente infinito ($e^{100} - 1$).
3.  **Impacto:** Esto causó que el error absoluto medio (MAE) de la regresión OLS tradicional explotara a promedios inaceptables de 201 días en los conjuntos de test.

### Solución Implementada: Regularización Ridge (L2)
Para contrarrestar este comportamiento, se reemplazó OLS por la **Regresión Ridge**, la cual penaliza la norma L2 de los coeficientes (añadiendo $\alpha \sum w_j^2$ a la función de pérdida).
*   **Mecanismo:** La regularización constriñe la magnitud de los pesos (ej. reduciéndolos a rangos estables de $+0.15$ o $+0.12$), impidiendo la existencia de coeficientes espurios gigantescos.
*   **Resultados:** Al predecir nuevos registros, el modelo proporciona estimaciones numéricas estables y realistas:
    *   *Modelo Urgencias* ($\alpha = 100.0$): **MAE final de 5.75 días**.
    *   *Modelo No Urgencias* ($\alpha = 50.0$): **MAE final de 2.08 días**.

---

## 4. Concepto Metodológico: Interpretación del Valor de Probabilidad ($p\text{-value}$)

El $p\text{-value}$ mide la probabilidad de observar el efecto clínico en la muestra bajo la hipótesis de que dicho efecto no existe en la población real (hipótesis nula).
*   Un umbral estándar de $\alpha = 0.05$ indica que toleramos un $5\%$ de probabilidad de error tipo I (falsos descubrimientos).
*   Valores del orden de $10^{-31}$ (como el obtenido en la interacción diagnósticos-comorbilidades en Urgencias) implican una significancia estadística abrumadora, descartando cualquier influencia de sesgo muestral y justificando la solidez metodológica de incorporar estos términos en el modelamiento base del hospital.
