# Resumen Consolidado del Proyecto: Predicción de Estancia Hospitalaria (LOS)
**Documento Metodológico y Técnico Completo para la Entrega Final (Grupo 16)**

Este documento resume de manera exhaustiva el contexto, la metodología, el procesamiento de datos, los modelos de Machine Learning (avanzados y base), el análisis estadístico de interacciones y la solución de software interactiva de nuestro proyecto de predicción de **Length of Stay (LOS)** hospitalario.

---

## 1. Contexto y Objetivos del Proyecto

El **Length of Stay (LOS)** (días reales de hospitalización del paciente) es uno de los indicadores de gestión de recursos sanitarios y planificación clínica más importantes en cualquier institución médica. Predecir con precisión el LOS permite:
*   Optimizar la asignación de camas y reducir la saturación en salas y urgencias.
*   Planificar de forma anticipada el alta médica y los cuidados posteriores.
*   Identificar de forma temprana a pacientes críticos con riesgo de estancia prolongada.

**Objetivo General:** Desarrollar un sistema de predicción del LOS hospitalario basado en aprendizaje supervisado a partir de información disponible en el momento del ingreso clínico, comparando un modelo base lineal riguroso con modelos avanzados no lineales (Random Forest y XGBoost), e integrando la solución en una aplicación web interactiva.

---

## 2. El Dataset y Procesamiento de Datos

El dataset original consta de **11,951 registros de pacientes únicos** con variables administrativas y clínicas estructuradas.

### A. Agrupación Jerárquica de Códigos Clínicos (ICD-10-CM / ICD-10-PCS)
Los diagnósticos (ICD-10-CM) y procedimientos (ICD-10-PCS) presentan alta dimensionalidad y dispersión (*sparsity*). Para evitar el sobreajuste, se implementó una regla de soporte mínimo jerárquico:
1.  **Código Completo:** Si aparece en al menos 20 pacientes únicos, se conserva tal cual.
2.  **Agrupación a 3 dígitos (Categoría):** Si tiene menos de 20 casos, se trunca a sus primeros 3 caracteres (ej. `E6601` $\rightarrow$ `E66`).
3.  **Agrupación a Capítulo/Sección:** Si la categoría de 3 dígitos tampoco tiene 20 casos, se agrupa en su capítulo diagnóstico (letra general) o sección de procedimiento.
4.  **Tratamiento de repeticiones:** Si un paciente tiene el mismo diagnóstico registrado varias veces, cuenta como presencia binaria ($1$), pero se capturan variables agregadas de carga clínica total.

### B. Tratamiento del Código Administrativo `UUUUUU`
El código `UUUUUU` no corresponde a una patología, sino a un registro administrativo de **Ingreso por Urgencia**. Metodológicamente se excluyó del catálogo de diagnósticos clínicos y se convirtió en una variable predictiva estructurada binaria llamada `es_urgencia`.

### C. Escenarios de Modelamiento Comparados
Para evaluar la mejor representación clínica de comorbilidades, el dataset se estructuró en tres escenarios:
*   **Escenario A (ICD/PCS Agrupados):** La representación más granular con todos los grupos ICD y PCS que pasaron el soporte mínimo ($1,650$ variables).
*   **Escenario B (ICD/PCS + Charlson):** Agrega el Índice de Comorbilidad de Charlson (`charlson_index`), calculado dinámicamente según el mapeo de *Quan et al. (2005)*. **Este escenario demostró ser el de mejor desempeño y generalización.**
*   **Escenario C (Base + Procedimientos + Elixhauser):** Reemplaza las columnas de diagnósticos dispersas por 31 categorías de comorbilidad de Elixhauser y el Score ponderado de *van Walraven*.

---

## 3. Arquitectura y Entrenamiento de Modelos Avanzados

Se entrenaron dos arquitecturas basadas en árboles de decisión para capturar patrones no lineales e interacciones complejas de forma implícita:
1.  **Random Forest (Bosque Aleatorio):** Entrenado promediando múltiples árboles de decisión paralelos independientes para reducir varianza.
2.  **XGBoost (Extreme Gradient Boosting):** Algoritmo secuencial que construye árboles iterativamente para corregir los errores residuales del paso anterior, aplicando regularización explícita L1 y L2 para evitar el sobreajuste.

### El problema de Overfitting V1 y la Solución V2 (Regularizada)
En la fase preliminar (**V1**), los modelos de árboles sufrieron de sobreajuste masivo al aprender de memoria el train set (MAE en entrenamiento de $0.68$ días vs. $2.95$ en validación cruzada para XGBoost).
Para obtener los **modelos finales definitivos (V2)** se aplicaron tres técnicas fundamentales:
*   **Partición Holdout:** Separación estricta de un 20% del dataset desde el inicio como conjunto de prueba intocable.
*   **Transformación Logarítmica:** Modelamiento sobre $\log(1 + \text{LOS})$ para comprimir la cola larga de estancias prolongadas y estabilizar la varianza, revirtiendo con $\exp(x) - 1$ para reportar en días reales.
*   **Regularización Estricta:** Incorporación de hiperparámetros de penalización de complejidad en XGBoost (`reg_alpha`, `reg_lambda`, `min_child_weight` y limitación de `max_depth`).

---

## 4. Modelamiento del Modelo Base Lineal Regularizado
*Archivos y scripts de soporte:* [analisis_critico_interacciones.md](file:///c:/Users/tomas/Downloads/E3_Capstone/Capstone-Grupo-16--main/Modelo_Base_Ultima%20entrega/analisis_critico_interacciones.md) y [entrenar_lr_urgencias_no_urgencias.py](file:///c:/Users/tomas/Downloads/E3_Capstone/Capstone-Grupo-16--main/Modelo_Base_Ultima%20entrega/entrenar_lr_urgencias_no_urgencias.py).

Para responder con máxima rigurosidad a las observaciones sobre la comparabilidad lineal de los modelos, se diseñó un modelo base estructurado bajo las siguientes especificaciones metodológicas:

### A. Separación Metodológica de Datos (Urgencias vs. No Urgencias)
El dataset se dividió en dos subconjuntos independientes: **Urgencias** (ingresos por emergencia) y **No Urgencias** (ingresos electivos o programados).
*   **Justificación Matemática:** Entrenar dos modelos por separado equivale a incorporar de forma matemática interacciones completas entre la variable `es_urgencia` y las $1,650$ variables clínicas ($X_j \times \text{es\_urgencia}$).
*   **Justificación Clínica:** La naturaleza de la estancia varía drásticamente según el ingreso. En ingresos planificados, el LOS suele estar acotado por el procedimiento quirúrgico a realizar. En urgencias, depende fuertemente de la agudeza diagnóstica de ingreso.

### B. Selección de Interacciones mediante Pruebas de Hipótesis ($p$-values)
Se evaluaron tres términos de interacción continua sobre el dataset real mediante regresión en `statsmodels`:
1.  `charlson_index` $\times$ `n_diag_total` (Gravedad Crónica $\times$ Total de Diagnósticos).
2.  `charlson_index` $\times$ `n_procedimientos` (Gravedad Crónica $\times$ Cantidad de Procedimientos).
3.  `n_procedimientos` $\times$ `n_diag_total` (Cantidad de Procedimientos $\times$ Total de Diagnósticos).

**Resultados del análisis estadístico real sobre los datos:**
*   **Grupo Urgencias:** Las 3 interacciones son altamente significativas (p-values $< 10^{-7}$). **Se integran las 3.**
*   **Grupo No Urgencias:** Las interacciones 1 y 2 son significativas ($p < 0.05$). Sin embargo, la interacción `n_procedimientos` $\times$ `n_diag_total` arrojó un p-value de **$0.5545$** (55% de probabilidad de ser ruido fortuito). **Se descarta esta interacción en No Urgencias** para mantener la parsimonia y evitar coeficientes espurios.

### C. Solución a la Inestabilidad del Modelo Lineal Tradicional (Ridge Regression)
Cuando se entrena una regresión lineal clásica por Mínimos Cuadrados Ordinarios (OLS) en escala logarítmica sobre $1,650$ variables con alta colinealidad, el optimizador asigna coeficientes excesivamente grandes y opuestos (ej. $+100.0$ a diabetes y $-98.0$ a hipertensión) que se compensan en el entrenamiento. 

Al predecir un paciente nuevo que presenta una sola enfermedad, la compensación no ocurre y el modelo produce una estancia irreal:
$$\text{Días Predichos} = e^{100.0} - 1 \approx \text{infinito}$$
Esto causó que el error promedio (MAE) en el conjunto de test original explotara.

**Solución Implementada:**
Se implementó la **Regresión Ridge (Regularización L2)**. Esta técnica restringe la magnitud de los coeficientes, forzándolos a permanecer en valores clínicamente razonables (como $+0.15$ y $+0.12$).
*   **Urgencias:** Ridge con $\alpha = 100.0$ $\rightarrow$ **MAE final de 5.75 días**.
*   **No Urgencias:** Ridge con $\alpha = 50.0$ $\rightarrow$ **MAE final de 2.08 días**.

Los modelos estables y sus columnas correspondientes quedaron guardados en archivos binarios pickle (`lr_base_Urgencias.pkl` y `lr_base_No_Urgencias.pkl`).

---

## 5. Resultados Comparativos Finales (Set de Prueba Holdout)

Los modelos finales de Machine Learning regularizados y el modelo base Ridge fueron evaluados en pacientes nuevos del set de prueba holdout. Los resultados en días reales son:

### A. Métricas de Error Continuo (Días Reales)

| Modelo | Escenario | MAE (días) | RMSE (días) | MedAE (días) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Final** | B (Con Charlson) | **3.057** | **8.290** | **0.902** |
| **Random Forest Final** | B (Con Charlson) | 3.268 | 8.959 | 0.972 |
| **Modelo Base Lineal (Ridge)** | B (Con Charlson) | 5.75 (Urg) / 2.08 (No Urg) | 11.64 / 7.26 | 2.39 / 0.60 |

### B. Capacidad para Detectar Pacientes Críticos (PLOS $\ge 27$ días)
Para evaluar la utilidad clínica en la gestión de camas, convertimos la predicción continua en una alerta binaria de Estancia Prolongada (PLOS $\ge 27$ días):

| Métrica | Regresión Ridge (Urgencias) | Random Forest Final | XGBoost Final |
| :--- | :---: | :---: | :---: |
| **Precision PLOS** | 71.79% | **80.85%** | 78.67% |
| **Recall PLOS** | 36.84% | 31.15% | **48.36%** |
| **F1-Score PLOS** | 48.69% | 44.97% | **59.90%** |

*   **XGBoost Final** ofrece el mejor balance clínico general (F1-score de 59.90%), logrando identificar casi al 50% de los pacientes de larga estancia con una precisión de alerta del 78.67%.
*   **Limitación conocida:** Todos los modelos presentan regresión a la media (subestiman las estancias extremadamente largas). Esto es una limitación esperable de los modelos de ingreso, dado que no capturan la evolución clínica intrahospitalaria diaria.

---

## 6. Solución de Software: Aplicación Web Interactiva

Para desplegar esta tecnología e integrarla en la toma de decisiones clínicas, se diseñó e implementó una aplicación web interactiva en la carpeta `Web/` con las siguientes características:

*   **Backend (Flask - `app.py`):**
    *   Carga dinámicamente los modelos entrenados (`lr_base_Urgencias.pkl` y `lr_base_No_Urgencias.pkl`).
    *   Calcula en tiempo real el Índice de Comorbilidad de Charlson basándose en los diagnósticos suministrados usando la librería `comorbidipy` (versión 0.8.0).
    *   Construye las variables de interacciones de manera dinámica antes de evaluar la regresión regularizada Ridge.
*   **Frontend (HTML5/Vanilla CSS/Tailwind CSS):**
    *   Diseño estético premium con interfaz responsiva y componentes interactivos para la predicción de estancia.
    *   Selector de modelo que permite comparar en tiempo real las estimaciones del **Modelo Base Lineal (Ridge)** con los modelos avanzados de Machine Learning (XGBoost/Random Forest).
    *   Visualización de resultados con alertas semánticas de riesgo y desglose de factores influyentes.

---

## 7. Justificación Metodológica de las Decisiones de Diseño

Para sustentar estadísticamente las decisiones tomadas frente al comité evaluador:

1.  **Justificación de las Interacciones:** Se evitó utilizar un modelo lineal con aditividad pura. Se incorporaron las interacciones de segundo orden más significativas según los datos reales. Se descartó la interacción de procedimientos con diagnósticos en pacientes no urgentes debido a un p-value de 0.55, lo que prueba su nulo aporte de valor real en admisiones programadas y previene el sobreajuste.
2.  **Justificación de la Separación de Datos:** La división en Urgencias y No Urgencias no es arbitraria; es matemáticamente equivalente a incluir interacciones completas de todas las variables con el tipo de ingreso, lo cual es fundamental debido al comportamiento clínico dispar entre casos agudos y planificados.
3.  **Ventajas de Ridge sobre OLS:** La regresión lineal clásica por Mínimos Cuadrados Ordinarios (OLS) tiende a sobreajustar en presencia de multicolinealidad, lo que produce coeficientes inestables de gran magnitud. Al aplicar la exponencial inversa (expm1), los errores en el espacio logarítmico se amplifican exponencialmente. La regularización de Ridge (L2) restringe la magnitud de los coeficientes, asegurando predicciones realistas y estables frente a nuevos registros de pacientes.
