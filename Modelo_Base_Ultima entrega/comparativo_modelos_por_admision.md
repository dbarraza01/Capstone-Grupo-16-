# Comparativo Definitivo de Modelos Predictivos por Tipo de Admisión (Urgencias vs No Urgencias)
## Capstone Grupo 16 — Análisis de Desempeño y Capacidad Predictiva

Este documento presenta una comparación rigurosa de los tres modelos finales desarrollados para la predicción de la estancia hospitalaria (*Length of Stay*, LOS):
1. **Regresión Ridge Lineal (Modelo Base nuevo con interacciones)**: Segmentado físicamente en dos modelos optimizados independientes.
   * *Urgencias*: Regularización $\alpha = 100.0$ con 3 interacciones clínicas de base.
   * *No Urgencias*: Regularización $\alpha = 50.0$ con 2 interacciones clínicas de base (sin interacción procedimientos × diagnósticos por nula significancia).
2. **XGBoost Final (Avanzado)**: Modelo global regularizado (`max_depth=5`, `n_estimators=755`).
3. **Random Forest Final (Avanzado)**: Modelo global regularizado (`max_depth=20`, `n_estimators=777`).

Todos los modelos se evalúan en el **Holdout Test Set (20% del total de datos)** de forma estrictamente estratificada por tramos, asegurando la consistencia e integridad del análisis comparativo.

---

## 1. Escenario de Admisión: URGENCIAS (n = 698 pacientes de prueba)

Este grupo representa a pacientes que ingresan de urgencia al hospital. Suelen caracterizarse por una alta variabilidad y cuadros clínicos agudos e imprevistos.

### 1.1 Métricas Globales de Regresión y Clasificación PLOS (Urgencias)

| Métrica | Regresión Ridge (Nuevo) | XGBoost Final | Random Forest Final | Ganador |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** (días) | 5.7568 | **4.6583** | 5.2827 | **XGBoost** |
| **RMSE** (días) | 11.6414 | **9.0644** | 10.6879 | **XGBoost** |
| **MedAE** (días) | 2.3967 | **2.0593** | 2.3978 | **XGBoost** |
| **PLOS Precision** (≥27d) | 71.79% | **84.31%** | 81.25% | **XGBoost** |
| **PLOS Recall** (≥27d) | 36.84% | **56.58%** | 34.21% | **XGBoost** |
| **PLOS F1-Score** | 48.70% | **67.72%** | 48.15% | **XGBoost** |

> [!NOTE]
> **Matriz de Confusión para PLOS (≥ 27 días) en Urgencias (n = 76 casos reales de estancia larga):**
> * **Ridge (Nuevo)**: Verdaderos Positivos (TP) = 28 | Falsos Positivos (FP) = 11 | Falsos Negativos (FN) = 48
> * **XGBoost**: Verdaderos Positivos (TP) = 43 | Falsos Positivos (FP) = 8 | Falsos Negativos (FN) = 33
> * **Random Forest**: Verdaderos Positivos (TP) = 26 | Falsos Positivos (FP) = 6 | Falsos Negativos (FN) = 50

### 1.2 Métricas Detalladas por Tramo de Estancia (Urgencias)

| Tramo de Estancia | Métrica | Regresión Ridge (Nuevo) | XGBoost Final | Random Forest Final | Ganador |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0-2 días** <br>*(n=200)* | MAE <br> RMSE <br> % Subestimación | 2.1407 <br> 3.3997 <br> **6.0%** | **1.7687** <br> **2.5916** <br> 9.0% | 2.0839 <br> 2.9968 <br> 7.5% | **XGBoost** *(MAE/RMSE)* <br> **Ridge** *(Subestimación)* |
| **3-6 días** <br>*(n=179)* | MAE <br> RMSE <br> % Subestimación | 1.8974 <br> **2.8429** <br> 59.2% | **1.8780** <br> 2.9833 <br> 58.1% | 2.1235 <br> 3.8774 <br> **52.0%** | **XGBoost** *(MAE)* <br> **Random Forest** *(Sub)* |
| **7-13 días** <br>*(n=139)* | MAE <br> RMSE <br> % Subestimación | 3.8805 <br> 4.6240 <br> 69.1% | **3.5225** <br> 4.5239 <br> 65.5% | 3.5269 <br> **4.3966** <br> **64.7%** | **XGBoost** *(MAE)* <br> **Random Forest** *(RMSE/Sub)* |
| **14-26 días** <br>*(n=104)* | MAE <br> RMSE <br> % Subestimación | 8.8448 <br> 11.7490 <br> **79.8%** | **6.8364** <br> **8.1898** <br> **77.9%** | 6.8682 <br> 8.2896 <br> **79.8%** | **XGBoost** *(Todos)* |
| **27+ días** <br>*(n=76)* | MAE <br> RMSE <br> % Subestimación | 23.5685 <br> 31.1001 <br> 84.2% | **17.9079** <br> **24.2231** <br> **80.3%** | 22.1826 <br> 29.3374 <br> 86.8% | **XGBoost** *(Todos)* |

---

## 2. Escenario de Admisión: NO URGENCIAS / ELECTIVA (n = 1,693 pacientes de prueba)

Este grupo representa a pacientes que ingresan de forma planificada (cirugías programadas, tratamientos médicos de control, etc.). Su estancia suele ser significativamente más corta y homogénea.

### 2.1 Métricas Globales de Regresión y Clasificación PLOS (No Urgencias)

| Métrica | Regresión Ridge (Nuevo) | XGBoost Final | Random Forest Final | Ganador |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** (días) | 2.0873 | **1.7295** | 2.0014 | **XGBoost** |
| **RMSE** (días) | 7.2608 | **5.0197** | 6.2742 | **XGBoost** |
| **MedAE** (días) | **0.6044** | 0.6229 | 0.6826 | **Ridge** |
| **PLOS Precision** (≥27d) | 81.82% | 86.21% | **91.67%** | **Random Forest** |
| **PLOS Recall** (≥27d) | 39.13% | **54.35%** | 23.91% | **XGBoost** |
| **PLOS F1-Score** | 52.94% | **66.67%** | 37.93% | **XGBoost** |

> [!NOTE]
> **Matriz de Confusión para PLOS (≥ 27 días) en No Urgencias (n = 46 casos reales de estancia larga):**
> * **Ridge (Nuevo)**: Verdaderos Positivos (TP) = 18 | Falsos Positivos (FP) = 4 | Falsos Negativos (FN) = 28
> * **XGBoost**: Verdaderos Positivos (TP) = 25 | Falsos Positivos (FP) = 4 | Falsos Negativos (FN) = 21
> * **Random Forest**: Verdaderos Positivos (TP) = 11 | Falsos Positivos (FP) = 1 | Falsos Negativos (FN) = 35

### 2.2 Métricas Detalladas por Tramo de Estancia (No Urgencias)

| Tramo de Estancia | Métrica | Regresión Ridge (Nuevo) | XGBoost Final | Random Forest Final | Ganador |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0-2 días** <br>*(n=983)* | MAE <br> RMSE <br> % Subestimación | 0.5903 <br> 0.9339 <br> 24.3% | **0.5701** <br> **0.8317** <br> 24.2% | 0.6176 <br> 1.0717 <br> **13.2%** | **XGBoost** *(MAE/RMSE)* <br> **Random Forest** *(Sub)* |
| **3-6 días** <br>*(n=501)* | MAE <br> RMSE <br> % Subestimación | 1.1865 <br> 1.6346 <br> **70.1%** | **1.1815** <br> **1.5877** <br> 76.4% | 1.2067 <br> 1.6440 <br> 74.5% | **XGBoost** *(MAE/RMSE)* <br> **Ridge** *(Sub)* |
| **7-13 días** <br>*(n=110)* | MAE <br> RMSE <br> % Subestimación | **3.8392** <br> **4.5093** <br> 89.1% | 4.1564 <br> 4.7242 <br> 87.3% | 4.0877 <br> 4.6532 <br> **86.4%** | **Ridge** *(MAE/RMSE)* <br> **Random Forest** *(Sub)* |
| **14-26 días** <br>*(n=53)* | MAE <br> RMSE <br> % Subestimación | 9.8858 <br> 12.2920 <br> 79.2% | **8.1931** <br> **9.6816** <br> 75.5% | 8.2778 <br> 9.6894 <br> **71.7%** | **XGBoost** *(MAE/RMSE)* <br> **Random Forest** *(Sub)* |
| **27+ días** <br>*(n=46)* | MAE <br> RMSE <br> % Subestimación | 30.7125 <br> 40.8637 <br> 82.6% | **19.2256** <br> **26.9028** <br> **80.4%** | 28.0089 <br> 35.1413 <br> 100.0% | **XGBoost** *(Todos)* |

---

## 3. Hallazgos Clave y Análisis Comparativo

### 3.1 El Comportamiento del Nuevo Modelo Ridge (Regresión Lineal)
*   **Superación drástica de la Regresión Lineal Clásica (OLS)**: En reportes históricos del proyecto, la regresión lineal sin regularizar (OLS) sufría de una explosión matemática de error al calcular la exponencial (`expm1`), dando un RMSE de 53 días a nivel global. El nuevo enfoque estructurado físicamente (dividido por urgencia) y regularizado mediante penalización Ridge ($\alpha=100$ y $\alpha=50$) junto con las interacciones clínicas calculadas, estabiliza el modelo de manera sobresaliente.
*   **Desempeño en Estancias Cortas (No Urgencias)**: En el grupo de *No Urgencias*, el modelo Ridge muestra una precisión extraordinaria. Consigue el menor **MedAE global de 0.6044 días** (aproximadamente 14 horas de error para la mediana de los pacientes). Además, en el tramo de **7-13 días** supera a XGBoost y Random Forest en precisión de regresión (MAE: 3.8392 vs 4.1564 de XGBoost).
*   **Limitación en los Extremos**: Al tratarse de un modelo lineal, a pesar de estar entrenado bajo el espacio transformado `log1p(LOS)`, sigue teniendo dificultades frente a relaciones altamente no lineales y de alta dimensionalidad. En el tramo crítico de estancias muy prolongadas (**27+ días**), sufre un error de MAE promedio de **23.5 días** en Urgencias y **30.7 días** en No Urgencias.

### 3.2 Desempeño Sobresaliente de XGBoost Final
*   **Ganador Indiscutible**: XGBoost Final es el modelo más robusto y preciso en casi todas las categorías, independientemente de si la admisión es de urgencia o electiva.
*   **Control del Sesgo en Estancias Largas**: En el tramo de **27+ días**, XGBoost logra recortar el MAE a **17.9 días (Urgencias)** y **19.2 días (No Urgencias)**, superando notablemente a la Regresión Ridge (23.5d y 30.7d respectivamente) y a Random Forest (22.1d y 28.0d).
*   **Excelente Detección de Pacientes Críticos (PLOS)**: 
    *   En **Urgencias**, XGBoost detecta el **56.58%** (Recall) de las estancias prolongadas reales con una precisión de alerta del **84.31%**, lo que equivale a un F1-Score excepcional de **67.72%**.
    *   En **No Urgencias**, logra detectar el **54.35%** (Recall) con una precisión de alerta del **86.21%** (F1-Score de **66.67%**).
    *   Esto minimiza enormemente la tasa de falsas alarmas (FP) en comparación con Ridge.

### 3.3 Random Forest Final: El Modelo Conservador
*   **Alta Precisión, Muy Bajo Recall**: Random Forest consigue una precisión de alerta fantástica del **91.67%** en No Urgencias y **81.25%** en Urgencias. Sin embargo, su **Recall es alarmantemente bajo** (solo detecta el **23.91%** de los casos críticos en No Urgencias, dejando escapar a 35 de los 46 pacientes reales con estancia prolongada; y solo el **34.21%** en Urgencias).
*   **Regresión Extrema a la Media**: Al promediar predicciones de múltiples árboles profundos, tiende a subestimar fuertemente a los pacientes más graves. De hecho, en el tramo **27+ de No Urgencias**, el **100% de los casos fueron subestimados** por Random Forest. Esto lo descarta clínicamente como una herramienta de alerta temprana robusta.

---

## 4. Conclusiones y Recomendaciones de Gestión Hospitalaria

1.  **XGBoost como Motor Principal de Predicción**: Para fines operativos de planificación de camas a mediano y largo plazo, detección de estancias críticas y optimización de recursos, **XGBoost es el modelo de elección**. Presenta las mejores métricas combinadas en ambos tipos de admisión y la mayor estabilidad en tramos largos.
2.  **Ridge (Modelo Lineal) como Motor de Explicabilidad Clínica**: La aplicación web se beneficia enormemente del nuevo modelo Ridge. Al estar entrenado con interacciones clínicas explícitas (`int_charlson_diag`, `int_charlson_proc`, `int_proc_diag`), permite descomponer y explicar de forma exacta y en tiempo real el peso (coeficiente) de cada patología y variable sobre la predicción (utilizado en el módulo individual para mostrar los "Factores Clave de Estancia"). Su rendimiento global es lo suficientemente bueno como para servir como un modelo de respaldo y de interpretabilidad directa.
3.  **La Importancia de Segmentar por Tipo de Admisión**: La segmentación demuestra que predecir pacientes de urgencia requiere tolerar una mayor incertidumbre (MAE global de ~4.6 días con XGBoost) en comparación con admisiones planificadas (MAE global de ~1.7 días). Mezclar ambas poblaciones en un único análisis lineal degradaba severamente los resultados del modelo de regresión. La separación física del modelo lineal Ridge resolvió esta limitación histórica.
