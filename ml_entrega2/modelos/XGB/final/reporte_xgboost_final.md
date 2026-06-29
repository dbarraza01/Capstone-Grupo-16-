# Reporte Analítico — XGBoost Final (Consolidación)
**Predicción de Length of Stay (LOS) Hospitalario — Capstone Grupo 16**  
**Modelo:** `XGBRegressor` (Consolidado) | **Script:** `entrenar_xgboost_final.py`

---

## 1. Contexto y Objetivo

Este modelo representa la versión final y consolidada de XGBoost para la fase de modelamiento base avanzado. El objetivo principal de este script no es explorar nuevos parámetros, sino **evaluar y reportar rigurosamente el modelo ganador** utilizando las mejores prácticas de validación.

### Dataset Usado
Se seleccionó el **Escenario B** (`model_data_v3_escenario_B_charlson.csv`), el cual combina las variables agrupadas de la v2 con el **Índice de Comorbilidad de Charlson**. Esta elección fue fundamentada en la fase de hiper-tuning, donde demostró ser la arquitectura con la mejor relación entre capacidad predictiva (menor MAE) y retención del detalle clínico, superando tanto al baseline original como al escenario puro de Elixhauser.

---

## 2. Configuración del Modelo

El modelo fue entrenado utilizando los parámetros óptimos encontrados previamente mediante `RandomizedSearchCV` con validación cruzada y transformación asimétrica (`log1p`). 

### Hiperparámetros Fijos
```python
BEST_PARAMS = {
    "colsample_bytree": 0.6590,
    "gamma": 1.0031,
    "learning_rate": 0.0417,
    "max_depth": 5,
    "min_child_weight": 9,
    "n_estimators": 755,
    "reg_alpha": 2.6704,
    "reg_lambda": 4.8786,
    "subsample": 0.8077
}
```

### Justificación Metodológica: Transformación Logarítmica (`log1p` / `expm1`)
Dado que la variable objetivo (LOS) presenta una asimetría positiva extrema (la mayoría de estancias son cortas, pero existe una larga cola de estancias prolongadas), el entrenamiento directo sobre días reales causa sobreajuste en los valores bajos y mal rendimiento en los altos. 
Para mitigar esto, el modelo fue entrenado prediciendo el logaritmo de los días (`np.log1p(y_train)`). Posteriormente, **todas las predicciones fueron transformadas de vuelta a días reales** (`np.expm1()`) antes de calcular cualquier métrica reportada en este documento.

---

## 3. Metodología de Validación

Para garantizar que el modelo sea robusto y no sufra de *data leakage* o sobreajuste, se implementó una doble estrategia de validación:

1.  **Holdout Test (80/20):** Se separó el 20% de los pacientes (2,391) como un conjunto de datos puro e intocable. El modelo no vio estos datos durante su entrenamiento ni validación interna. Esta es la evaluación final y oficial.
2.  **Stratified K-Fold (5 Splits):** Exclusivamente sobre el 80% de entrenamiento (9,560 pacientes), se realizó una partición en 5 pliegues estratificados por tramos de LOS. 
    *   *Nota:* Este K-fold **no se utilizó para buscar parámetros** (tuning), sino para calcular la varianza del error y garantizar la estabilidad del modelo frente a diferentes muestras de pacientes.

**Advertencia metodológica:** La variable auxiliar `tramos_los` se utilizó únicamente para estratificar y balancear la distribución de pacientes largos y cortos en las divisiones de entrenamiento, prueba y validación cruzada. Esta variable no ingresó como predictor del modelo, con lo que se evitó fuga de información.

---

## 4. Resultados Globales (Holdout Test)

Evaluación final sobre los 2,391 pacientes invisibles para el modelo.

| Métrica | Valor | Interpretación |
| :--- | :--- | :--- |
| **MAE** | 3.057 días | Error absoluto promedio por paciente. |
| **RMSE** | 8.289 días | Penaliza fuertemente los errores graves (outliers). |
| **MedAE** | 0.902 días | El 50% de las predicciones tiene un error menor a ~0.9 días. |
| **WAPE** | 47.6% | Error porcentual ponderado por el volumen de días reales. |
| **Sesgo (Bias)** | -1.092 días | El modelo tiende a subestimar la estancia en ~1 día en promedio. |

### Distribución de Direccionalidad del Error
*   **Subestimación (Predicción < Real):** 49.18% de los casos.
*   **Sobreestimación (Predicción > Real):** 50.81% de los casos.

---

## 5. Estabilidad del Modelo (Stratified K-Fold)

Para validar que el modelo no dependió de un "split afortunado", estos son los promedios y desviaciones estándar obtenidos de las 5 validaciones cruzadas sobre el set de entrenamiento:

*   **MAE:** 3.018 ± 0.112 días
*   **RMSE:** 7.857 ± 0.799 días
*   **Recall PLOS:** 0.439 ± 0.071

**Nota:** La baja desviación estándar del MAE (±0,11) indica estabilidad entre pliegues. La variación del RMSE (±0,79) es mayor debido a la presencia desigual de pacientes con estancias extremas en cada subconjunto de validación.

---

## 6. Predicción de Estancias Prolongadas (PLOS ≥ 27 días)

Identificar a los pacientes extremos es el desafío clínico más importante del proyecto.

### Matriz de Confusión (Holdout)
*   **Verdaderos Negativos (TN):** 2,253 *(Estancias cortas correctamente predichas)*
*   **Falsos Positivos (FP):** 16 *(Modelo dijo largo, pero fue corto)*
*   **Falsos Negativos (FN):** 63 *(Modelo dijo corto, pero fue largo)*
*   **Verdaderos Positivos (TP):** 59 *(Estancias largas correctamente atrapadas)*

### Métricas PLOS
*   **Precision (VPP):** 78.67% — *Casi 8 de cada 10 pacientes que el modelo marcó como PLOS, efectivamente superaron los 27 días.*
*   **Recall (Sensibilidad):** 48.36% — *El modelo logró atrapar casi a la mitad de los pacientes críticos invisibles.*
*   **F1-Score:** 0.598

---

## 7. Análisis de Error por Tramos de Estancia (Holdout)

La evaluación por tramos demuestra el clásico problema de predicción en datos altamente asimétricos:

| Tramo Real | N Casos | LOS Real Prom. | LOS Pred Prom. | MAE | Sesgo (Bias) | Subestimación |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0-2 días** | 1,183 | 1.23 días | 1.97 días | 0.82 días | +0.73 días | 21.4% |
| **3-6 días** | 680 | 4.00 días | 3.59 días | 1.43 días | -0.41 días | 75.2% |
| **7-13 días** | 249 | 9.22 días | 7.35 días | 4.38 días | -1.86 días | 75.1% |
| **14-26 días** | 157 | 18.21 días | 12.82 días | 8.39 días | -5.38 días | 78.9% |
| **27+ días** | 122 | 49.15 días | 33.62 días | 24.14 días | -15.52 días | 81.1% |

### Interpretación de la subestimación
El comportamiento del modelo evidencia la **Paradoja de la Regresión Simétrica**.
En los pacientes con estancias de 0 a 2 días, que constituyen el grupo más frecuente, el modelo sobreestima en promedio 0,7 días. En cambio, en los pacientes con LOS ≥ 27 días se observa subestimación en el 81,1% de los casos y una diferencia promedio de 15,52 días entre el LOS estimado y el observado.

### Métricas de Costo Asimétrico (Evaluación de Negocio)
Dado que clínicamente es más riesgoso enviar un paciente grave a casa (subestimar) que retener a uno sano un día extra (sobreestimar), se calcularon métricas de impacto:
*   **Costo Asimétrico 2X:** 5.132 días *(Mide el MAE penalizando las subestimaciones al doble)*
*   **Costo Asimétrico 3X:** 7.207 días *(Mide el MAE penalizando las subestimaciones al triple)*

**Limitación:** Estos costos asimétricos tienen un propósito exclusivamente evaluativo. La función objetivo utilizada por XGBoost (`reg:squarederror`) no incorporó esta penalización durante el entrenamiento y asignó el mismo costo a errores positivos y negativos de igual magnitud.

---

## 8. Conclusiones y líneas de trabajo futuras

El **XGBoost consolidado** presenta el mejor desempeño entre los modelos avanzados evaluados. Su precisión de 78,67% en pacientes PLOS indica que aproximadamente ocho de cada diez predicciones de estancia prolongada corresponden a casos que efectivamente superan los 27 días.

No obstante, el recall de 48,36% muestra que el modelo identifica menos de la mitad de los pacientes PLOS reales. Esta limitación coincide con la subestimación observada en los tramos de mayor LOS.

### Propuestas metodológicas
Para combatir la subestimación crónica en estancias largas, la arquitectura debe evolucionar hacia el manejo de desbalances numéricos. Las técnicas sugeridas incluyen:
1.  **Regresión por Cuantiles (*Quantile Regression*):** Forzar al modelo a predecir la cota superior del 90º percentil, garantizando un colchón de seguridad.
2.  **Pesos por Muestra (*Sample Weights* / SMOGN):** Entrenar multiplicando la importancia de los pacientes largos, obligando al modelo a enfocarse en la cola derecha.
3.  **Funciones de Pérdida Asimétrica (Asymmetric Loss Functions):** Reemplazar el Error Cuadrático por una función de costo que castigue matemáticamente a los árboles si subestiman un valor.
