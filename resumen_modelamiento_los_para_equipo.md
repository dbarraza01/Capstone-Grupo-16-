# Resumen metodológico para el modelamiento de LOS hospitalario

**Proyecto:** Predicción de Length of Stay (LOS) hospitalario  
**Objetivo de este documento:** entregar contexto rápido al equipo sobre qué se está modelando, por qué se modela así, qué datasets se están comparando, por qué se usan Random Forest y XGBoost, cómo se hizo el tuning y cómo interpretar los resultados preliminares.

---

## 1. Problema que estamos resolviendo

El **LOS hospitalario** corresponde a la cantidad de días que un paciente permanece hospitalizado. En este proyecto, el objetivo es construir una prueba de concepto de aprendizaje supervisado que permita estimar el LOS a partir de información clínica estructurada del paciente, principalmente:

- diagnósticos ICD-10-CM;
- procedimientos ICD-10-PCS;
- variables generales del caso;
- medidas de carga comórbida como Charlson y Elixhauser.

La dificultad principal es que el LOS no se distribuye de forma normal. La mayoría de los pacientes tiene estancias cortas, mientras que una minoría tiene estancias muy largas. En nuestros datos, esto se observa como una **cola derecha larga**, con pacientes de varios días o incluso cientos de días. Por eso no basta con mirar el error promedio global: también hay que analizar cómo se comporta el modelo en pacientes de estancia prolongada.

---

## 2. Qué significa hacer aprendizaje supervisado en este proyecto

El modelo aprende a partir de ejemplos históricos:

```text
X = variables del paciente
y = LOS real del paciente
```

Ejemplo conceptual:

| Paciente | Diagnósticos agrupados | Procedimientos agrupados | Urgencia | Comorbilidad | LOS real |
|---|---|---|---:|---:|---:|
| A | E66, I10 | 0DB | 1 | 2 | 8 |
| B | O80, Z37 | — | 0 | 0 | 2 |
| C | J18, N18 | 5A1 | 1 | 5 | 21 |

Durante el entrenamiento, el modelo sí ve el `LOS real` porque esa es la respuesta que debe aprender. Pero al predecir un caso nuevo, el modelo recibe solo las variables `X` y debe estimar el LOS.

**Regla crítica:** `los_dias` nunca debe entrar como predictor dentro de `X`; debe usarse solo como target `y`.

---

## 3. Por qué no usamos los códigos crudos directamente

Los códigos ICD-10-CM e ICD-10-PCS son textos/categorías. Los modelos de árboles no deben recibirlos como listas crudas, sino como variables estructuradas, normalmente binarias:

```text
diag_E66 = 1 si el paciente tiene ese grupo diagnóstico
diag_E66 = 0 si no lo tiene
```

El problema es que hay miles de códigos posibles. Si cada código completo se transforma en una columna, se genera una matriz con muchas variables muy poco frecuentes. Esto se conoce como **alta dimensionalidad** y **sparsity**: muchas columnas, casi todas con ceros.

Consecuencias:

- mayor riesgo de sobreajuste;
- menor estabilidad estadística;
- más costo computacional;
- más dificultad de interpretación;
- códigos raros pueden parecer importantes por aparecer en pocos casos extremos.

Por eso se creó una estrategia de **agrupación jerárquica**.

---

## 4. Agrupación jerárquica de ICD/PCS

La lógica es conservar el mayor detalle posible solo cuando existe soporte suficiente en la muestra.

### Diagnósticos ICD-10-CM

Se usa la jerarquía:

```text
código completo → primeros 3 caracteres → capítulo
```

Ejemplo:

```text
E6601 → E66 → E
```

Regla usada:

1. Si el código completo aparece en al menos 20 pacientes únicos, se conserva completo.
2. Si aparece en menos de 20 pacientes, se reemplaza por sus primeros 3 caracteres.
3. Si ese grupo de 3 caracteres también aparece en menos de 20 pacientes, se reemplaza por el capítulo.

### Procedimientos ICD-10-PCS

Se usa la jerarquía:

```text
código completo → primeros 3 caracteres → sección
```

Ejemplo:

```text
0DB64Z3 → 0DB → 0
```

La regla de soporte mínimo es la misma.

### Qué termina siendo un “grupo”

Un grupo es la columna final que verá el modelo. Por ejemplo:

```text
diag_full_E6601
diag3_E66
diag_rare_cap_E
proc_full_0DB64Z3
proc3_0DB
proc_sec_0
```

Cada paciente tiene un `1` si pertenece al grupo y `0` si no pertenece.

---

## 5. Tratamiento de repeticiones

La unidad de análisis es el paciente/caso hospitalario, no cada fila de diagnóstico o procedimiento.

Por eso, si un paciente tiene el mismo código repetido varias veces, para la matriz binaria cuenta como una sola presencia:

```text
Paciente 1: E6601, E6601, I10
```

Se representa como:

| case_id | diag_E6601 | diag_I10 |
|---|---:|---:|
| Paciente 1 | 1 | 1 |

No como:

| case_id | diag_E6601 | diag_I10 |
|---|---:|---:|
| Paciente 1 | 2 | 1 |

Sin embargo, se pueden agregar variables complementarias de carga, como:

- número total de diagnósticos;
- número total de procedimientos;
- número de códigos repetidos;
- máximo número de repeticiones de un grupo.

Estas variables no duplican la presencia del código, solo capturan complejidad administrativa o clínica adicional.

---

## 6. Tratamiento del código `UUUUUU`

`UUUUUU` no representa un diagnóstico clínico interpretable como tal. En el proyecto se interpreta como una marca administrativa vinculada a ingreso por urgencia/no electivo.

Por eso, la decisión metodológica más limpia es:

- no tratar `UUUUUU` como diagnóstico clínico normal;
- no agruparlo jerárquicamente como ICD;
- excluirlo de las columnas diagnósticas predictivas;
- conservar la información de urgencia mediante una variable separada, por ejemplo `es_urgencia`.

Esto evita que el modelo aprenda una señal demasiado general y difícil de interpretar clínicamente.

---

## 7. Escenarios de modelamiento de datos

Se construyeron tres escenarios para comparar qué representación clínica funciona mejor.

### Escenario A: ICD/PCS agrupados

Contiene:

- variables base;
- diagnósticos ICD agrupados;
- procedimientos PCS agrupados;
- variables de repetición/carga.

Es el escenario más granular.

### Escenario B: ICD/PCS agrupados + Charlson

Toma el escenario A y agrega:

```text
charlson_index
```

El **Charlson Comorbidity Index** resume la carga de enfermedades crónicas severas. Es compacto y clínicamente interpretable. En nuestro procesamiento se calculó usando `comorbidipy`, con mapeo ICD-10 basado en Quan et al. (2005).

### Escenario C: Base + procedimientos + Elixhauser

En este escenario se reemplazan muchas columnas diagnósticas dispersas por categorías de **Elixhauser**, manteniendo variables base y procedimientos.

Elixhauser es más amplio que Charlson y considera 31 categorías de comorbilidad. En el proyecto se usaron categorías Elixhauser y el score ponderado de van Walraven.

**Interpretación esperada:**  
El escenario C es más compacto e interpretable, pero puede perder información diagnóstica específica que sí está presente en el escenario A o B.

---

## 8. Por qué usar Random Forest

Random Forest es un modelo supervisado basado en muchos árboles de decisión. Cada árbol aprende reglas sobre los datos y luego el bosque promedia las predicciones.

Ventajas para este problema:

- maneja relaciones no lineales;
- captura interacciones entre diagnósticos, procedimientos y comorbilidades;
- funciona bien con datos tabulares;
- es robusto frente a muchas variables;
- permite medir importancia de variables;
- es una buena referencia para comparar contra boosting.

Ejemplo conceptual de regla aprendida:

```text
Si tiene procedimiento digestivo complejo
y además tiene muchos diagnósticos secundarios
y además ingresó por urgencia
entonces el LOS esperado aumenta.
```

Random Forest es adecuado para una primera prueba de concepto porque es potente, relativamente interpretable y ampliamente usado en problemas de predicción clínica.

---

## 9. Por qué usar XGBoost

XGBoost también usa árboles, pero de forma secuencial. En vez de construir muchos árboles independientes como Random Forest, XGBoost va creando árboles que corrigen los errores de los árboles anteriores.

Ventajas:

- suele tener muy buen desempeño en datos tabulares;
- captura interacciones complejas;
- tiene regularización explícita;
- permite submuestreo de filas y columnas;
- puede superar a Random Forest si se ajusta bien.

Riesgo:

- puede sobreajustar si se permite demasiada profundidad, demasiados árboles o poca regularización.

Por eso el tuning de XGBoost debe mirar no solo el MAE, sino también la brecha entre error de entrenamiento y error de validación/test.

---

## 10. Qué es el tuning de hiperparámetros

Los hiperparámetros son decisiones del modelo que no se aprenden automáticamente, sino que se deben configurar o buscar. Ejemplos:

### Random Forest

- `n_estimators`: número de árboles.
- `max_depth`: profundidad máxima de cada árbol.
- `min_samples_leaf`: mínimo de pacientes por hoja.
- `min_samples_split`: mínimo de pacientes para dividir un nodo.
- `max_features`: proporción de variables consideradas en cada split.
- `max_samples`: proporción de pacientes usados por árbol.

### XGBoost

- `n_estimators`: número de árboles.
- `max_depth`: profundidad de cada árbol.
- `learning_rate`: qué tan fuerte aprende cada árbol.
- `subsample`: proporción de filas usadas por árbol.
- `colsample_bytree`: proporción de columnas usadas por árbol.
- `gamma`: ganancia mínima requerida para dividir.
- `reg_alpha`: regularización L1.
- `reg_lambda`: regularización L2.
- `min_child_weight`: mínimo peso/muestras en una hoja.

---

## 11. Cómo se hizo el tuning

Los scripts de tuning usan:

```text
RandomizedSearchCV
50 combinaciones aleatorias
5 folds de validación cruzada
MAE como criterio principal
```

Esto significa:

```text
50 combinaciones × 5 folds = 250 entrenamientos por escenario
```

Como hay tres escenarios:

```text
250 × 3 = 750 entrenamientos por modelo
```

Y como se evaluaron Random Forest y XGBoost:

```text
750 + 750 = 1.500 entrenamientos aproximados
```

Esto no es un entrenamiento final. Es una fase de búsqueda para encontrar combinaciones prometedoras de hiperparámetros.

**Importante:** la validación cruzada no es lo mismo que el test final. Para el resultado final del informe se debe separar un holdout test antes del tuning, hacer tuning solo en train y evaluar una única vez en test.

---

## 12. Por qué probar `log1p(LOS)`

El LOS tiene cola derecha larga. Si entrenamos directamente con `los_dias`, los pacientes extremos pueden dominar el ajuste.

La transformación:

```text
log1p(LOS) = log(1 + LOS)
```

comprime los valores grandes.

Ejemplos:

| LOS real | log(1 + LOS) |
|---:|---:|
| 0 | 0.000 |
| 1 | 0.693 |
| 3 | 1.386 |
| 27 | 3.332 |
| 60 | 4.111 |
| 262 | 5.572 |

El modelo puede entrenarse sobre `log1p(LOS)` y luego volver a días reales con:

```text
expm1(predicción) = exp(predicción) - 1
```

Las métricas finales deben reportarse siempre en días reales.

---

## 13. Métricas recomendadas

No se usará `R²`.

Las métricas principales para comparar modelos son:

| Métrica | Interpretación |
|---|---|
| MAE | Error absoluto promedio en días. Es la métrica principal. |
| RMSE | Penaliza errores grandes; útil para ver si el modelo falla en estancias largas. |
| MedAE | Error mediano; más robusto ante outliers. |
| WAPE | Error absoluto total relativo al total de LOS observado. |
| SMAPE | Error porcentual simétrico; útil con cuidado cuando hay LOS bajos. |
| Bias / error medio | Indica si el modelo sobreestima o subestima en promedio. |
| % subestimación | Proporción de pacientes cuyo LOS fue subestimado. |
| Error por tramo | Permite ver desempeño en estancias cortas, medias y largas. |
| Recall PLOS 27 | Entre pacientes con LOS ≥ 27, cuántos son detectados como largos por la predicción. |

### PLOS

PLOS significa:

```text
Prolonged Length of Stay
```

En este proyecto puede definirse operativamente como:

```text
PLOS = 1 si LOS ≥ 27 días
PLOS = 0 si LOS < 27 días
```

No se usa como predictor; se usa para evaluar si el modelo identifica bien las estancias prolongadas.

---

## 14. Resultados V1 (Fase Preliminar NO REGULARIZADA - ⚠️ NO USAR PARA EL INFORME)

> **ATENCIÓN EQUIPO:** Estos resultados corresponden a la primera iteración del tuning (V1). Como se puede ver en la columna "Train MAE", los modelos memorizaron los datos (sobreajuste masivo). **NO COPIEN ESTOS NÚMEROS** en la tesis como el resultado definitivo. Esta sección solo existe en este documento para justificar por qué tuvimos que regularizar el modelo más adelante.

Estos resultados vienen de los archivos originales no regularizados:

- `mejores_hiperparametros_random_forest.json`
- `mejores_hiperparametros_xgboost.json`

### Random Forest V1 (Sobreajustado)

| Escenario | Features | Mejor MAE CV | Train MAE | Desv. CV |
|---|---:|---:|---:|---:|
| A: ICD/PCS agrupados | 1650 | 2.9938 | 1.7045 | 0.1030 |
| B: ICD/PCS + Charlson | 1651 | 2.9912 | 1.6987 | 0.0902 |
| C: Elixhauser compacto | 560 | 3.1520 | 1.8919 | 0.0792 |

### XGBoost V1 (Sobreajustado)

| Escenario | Features | Mejor MAE CV | Train MAE | Desv. CV |
|---|---:|---:|---:|---:|
| A: ICD/PCS agrupados | 1650 | 2.9585 | 0.6914 | 0.0944 |
| B: ICD/PCS + Charlson | 1651 | 2.9556 | 0.6808 | 0.0767 |
| C: Elixhauser compacto | 560 | 3.0966 | 1.0494 | 0.0717 |

---

## 15. Por qué la V1 fracasó y obligó a hacer la V2 (Versión Final)

La brecha entre `Train MAE` y `CV MAE` en la V1 es inaceptable clínica y estadísticamente:
- **XGBoost B V1:** Train MAE = 0.6808 / CV MAE = 2.9556 (Gap de 2.27 días).

El modelo simplemente "se memorizó" a los pacientes de entrenamiento. Para arreglar esto y obtener nuestro **MODELO FINAL DEFINITIVO**, se aplicó el pipeline V2:
1. Separación del 20% como Holdout Test intocable desde el día cero.
2. Transformación logarítmica `log1p(LOS)` en el entrenamiento para estabilizar la asimetría de estancias largas.
3. Fuerte regularización L1 (Alpha) y L2 (Lambda) en XGBoost.

---

## 16. RESULTADOS FINALES (V2 Regularizada - ✅ USAR ESTOS NÚMEROS)

Estos son los **verdaderos resultados a reportar en el informe y la presentación**, obtenidos al evaluar los modelos finales sobre el **Set de Prueba (20% de pacientes nuevos que el modelo nunca vio)**.

| Modelo | Escenario | MAE (días) | RMSE (días) | MedAE (días) |
|---|:---:|:---:|:---:|:---:|
| **XGBoost Final** | B (Charlson) | **3.057** | **8.290** | **0.902** |
| **Random Forest Final** | B (Charlson) | 3.268 | 8.959 | 0.972 |
| **Regresión Lineal Baseline** | B (Charlson) | 6.311 | 53.070 | 0.866 |

### Capacidad para detectar pacientes críticos (PLOS ≥ 27 días)

| Métrica | Reg. Lineal | RF Final | XGBoost Final |
|:---|:---:|:---:|:---:|
| **Precision PLOS** | 65.98% | **80.85%** | 78.67% |
| **Recall PLOS** | **52.46%** | 31.15% | 48.36% |
| **F1 PLOS** | 58.45% | 44.97% | **59.90%** |

- **XGBoost Final:** Detecta casi a la mitad de los pacientes graves (Recall 48.36%) y cuando lanza la alerta, acierta el 78.67% de las veces (Precision). Es el modelo con el mejor equilibrio general (F1 de 59.9%).
- **Random Forest Final:** Es demasiado conservador. Solo detecta al 31.15% de los pacientes graves (Recall deficiente).
- **Regresión Lineal:** Aunque su Recall es alto (52.46%), es estadísticamente frágil en la regresión continua; al aplicarle el logaritmo para igualar condiciones, sus predicciones diarias explotan ante casos extremos.

### Subestimación por Tramo de Estadía

Todos los modelos sufren de **regresión a la media** (al ser la mayoría de estancias muy cortas, los modelos tienden a predecir a la baja en estancias muy largas).

| Tramo de Estadía | Reg. Lineal | RF Final | XGBoost Final |
|:---|:---:|:---:|:---:|
| **0–2 días** | 29.7% | **11.7%** | 21.5% |
| **3–6 días** | 65.1% | **68.8%** | 75.3% |
| **7–13 días** | 75.1% | 75.9% | **75.1%** |
| **14–26 días** | 76.4% | **85.4%** | 79.0% |
| **27+ días** | 63.9% | **89.3%** | 81.1% |

**Conclusión clave de esta tabla:** Si bien XGBoost es el mejor, subestima el 81.1% de los casos de más de 27 días. Esto significa que si un paciente en la vida real se va a quedar 40 días, XGBoost probablemente predecirá unos 24 días. Esta es una limitación conocida en la literatura médica y debe ser mencionada en la conclusión de la tesis como "Trabajo Futuro" (ej. implementar Regresión por Cuantiles).

---

## 17. Conclusión Definitiva de la Tesis

> **El mejor modelo final es XGBoost Regularizado operando sobre el Escenario B.**
> Su MAE final es de **3.057 días** en datos nunca antes vistos. Controla exitosamente el sobreajuste (Train MAE de 2.60 vs Test MAE de 3.05, cerrando el gap de 2.27 a solo 0.45 días) y domina el equilibrio entre predicción continua y clasificación de pacientes críticos.

---

## 18. Qué SÍ se debe afirmar categóricamente en la presentación/informe

- **XGBoost venció a la Regresión Lineal y a Random Forest**, especialmente en estabilidad (confirmado con Validación Cruzada de 5 pliegues).
- El **Índice de Charlson (Escenario B)** aporta información clínica valiosa sin afectar la generalización del modelo.
- La **transformación logarítmica (log1p)** fue indispensable matemáticamente para que el algoritmo optimizara de manera justa sin verse penalizado artificialmente por pacientes con más de 200 días de estancia.
- Todos los modelos basados en árboles tienen una limitación intrínseca: **tienden a subestimar** (regresión a la media) a los pacientes que se quedan meses. Esto se debe a que el modelo "fotografía" al paciente al ingreso, pero carece de variables dinámicas de evolución diaria para saber si el paciente empeoró en la UCI.

---

## 21. Referencias bibliográficas

1. Breiman, L. (2001). Random Forests. *Machine Learning, 45*, 5–32.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

3. Chrusciel, J., et al. (2021). Machine learning approaches for the prediction of hospital length of stay and patient discharge. *BMC Medical Informatics and Decision Making*.  
   Uso en este proyecto: justificación de modelos basados en árboles, codificación de variables clínicas y reducción de categorías infrecuentes.

4. Cho, J., et al. (2024). Machine learning prediction of postoperative length of stay using preoperative variables. *BMC Medical Informatics and Decision Making*.  
   Uso en este proyecto: comparación de modelos de regresión para LOS, evaluación por grupos de duración de estancia y uso de métricas como MAE/RMSE.

5. Jain, R., et al. (2024). Machine learning for hospital length of stay prediction using administrative and clinical features. *BMC Health Services Research*.  
   Uso en este proyecto: comparación de modelos predictivos, cuidado con leakage y validación mediante holdout y cross-validation.

6. Oshiro, T. M., Perez, P. S., & Baranauskas, J. A. (2012). How many trees in a random forest? *Machine Learning and Data Mining in Pattern Recognition*, 154–168.  
   Uso en este proyecto: motivación para evaluar distintos números de árboles y no asumir un valor universal.

7. Quan, H., Sundararajan, V., Halfon, P., et al. (2005). Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. *Medical Care, 43*(11), 1130–1139.  
   Uso en este proyecto: cálculo de Charlson y Elixhauser desde códigos ICD-10.

8. Stone, K., et al. (2022). A systematic review of hospital length of stay prediction using machine learning. *BMC Medical Informatics and Decision Making*.  
   Uso en este proyecto: respaldo general para el uso de machine learning en predicción de LOS y relevancia del preprocesamiento, selección de variables y validación.

9. van Walraven, C., Austin, P. C., Jennings, A., Quan, H., & Forster, A. J. (2009). A modification of the Elixhauser comorbidity measures into a point system for hospital death using administrative data. *Medical Care, 47*(6), 626–633.  
   Uso en este proyecto: score ponderado de Elixhauser.

10. Zeleke, A. A., et al. (2023). Prediction of prolonged length of stay using machine learning models. *Frontiers in Artificial Intelligence*.  
    Uso en este proyecto: distinción entre regresión de LOS y clasificación de estancia prolongada.

---

## 22. Fuentes internas del proyecto

Archivos usados para este resumen:

- `procesamiento_features_v3.py`
- `reporte_analisis_comorbilidades_v3.md`
- `tuning_random_forest.py`
- `tuning_xgboost.py`
- `mejores_hiperparametros_random_forest.json`
- `mejores_hiperparametros_xgboost.json`

