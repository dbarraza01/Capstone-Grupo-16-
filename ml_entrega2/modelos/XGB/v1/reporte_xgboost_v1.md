# 📋 Reporte Analítico — XGBoost v1
**Predicción de Length of Stay (LOS) Hospitalario — Capstone Grupo 16**  
**Modelo:** `XGBRegressor_v1` | **Script:** `entrenar_xgboost_v1.py`

---

## 1. Contexto y Objetivo

Este modelo tiene **dos propósitos simultáneos**:

1. **Comparar** directamente con el Random Forest v1 usando los mismos datos, el mismo split y la misma transformación del target — para aislar si las diferencias de desempeño se deben al algoritmo o a los datos.
2. **Establecer una nueva línea base** más sólida antes de implementar las técnicas avanzadas de la Fase 3 (SMOGN, pesos, modelo de dos etapas).

---

## 2. Configuración del Modelo

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 300 | Igual que RF v1 para comparación justa |
| `max_depth` | 6 | Menor que RF (8) — XGBoost es más propenso a overfitting al ser secuencial |
| `learning_rate` | 0.1 | Tasa estándar para primera iteración; controla cuánto corrige cada árbol |
| `subsample` | 0.8 | 80% de filas por árbol — reduce overfitting y varianza |
| `colsample_bytree` | 0.8 | 80% de features por árbol — estándar para datos esparsos |
| `reg_alpha` | 0.1 | Regularización L1 — penaliza features redundantes en dataset de 0.58% densidad |
| `reg_lambda` | 1.0 | Regularización L2 estándar |
| `objective` | `reg:squarederror` | Regresión de mínimos cuadrados |
| `random_state` | 42 | Misma semilla que RF v1 |

**Transformación del target:** `log1p(los_dias)` → entrenamiento / `expm1` → predicción. Idéntica al RF v1.  
**Split:** 80/20 estratificado por tramos de LOS, `random_state=42`. Idéntico al RF v1.

---

## 3. Distribución del Dataset (Idéntica al RF v1)

| Tramo LOS | Proporción |
|-----------|-----------|
| 0–2 días | 49.5% |
| 3–6 días | 28.4% |
| 7–13 días | 10.4% |
| 14–26 días | 6.6% |
| 27+ días | 5.1% |

---

## 4. Archivos de Salida

### 4.1 `predicciones_xgboost_v1.csv`
Predicciones individuales para los 2,391 pacientes del test set.

| Columna | Descripción |
|---------|-------------|
| `case_id` | Identificador del paciente |
| `los_real` | LOS real en días |
| `los_pred` | LOS predicho por XGBoost |
| `error` | `los_pred − los_real` (negativo = subestimación) |
| `abs_error` | Valor absoluto del error |
| `subestima` | 1 si el modelo predijo menos días de los reales |
| `tramo_los` | Tramo real del paciente |
| `plos_real` | 1 si LOS real ≥ 27 días |
| `plos_pred` | 1 si XGBoost predijo LOS ≥ 27 días |

### 4.2 `metricas_xgboost_v1.csv`

| Métrica | Valor XGB v1 | Valor RF v1 | Mejora |
|---------|-------------|------------|--------|
| MAE | **2.97 días** | 4.16 días | ↓ 28.6% |
| RMSE | **7.69 días** | 11.13 días | ↓ 30.9% |
| MedAE | **0.83 días** | 1.38 días | ↓ 39.9% |
| Precision PLOS | **0.747** | 0.0 | ↑ de cero |
| Recall PLOS | **0.484** | 0.0 | ↑ de cero |
| F1 PLOS | **0.587** | 0.0 | ↑ de cero |

### 4.3 `metricas_por_tramo_xgboost_v1.csv`

| Tramo | N | LOS Real Prom. | LOS Pred. Prom. | MAE | MedAE | Error Medio | % Subestima |
|-------|---|---------------|-----------------|-----|-------|-------------|-------------|
| 0–2 días | 1,183 | 1.24 días | 1.90 días | 0.76 | 0.46 | +0.66 | 22.2% |
| 3–6 días | 680 | 4.00 días | 3.74 días | 1.46 | 0.99 | -0.26 | 71.6% |
| 7–13 días | 249 | 9.22 días | 7.71 días | 4.47 | 3.85 | -1.51 | 75.5% |
| 14–26 días | 157 | 18.21 días | 13.05 días | 8.66 | 8.50 | -5.16 | 80.9% |
| **27+ días** | **122** | **49.16 días** | **33.48 días** | **22.43** | **17.55** | **-15.67** | **80.3%** |

### 4.4 `matriz_confusion_plos_xgb_v1.csv`

| | Pred: Corto (<27d) | Pred: Largo (≥27d) |
|---|---|---|
| **Real: Corto (<27d)** | TN = 2,249 | FP = 20 |
| **Real: Largo (≥27d)** | **FN = 63** | **TP = 59** |

### 4.5 `xgboost_v1.pkl`
Modelo serializado. Cargable con `joblib.load()` para predicciones futuras sin reentrenar.

---

## 5. Interpretación de los Resultados

### 5.1 El salto más importante: XGBoost dejó de estar "ciego" al PLOS

El hallazgo más significativo de este modelo es que el Recall PLOS pasó de **0.0% a 48.4%**. Esto significa que de los 122 pacientes que realmente tuvieron LOS ≥ 27 días, XGBoost detectó correctamente a **59 de ellos** — algo que el Random Forest fue incapaz de hacer con los mismos datos.

¿Por qué? La diferencia fundamental es el mecanismo de aprendizaje:

- **Random Forest:** Genera 300 árboles independientes y promedia sus votos. Los 300 árboles, entrenados sobre el mismo dataset con 49.5% de LOS cortos, aprenden individualmente a predecir estancias cortas. Al promediar 300 "votos" sesgados, el resultado nunca supera los 20 días.
- **XGBoost:** Los árboles se construyen **secuencialmente**. Cada nuevo árbol se entrena para corregir los errores del anterior. Después de las primeras iteraciones (que aprenden a predecir estancias cortas), las siguientes iteraciones se enfocan en los **residuos** — es decir, en los pacientes de LOS largo que se están prediciendo mal. Esto permite que el modelo eventualmente aprenda patrones específicos de estancias prolongadas.

### 5.2 Mejora global sustancial en todas las métricas de regresión

- **MAE: 4.16 → 2.97 días** (−28.6%): El error promedio se redujo casi 1.2 días por paciente.
- **RMSE: 11.13 → 7.69 días** (−30.9%): Los errores en casos extremos se redujeron significativamente. El RMSE penaliza cuadráticmaente los errores grandes, por lo que esta mejora indica que XGBoost comete menos errores catastróficos en los outliers.
- **MedAE: 1.38 → 0.83 días** (−39.9%): El error "típico" bajó de 1.38 a 0.83 días. Para el 50% central de los pacientes, XGBoost es casi medio día más preciso que RF.

### 5.3 El gradiente de subestimación cambió radicalmente

En el RF v1, la subestimación era prácticamente total para estancias medianas y largas (93–100%). En XGBoost, el patrón cambió:

| Tramo | RF v1 (% subestima) | XGB v1 (% subestima) | Mejora |
|-------|---------------------|----------------------|--------|
| 0–2 días | 0% | 22.2% | (Ahora subestima algo) |
| 3–6 días | 83.1% | 71.6% | ↓ 11.5 pp |
| 7–13 días | 92.8% | 75.5% | ↓ 17.3 pp |
| 14–26 días | 99.4% | 80.9% | ↓ 18.5 pp |
| 27+ días | 100.0% | 80.3% | ↓ 19.7 pp |

Dos observaciones importantes:

1. **En estancias largas (27+), el modelo aún subestima en un 80% de los casos.** La predicción media para este grupo es 33.5 días cuando la realidad es 49.2 días. Sin embargo, predecir "33 días" en lugar de "10 días" (RF v1) ya es enormemente más útil para la gestión hospitalaria.

2. **En el tramo 0–2 días, apareció un 22.2% de subestimación.** Esto es esperable: como XGBoost ahora dedica más esfuerzo a los casos largos, "sacrifica" un poco de precisión en los pacientes de estancia muy corta. Es el trade-off inherente al boosting.

### 5.4 Análisis de la Matriz de Confusión PLOS

De los 122 pacientes con LOS ≥ 27 días en el test set:
- **59 detectados (TP):** El modelo alertó correctamente sobre pacientes complejos
- **63 no detectados (FN):** Pacientes con estancias largas clasificados como "normales" — el riesgo clínico más importante

De los 2,269 pacientes con LOS corto:
- **20 falsas alarmas (FP):** El hospital se prepararía para un paciente largo que en realidad sale antes — molesto pero no peligroso
- **2,249 correctos (TN):** La gran mayoría de los pacientes simples se identifican bien

La **Precision de 74.7%** indica que cuando XGBoost dice "este paciente se quedará más de 27 días", tiene razón en casi 3 de cada 4 casos. Esto es valioso para gestión hospitalaria porque las falsas alarmas son pocas.

El **Recall de 48.4%** es el área de oportunidad: todavía se escapan la mitad de los casos prolongados. Sin embargo, pasar de 0% a 48.4% con los mismos datos y solo cambiando el algoritmo es una mejora fundamental.

---

## 6. Ventajas del Modelo XGBoost v1

| Ventaja | Descripción |
|---------|-------------|
| **Mejora en todas las métricas** | MAE −28.6%, RMSE −30.9%, MedAE −39.9% respecto al RF v1 |
| **PLOS detectable** | Recall del 48.4% vs. 0% del RF — el avance más importante |
| **Precision alta (74.7%)** | Las alarmas PLOS son confiables; pocas falsas alertas |
| **Rango de predicción ampliado** | Predice hasta ~80–90 días vs. máximo de 20 días del RF |
| **Reproducible** | `random_state=42` garantiza resultados idénticos |
| **Regularización nativa** | `reg_alpha` y `reg_lambda` controlan el overfitting en el dataset esparso |

---

## 7. Desventajas y Limitaciones

| Desventaja | Descripción | Gravedad |
|------------|-------------|----------|
| **Recall PLOS al 51.6% de error** | Todavía se escapan 63 de 122 pacientes con LOS largo | 🔴 Alta |
| **MAE tramo 27+ = 22.4 días** | Predice ~33 días cuando la realidad es ~49 días | 🔴 Alta |
| **Subestimación sistemática persiste** | 80% de los casos LOS ≥27 se subestiman | 🟡 Media |
| **Trade-off en estancias cortas** | 22.2% de subestimación en tramo 0–2 (era 0% en RF) | 🟡 Baja |
| **Features sin severidad** | Sigue sin capturar qué tan grave es la condición del paciente | 🔴 Alta |

---

## 8. Diagnóstico Técnico — ¿Por qué persiste la subestimación?

El 80% de subestimación en el tramo 27+ no es una falla del algoritmo: es un **problema de datos**. Con solo el 5.1% de los pacientes en el tramo 27+, el modelo tiene ~487 ejemplos de LOS largo para aprender durante el entrenamiento. En un dataset de 1,650 features, 487 ejemplos no son suficientes para aprender con solidez los patrones de estancias extremas.

Además, los features actuales (códigos ICD binarios) no distinguen la *severidad* dentro de un mismo diagnóstico. Un `diag_I10 = 1` (hipertensión) puede corresponder a una crisis hipertensiva de 60 días o a un control de rutina de 1 día. Sin información de severidad, el modelo no puede aprender esta diferencia.

---

## 9. Comparación Final RF v1 vs. XGBoost v1

| Métrica | RF v1 | XGB v1 | Ganador |
|---------|-------|--------|---------|
| MAE global | 4.16 días | **2.97 días** | XGBoost |
| RMSE global | 11.13 días | **7.69 días** | XGBoost |
| MedAE global | 1.38 días | **0.83 días** | XGBoost |
| MAE tramo 0–2 | **1.34 días** | 0.76 días | XGBoost |
| MAE tramo 27+ | 38.31 días | **22.43 días** | XGBoost |
| Recall PLOS | 0.0% | **48.4%** | XGBoost |
| Precision PLOS | 0.0% | **74.7%** | XGBoost |
| F1 PLOS | 0.0 | **0.587** | XGBoost |

> **Conclusión:** XGBoost v1 supera al Random Forest v1 en absolutamente todas las métricas evaluadas, usando exactamente los mismos datos. Esto confirma que el principal cuello de botella del RF v1 era algorítmico (el promediado de árboles). XGBoost es el algoritmo base para los modelos de la Fase 3.

---

## 10. Pasos Futuros (Fase 3)

Los resultados del XGBoost v1 apuntan a dos prioridades claras:

### 🔴 Prioridad 1 — Features de Severidad (Fase 2 del plan)
El Recall PLOS del 48.4% es un techo que no se puede superar solo ajustando hiperparámetros. Es necesario agregar información de **severidad clínica** (Índice de Charlson, Elixhauser, features de interacción) para que el modelo distinga entre un "caso leve" y un "caso grave" con el mismo código ICD.

### 🔴 Prioridad 2 — Manejo del desbalance (Fase 3 del plan)
Con los mismos features, el siguiente paso es aplicar:
- **Pesos de muestra:** Asignar peso 10x a pacientes de LOS ≥ 27 días para que el modelo "pague más" por equivocarse en ellos
- **SMOGN:** Generar muestras sintéticas de casos extremos para compensar los ~487 ejemplos insuficientes del grupo 27+
- **Modelo de dos etapas:** Entrenar un clasificador PLOS + regresores separados para estancias cortas y largas

### 🟡 Prioridad 3 — Ajuste de hiperparámetros
Con los features v3, explorar `learning_rate` más bajo (0.05) con más `n_estimators` (500) y validación cruzada K-Fold para seleccionar los mejores hiperparámetros de forma robusta.

---

## 11. Conclusión

> XGBoost v1 representa un avance sustancial respecto al RF v1. Con los mismos datos, logró reducir el MAE en un 28.6%, el RMSE en un 30.9% y — lo más importante — pasó de un Recall PLOS del 0% al 48.4%. Esto demuestra que el mecanismo de boosting secuencial es significativamente más adecuado para distribuciones de LOS asimétricas que el promediado de Random Forest.

> **Sin embargo, XGBoost v1 tampoco es apto para uso clínico.** Todavía pierde a la mitad de los pacientes con estancia prolongada y comete errores de ~22 días en los casos más graves. El siguiente paso es enriquecer los features con información de severidad (Fase 2) y aplicar técnicas de balanceo de datos (Fase 3).

> **El algoritmo está validado. El foco ahora es mejorar los datos.**

---

*Análisis generado sobre los archivos de salida de `entrenar_xgboost_v1.py`. Cohorte: 11,951 pacientes, 9,560 train / 2,391 test. Dataset: `model_data_ml_v2.csv`.*
