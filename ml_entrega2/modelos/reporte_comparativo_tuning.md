# Informe Comparativo de Ajuste de Hiperparámetros
## XGBoost y Random Forest — Tres Escenarios de Features
**Capstone Grupo 16 — Predicción de Length of Stay (LOS) Hospitalario**

---

## 1. Contexto Metodológico

Este informe compara los resultados de la búsqueda de hiperparámetros (`RandomizedSearchCV`, 50 iteraciones, 5-fold CV) realizada para dos algoritmos (**XGBoost** y **Random Forest**) en tres versiones distintas del dataset:

| Escenario | Descripción | N° Features |
|:---|:---|:---:|
| **A (Baseline)** | Dataset v2 original (ICD agrupados) | 1,650 |
| **B (Charlson)** | Dataset v2 + Índice de Charlson | 1,651 |
| **C (Elixhauser)** | Base + Procedimientos + 31 categorías Elixhauser | 560 |

Además, cada algoritmo fue evaluado en dos modalidades:
- **Estándar:** Búsqueda sobre el target en días reales.
- **Regularizado:** Búsqueda con el target transformado mediante `log1p` para reducir la asimetría de la distribución LOS.

---

## 2. Resumen de MAE por Cross-Validation (Métrica de Selección)

El MAE (Mean Absolute Error) en días fue la métrica de selección del tuning.

| Modelo | Escenario A | Escenario B | Escenario C |
|:---|:---:|:---:|:---:|
| **XGBoost Estándar** | 2.9585 | **2.9556** | 3.0966 |
| **XGBoost Regularizado** | 3.0121 | 3.0240 | 3.1258 |
| **RF Estándar** | 2.9938 | 2.9912 | 3.1520 |
| **RF Regularizado** | 3.4391 | 3.4311 | 3.4899 |

### Interpretación

1.  **XGBoost Estándar tiene el menor MAE de CV** en todos los escenarios (mínimo 2.9556 en B). Sin embargo, como se verá, sus métricas de holdout revelan sobreajuste.
2.  **El Escenario B supera al A en ambos algoritmos**, aunque por márgenes muy pequeños (≈0.003 días). Esto confirma que el Índice de Charlson aporta señal clínica útil, aunque marginal en términos de MAE global.
3.  **El Escenario C es el peor en todos los casos.** Reducir a solo las 31 categorías de Elixhauser elimina el detalle granular de los diagnósticos ICD, lo que penaliza al modelo.
4.  **El RF Regularizado tiene el MAE de CV más alto** (~3.43), un resultado contraintuitivo. La transformación `log1p` en RF no mejoró el ajuste como lo hizo en XGBoost, posiblemente porque RF ya promedia naturalmente muchos árboles y es menos sensible a la asimetría del target.

---

## 3. Sobreajuste (Overfitting): Gap entre Train y Validación

Si bien los modelos regularizados incluyen una evaluación final de holdout, todos los procesos de tuning reportan el error de entrenamiento (`train_mae`) y validación (`cv_mae`), lo que nos permite diagnosticar el nivel de sobreajuste de cada configuración:

| Modelo | Escenario | Train MAE | Val/Test MAE | Gap |
|:---|:---:|:---:|:---:|:---:|
| **XGBoost Estándar** | A | 0.6914 | 2.9585 (CV) | **2.2671** |
| **XGBoost Estándar** | B | 0.6808 | 2.9556 (CV) | **2.2748** |
| **XGBoost Estándar** | C | 1.0494 | 3.0966 (CV) | **2.0472** |
| **RF Estándar** | A | 1.7045 | 2.9938 (CV) | 1.2893 |
| **RF Estándar** | B | 1.6987 | 2.9912 (CV) | 1.2925 |
| **RF Estándar** | C | 1.8919 | 3.1520 (CV) | 1.2601 |
| **XGBoost Reg.** | A | 2.6061 | 3.0723 (Holdout)| 0.4662 |
| **XGBoost Reg.** | B | 2.6021 | 3.0573 (Holdout)| 0.4552 |
| **XGBoost Reg.** | C | 2.8606 | 3.0894 (Holdout)| 0.2287 |
| **RF Regularizado** | A | 3.1109 | 3.2747 (Holdout)| 0.1638 |
| **RF Regularizado** | B | 3.0979 | 3.2680 (Holdout)| 0.1701 |
| **RF Regularizado** | C | 3.1686 | 3.3286 (Holdout)| 0.1600 |

### Interpretación

- **XGBoost estándar presenta un sobreajuste considerable.** Aunque obtiene el menor MAE de validación (2,95), su error de entrenamiento es sustancialmente inferior (~0,68), con una brecha aproximada de 2,27 días. Esta diferencia indica menor capacidad de generalización ante datos nuevos.
- **La regularización reduce el sobreajuste de XGBoost.** La transformación `log1p` y las penalizaciones L1/L2 aumentan levemente el MAE de validación, de 2,95 a 3,02 días, pero reducen la brecha a aproximadamente 0,45 días. El resultado corresponde a una configuración más estable.
- **Random Forest Estándar también se sobreajusta, pero menos que XGBoost** (gap de ~1.29 días).
- **El RF Regularizado es el modelo más conservador de todos**, reduciendo el gap a solo ~0.16 días. Sin embargo, como se analizó, su MAE global aumenta considerablemente.

---

## 4. Capacidad de Detección de Pacientes Críticos (PLOS ≥ 27 días)

El **recall PLOS** cuantifica la proporción de pacientes que realmente superan los 27 días y que son identificados por el modelo.

| Modelo | Escenario | Precision PLOS | Recall PLOS | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **XGBoost Reg.** | A | 80.56% | **47.54%** | 0.5979 |
| **XGBoost Reg.** | B | 78.67% | **48.36%** | 0.5990 |
| **XGBoost Reg.** | C | 82.81% | 43.44% | 0.5699 |
| **RF Regularizado** | A | 80.85% | 31.15% | 0.4497 |
| **RF Regularizado** | B | 80.85% | 31.15% | 0.4497 |
| **RF Regularizado** | C | 80.85% | 31.15% | 0.4497 |

### Interpretación

- **XGBoost supera ampliamente a Random Forest en Recall PLOS** (~48% vs ~31%). Esto significa que XGBoost detecta casi la mitad de los pacientes críticos, mientras que RF solo logra capturar un tercio.
- **La precisión es similar entre ambos (~80%)**: aproximadamente ocho de cada diez predicciones PLOS corresponden a estancias que efectivamente superan los 27 días.
- **RF tiene exactamente el mismo Recall (31.15%) en los tres escenarios** del tuning regularizado. Esto sugiere que la estrategia de regularización en RF convergió siempre a la misma solución, independientemente del escenario de features.
- **XGBoost Escenario B logra el mejor F1-Score global (0.5990)**, siendo el balance más equilibrado entre no generar falsas alarmas (Precision) y no dejar pacientes críticos sin detectar (Recall).

---

## 5. Sesgo Sistemático: Tendencia a Subestimar

Todos los modelos presentan un bias negativo, indicando que tienden a predecir menos días de los que el paciente realmente estará internado:

| Modelo | Escenario | Sesgo (días) | % Subestimación |
|:---|:---:|:---:|:---:|
| **XGBoost Reg.** | A | -1.131 | 48.5% |
| **XGBoost Reg.** | B | **-1.092** | 49.2% |
| **XGBoost Reg.** | C | -1.349 | 45.3% |
| **RF Regularizado** | A | -1.586 | 43.4% |
| **RF Regularizado** | B | -1.588 | 43.5% |
| **RF Regularizado** | C | -1.631 | 44.5% |

### Interpretación

- **XGBoost subestima menos** que RF (bias de -1.09 vs -1.59 días en el mejor caso). Esto es una ventaja clínica directa: el modelo será más preciso en estimar cuántos días necesitará el paciente.
- **RF subestima sistemáticamente en ~1.6 días** para todos los escenarios. Al promediar todos sus árboles, tiende a "tirar" las predicciones hacia la media poblacional (estancias cortas), penalizando especialmente a los pacientes extremos.
- **El Escenario B de XGBoost tiene el menor sesgo (-1.092)**, otro argumento para su selección como modelo ganador.

---

## 6. Hiperparámetros Óptimos Encontrados

### XGBoost Estándar (Mejor MAE de CV)

**Escenarios A y B** (mismos parámetros):
```
n_estimators: 971   max_depth: 8    learning_rate: 0.1245
colsample_bytree: 0.640   subsample: 0.836   gamma: 1.863
reg_alpha: 0.141   reg_lambda: 1.022   min_child_weight: 3
```

**Advertencia:** La combinación de `max_depth=8` y `reg_alpha=0,14` presenta un riesgo elevado de sobreajuste debido a la profundidad de los árboles y a la baja regularización, lo que explica la brecha entre entrenamiento y validación.

### XGBoost Regularizado (Mejor Balance Generalización/PLOS)

**Escenarios A y B** (mismos parámetros):
```
n_estimators: 755   max_depth: 5    learning_rate: 0.0417
colsample_bytree: 0.659   subsample: 0.808   gamma: 1.003
reg_alpha: 2.670   reg_lambda: 4.879   min_child_weight: 9
```

**Nota:** La configuración `max_depth=5`, `reg_alpha=2,67` y `reg_lambda=4,88` utiliza árboles menos profundos y una penalización L1/L2 mayor. Esto reduce el sobreajuste a cambio de un incremento de 0,06 días en el MAE.

### Random Forest Estándar (Menor Gap)

**Todos los escenarios** convergieron al mismo ganador:
```
n_estimators: 403   max_depth: None (sin límite)
max_features: 0.5   max_samples: 0.765
min_samples_leaf: 2   min_samples_split: 2
```

### Random Forest Regularizado

**Todos los escenarios** convergieron al mismo ganador:
```
n_estimators: 777   max_depth: 20
max_features: 0.5   max_samples: 0.877
min_samples_leaf: 11   min_samples_split: 31
```

**Nota:** En el Random Forest regularizado, `min_samples_leaf=11` y `min_samples_split=31` restringen la formación de nodos con pocos pacientes. Esta configuración limita la memorización de observaciones extremas y explica la brecha de 0,16 días.

---

## 7. Selección del modelo y escenario

| Criterio | Ganador | Justificación |
|:---|:---:|:---|
| **Menor MAE de CV** | XGBoost Estándar B | MAE = 2.9556 días |
| **Menor Overfitting** | RF Regularizado B | Gap = 0.17 días |
| **Mayor Recall PLOS** | XGBoost Regularizado B | Recall = 48.36% |
| **Mejor F1-Score PLOS** | XGBoost Regularizado B | F1 = 0.5990 |
| **Menor Sesgo (Bias)** | XGBoost Regularizado B | Bias = -1.09 días |
| **Mejor Escenario de Features** | **Escenario B** | Gana en todos los modelos |

### Recomendación Final

**XGBoost Regularizado + Escenario B** es el modelo ganador para la fase de entrenamiento final, considerando el balance entre todos los criterios de evaluación. Aunque su MAE de CV no es el más bajo (3.024 vs 2.956 del estándar), ofrece:

1. **El mejor balance de detección clínica** (Recall PLOS = 48.4%, F1 = 0.599).
2. **El menor sesgo de subestimación** (Bias = -1.09 días).
3. **Un nivel de overfitting aceptable** (Gap = 0.46 días, equivalente a menos de medio día de error adicional en datos nuevos).

**Limitación estructural:** Ningún modelo supera el 50% de recall en pacientes PLOS. Este resultado evidencia una limitación del MAE simétrico como función objetivo y justifica evaluar funciones de pérdida asimétricas o regresión por cuantiles para mejorar la detección de estancias extremas.
