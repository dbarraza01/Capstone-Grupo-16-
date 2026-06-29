# Plan de Implementación — Rediseño Completo de Modelamiento de Machine Learning
**Capstone Grupo 16 — Enfoque Operacional, Modelos en Dos Etapas y Estratificación por Urgencia**

---

## 1. Conceptos básicos y glosario

Esta sección define los términos principales de machine learning utilizados en el proyecto:

*   **Modelo de clasificación o clasificador:** Modelo diseñado para estimar la probabilidad de pertenencia a una categoría. En este proyecto calcula la probabilidad de que un paciente presente una estancia de 14 días o más.
*   **Modelo de regresión o regresor:** Modelo diseñado para predecir una variable numérica continua; en este caso, los días de estancia hospitalaria.
*   **Stacking o apilamiento:** Estrategia que conecta modelos de forma secuencial. La salida del primer modelo se incorpora como una variable adicional para el segundo.
*   **Sesgo hacia el promedio (*regression to the mean*):** Tendencia de los modelos de regresión a producir estimaciones cercanas al centro de la distribución cuando optimizan errores promedio. En una distribución asimétrica de LOS, este comportamiento puede reducir la capacidad de representar estancias extremas.
*   **Tuning o ajuste de hiperparámetros:** Proceso de búsqueda de los valores de configuración que optimizan una métrica de validación. Estos valores se evalúan mediante combinaciones definidas antes del entrenamiento final.
*   **RandomizedSearchCV:** Función de `scikit-learn` que prueba combinaciones aleatorias de hiperparámetros y evalúa cada combinación con validación cruzada (K-Folds). Es más eficiente que probar todas las combinaciones posibles (Bergstra & Bengio, 2012).
*   **XGBClassifier y XGBRegressor:** Son modelos de ML de la librería `xgboost`. El primero se especializa en clasificar (estimar probabilidades de superar los 14 días) y el segundo en regresar (predecir el número exacto de días).

---

## 2. El Flujo de Trabajo Explicado Paso a Paso

El flujo de modelado no ocurre todo al mismo tiempo. Es un proceso estrictamente **secuencial** (uno detrás de otro) y se divide en 3 grandes etapas:

```text
Etapa 1: Segmentación (Urgentes vs. No Urgentes)
   │
   ├─► Subpoblación Urgencias (es_urgencia == 1) ────┐
   └─► Subpoblación No Urgentes (es_urgencia == 0) ──┼─► [Se aplican las Etapas 2 y 3 de forma independiente]
                                                      │
Etapa 2: Clasificación de Riesgo (XGBClassifier) ◄───┘
   │
   ├─► Pregunta: ¿LOS >= 14 días? (Etiqueta Real: Sí = 1, No = 0)
   └─► Resultado: Columna extra 'prob_los_14' (ej. 0.85)
   │
Etapa 3: Regresión de Días Exactos (XGBRegressor)
   │
   ├─► Variables: Datos clínicos originales + Columna 'prob_los_14'
   └─► Resultado: Predicción exacta de días (ej. 18.5 días)
```

### Etapa 1: Segmentación Inicial (Estratificación)
Los pacientes se separan según su vía de ingreso en dos conjuntos:
1.  **Pacientes de Urgencia** (`es_urgencia == 1`).
2.  **Pacientes Programados/No Urgencia** (`es_urgencia == 0`).
A partir de aquí, todo lo que sigue se entrena por separado para cada grupo. No se mezclan, porque operan bajo dinámicas clínicas y de gestión de camas radicalmente diferentes.

### Etapa 2: Clasificación de Riesgo (¿Se queda más de 14 días?)
En esta etapa se entrena un **modelo de clasificación** (`XGBClassifier`).
*   **Entrenamiento:** El modelo recibe las variables clínicas y una etiqueta binaria: `1` para LOS ≥ 14 días y `0` para LOS < 14 días.
*   **Rango de días:** El clasificador no utiliza los tramos de LOS durante el entrenamiento. Los tramos 0-2, 3-6, 7-13, 14-26 y 27+ se reservan para evaluar el error según la duración real de la estancia.
*   **Cálculo de la probabilidad:** `XGBClassifier` combina la salida de múltiples árboles y transforma la puntuación resultante mediante una función logística. El valor obtenido se encuentra entre 0 y 1 y representa la probabilidad estimada de pertenecer a la clase PLOS.
*   **Justificación de la probabilidad:** Es necesario distinguir entre el entrenamiento y la inferencia en un entorno hospitalario:
    *   **Durante el entrenamiento:** Se conocen los días exactos y la condición PLOS observada.
    *   **Durante la inferencia:** Solo están disponibles las variables clínicas definidas para el momento de predicción.
    *   La Etapa 2 estima el riesgo PLOS a partir de esas variables y entrega dicha probabilidad a la Etapa 3.

### Etapa 3: Regresión de Días Exactos (Predicción de LOS)
En esta etapa se entrena un **modelo de regresión** (`XGBRegressor`).
*   **¿Qué variables recibe?** Recibe todas las variables clínicas de ingreso (igual que el clasificador) **MÁS** la probabilidad calculada en la Etapa 2 como una nueva columna predictora.
*   **¿Cómo funciona el Stacking detalladamente aquí?** 
    La probabilidad aporta al regresor una señal continua de riesgo. Su contribución se combina con las demás variables clínicas para estimar los días de estancia; no constituye por sí sola una decisión determinista ni elimina necesariamente el sesgo hacia el promedio.

---

## 3. ¿Cómo funciona el K-Folds en Stacking y por qué evita el Data Leakage?

La fuga de información ocurriría si el clasificador se ajustara con todo el conjunto de entrenamiento y sus probabilidades sobre esos mismos pacientes se utilizaran para entrenar el regresor. En ese caso, la segunda etapa recibiría estimaciones excesivamente optimistas y no representativas del comportamiento frente a pacientes nuevos.

Para evitarlo se generan predicciones **out-of-fold (OOF)** mediante validación cruzada de cinco pliegues durante el entrenamiento:

```text
Paso 1: Dividir el dataset de entrenamiento en 5 grupos (Folds 1 a 5).

Paso 2: Generar probabilidades "limpias" para cada grupo de forma independiente:

   Intento 1: Entrenar clasificador con [Fold 2, 3, 4, 5] ──► Predice prob. para [Fold 1] (OOF)
   Intento 2: Entrenar clasificador con [Fold 1, 3, 4, 5] ──► Predice prob. para [Fold 2] (OOF)
   Intento 3: Entrenar clasificador con [Fold 1, 2, 4, 5] ──► Predice prob. para [Fold 3] (OOF)
   Intento 4: Entrenar clasificador con [Fold 1, 2, 3, 5] ──► Predice prob. para [Fold 4] (OOF)
   Intento 5: Entrenar clasificador con [Fold 1, 2, 3, 4] ──► Predice prob. para [Fold 5] (OOF)

Paso 3: Unir todas las predicciones OOF. Ahora todo el conjunto de entrenamiento
        tiene una columna de probabilidad generada por modelos que NUNCA vieron
        a esos pacientes durante su entrenamiento.

Paso 4: Entrenar el XGBRegressor usando este dataset con la columna de probabilidades.
```

### Propagación de errores entre etapas
La salida del clasificador puede transmitir error al regresor. El uso de probabilidades out-of-fold busca que la segunda etapa se entrene con una señal que reproduzca mejor las condiciones de inferencia y reduzca el riesgo de fuga de información. Su aporte neto debe comprobarse empíricamente mediante el estudio de ablación.

1.  **El regresor aprende del "ruido realista":** Al entrenar el regresor de la Etapa 3 utilizando las probabilidades *Out-of-Fold* (que contienen los errores normales del clasificador), el regresor aprende matemáticamente a **no confiar ciegamente** en la probabilidad. Si el clasificador asigna un 0.60 a un paciente que terminó quedándose 5 días, el regresor aprende a equilibrar esa probabilidad con las variables clínicas del paciente para moderar su predicción final.
2.  **Representación de la cola de la distribución:** La probabilidad PLOS proporciona una señal adicional para distinguir pacientes con diferente riesgo de estancia prolongada. El efecto observado depende de los datos y no garantiza por sí mismo una reducción del error final.

---

## 4. Búsqueda de Hiperparámetros dentro del Pipeline de Dos Etapas

Esta sección es crítica: el pipeline de dos etapas tiene **dos modelos de ML** que necesitan sus propios hiperparámetros óptimos. No basta con tener el flujo conceptual correcto; cada modelo necesita que le encontremos la mejor configuración (cuántos árboles usar, qué tan profundos, qué tan rápido aprender, cuánta regularización aplicar, etc.).

El ajuste también es secuencial: primero se seleccionan los hiperparámetros del clasificador y luego se ajusta el regresor utilizando la configuración resultante de la primera etapa.

### 4.1 El Macroproceso Completo con Tuning Integrado

El siguiente diagrama muestra el flujo completo desde la separación de datos hasta la evaluación final, incluyendo dónde encaja el tuning. Hay dos niveles distintos de partición que no deben confundirse:

1.  **Split operacional 80/20:** se hace una sola vez sobre toda la muestra, antes del tuning. Aquí la estratificación usa la combinación `es_urgencia + tramo_los`, para que el holdout conserve la proporción de pacientes urgentes/programados y de cada tramo de estancia.
2.  **Folds internos del tuning:** se hacen después, solo dentro del 80% de entrenamiento de cada subpoblación. En el clasificador se usa `StratifiedKFold(5)` para conservar la proporción de la etiqueta binaria `LOS >= 14`; en el regresor se usa `KFold(5)` porque la variable objetivo es continua.

Por lo tanto, el split no se hace "dentro del tuning". El tuning recibe los archivos ya separados en train/holdout y trabaja exclusivamente con el train.

```text
╔═══════════════════════════════════════════════════════════════════════════╗
║  PASO 0 — SPLIT OPERACIONAL GLOBAL                                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  0.1 Crear tramos de LOS:                                               ║
║      0-2, 3-6, 7-13, 14+ (PLOS)                                        ║
║                                                                         ║
║  0.2 Crear estrato combinado para cada paciente:                        ║
║      stratum = es_urgencia + "_" + tramo_los                            ║
║                                                                         ║
║      Ejemplos:                                                          ║
║      urgente_0-2, urgente_3-6, urgente_7-13, urgente_14+                ║
║      programado_0-2, programado_3-6, programado_7-13, programado_14+    ║
║                                                                         ║
║  0.3 Separar 80% Entrenamiento / 20% Holdout usando stratify=stratum    ║
║                                                                         ║
║  0.4 Exportar los archivos ya segmentados:                              ║
║      datos_train_urgente.csv, datos_train_programado.csv                ║
║      datos_holdout_urgente.csv, datos_holdout_programado.csv            ║
║                                                                         ║
║  0.5 El Holdout NO SE TOCA durante tuning ni entrenamiento              ║
║                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║  PASOS 1 A 5 — SE EJECUTAN POR SUBPOBLACIÓN                             ║
║  (Urgente y Programado se entrenan por separado usando sus propios CSV)  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  PASO 1 — TUNING DEL CLASIFICADOR (Etapa 2)                            ║
║  ├── Algoritmo: RandomizedSearchCV                                      ║
║  ├── Modelo: XGBClassifier o RandomForestClassifier                     ║
║  ├── Dataset: solo el train de la subpoblación correspondiente           ║
║  ├── Etiqueta interna: y_bin = 1 si LOS >= 14, 0 si LOS < 14            ║
║  ├── CV interna: StratifiedKFold(5) sobre y_bin                         ║
║  │   Cada fold mantiene una proporción similar de pacientes PLOS/no PLOS║
║  ├── Scoring: ROC-AUC (mide qué tan bueno es separando los de           ║
║  │   14+ días de los de menos de 14)                                    ║
║  ├── n_iter: 50 combinaciones aleatorias de hiperparámetros             ║
║  └── Salida: best_params_clf.json (mejores hiperparámetros              ║
║      del clasificador para cada algoritmo)                              ║
║                                                                         ║
║  PASO 2 — GENERACIÓN DE PROBABILIDADES OOF                             ║
║  ├── Usar los best_params_clf encontrados en el Paso 1                  ║
║  ├── Aplicar cross_val_predict con StratifiedKFold(5) para generar      ║
║  │   probabilidades "limpias" (sin leakage) para cada paciente          ║
║  │   del 80% de entrenamiento de esa subpoblación                       ║
║  └── Salida: Columna 'prob_los_14' añadida al dataset de               ║
║      entrenamiento para cada algoritmo correspondiente                  ║
║                                                                         ║
║  PASO 3 — TUNING DEL REGRESOR (Etapa 3)                                ║
║  ├── Algoritmo: RandomizedSearchCV                                      ║
║  ├── Modelo: XGBRegressor o RandomForestRegressor envuelto en           ║
║  │   TransformedTargetRegressor (con log1p/expm1 para manejar la        ║
║  │   distribución asimétrica del LOS)                                   ║
║  ├── Dataset: Variables clínicas + columna prob_los_14 del Paso 2       ║
║  ├── CV interna: KFold(5) sobre el 80% de entrenamiento                 ║
║  │   No es stratified porque el objetivo aquí es continuo: días de LOS  ║
║  ├── Scoring: MAE Asimétrico (penaliza 2x la subestimación,            ║
║  │   ver explicación abajo)                                             ║
║  ├── n_iter: 50 combinaciones aleatorias de hiperparámetros             ║
║  └── Salida: best_params_reg.json (mejores hiperparámetros              ║
║      del regresor para cada algoritmo)                                  ║
║                                                                         ║
║  PASO 4 — ENTRENAMIENTO FINAL                                           ║
║  ├── Entrenar clasificador definitivo con best_params_clf               ║
║  │   usando TODO el 80% de entrenamiento (para cada algoritmo)          ║
║  ├── Generar probabilidades OOF definitivas para entrenamiento          ║
║  ├── Entrenar regresor definitivo con best_params_reg                   ║
║  │   usando TODO el 80% + columna prob_los_14 correspondiente           ║
║  └── Guardar ambos modelos en archivos .joblib (para cada algoritmo)    ║
║                                                                         ║
║  PASO 5 — EVALUACIÓN EN HOLDOUT (20%)                                   ║
║  ├── Clasificador predice prob_los_14 para los pacientes del holdout    ║
║  ├── Regresor predice LOS exacto usando clínicas + prob_los_14          ║
║  └── Calcular métricas finales: MAE, RMSE, MedAE, ME, PUP,            ║
║      métricas por tramos                                                ║
║                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Ejemplo real de la estratificación del split operacional 80/20**

La estratificación hace que cada combinación `segmento + tramo_los` conserve aproximadamente un 20% de pacientes en holdout:

| Segmento | Tramo LOS | Train | Holdout | % Holdout |
|---|---:|---:|---:|---:|
| programado | 0-2 | 3930 | 983 | 20.01% |
| programado | 3-6 | 2006 | 501 | 19.98% |
| programado | 7-13 | 440 | 110 | 20.00% |
| programado | 14+ (PLOS) | 396 | 99 | 20.00% |
| urgente | 0-2 | 799 | 200 | 20.02% |
| urgente | 3-6 | 714 | 179 | 20.04% |
| urgente | 7-13 | 554 | 139 | 20.06% |
| urgente | 14+ (PLOS) | 721 | 180 | 19.98% |

Esto evita que, por azar, el holdout quede con demasiados pocos pacientes de un tramo importante. Por ejemplo, sin estratificación podría ocurrir que el holdout urgente tenga muy pocos pacientes `14+`, haciendo que la evaluación de PLOS sea poco representativa. Con estratificación, la proporción de cada estrato se conserva casi exactamente.

**Código conceptual del split operacional:**

```python
tramos = pd.cut(
    df["los_dias"],
    bins=[-1, 2, 6, 13, np.inf],
    labels=["0-2", "3-6", "7-13", "14+ (PLOS)"],
)

strata = df["es_urgencia"].astype(str) + "_" + tramos.astype(str)

train_df, holdout_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=strata,
)
```

**Código conceptual del fold interno del clasificador:**

```python
y_bin = (y_train_los >= 14).astype(int)

cv_clf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

search_clf = RandomizedSearchCV(
    estimator=clasificador,
    param_distributions=grid_clf,
    scoring="roc_auc",
    cv=cv_clf,
)

search_clf.fit(X_train_segmento, y_bin)
```

En síntesis, el primer `stratify` conserva la representatividad del holdout por `urgencia + tramo`. El segundo `StratifiedKFold` conserva la proporción de casos PLOS y no PLOS en los pliegues internos del clasificador.

**Justificación del ajuste secuencial:** El regresor depende de la probabilidad generada por el clasificador. Por esta razón, primero se ajusta el clasificador, luego se generan las probabilidades y finalmente se ajusta el regresor con `prob_los_14` disponible.

### 4.2 Tuning del Clasificador — Paso 1 en Detalle

**Objetivo:** Identificar la combinación de hiperparámetros que maximice la capacidad del `XGBClassifier` para discriminar el riesgo de una estancia de 14 días o más.

**Criterio de selección:** Se utiliza ROC-AUC, que mide la capacidad del clasificador para ordenar a los pacientes según su riesgo. Un valor de 1 representa discriminación perfecta y un valor de 0,5 equivale al rendimiento esperado por azar (Steyerberg et al., 2010; Cho et al., 2024).

**Espacio de búsqueda del clasificador (XGBClassifier):**

Los siguientes hiperparámetros se probarán en combinaciones aleatorias. Los rangos están basados en los valores del tuning regularizado previo del proyecto, adaptados para clasificación:

| Hiperparámetro | Rango de búsqueda | ¿Qué controla? |
|---|---|---|
| `n_estimators` | 200 – 800 | Cuántos árboles de decisión construir |
| `max_depth` | 3 – 7 | Profundidad máxima de cada árbol (árboles más profundos memorizan más, pero pueden sobreajustar) |
| `learning_rate` | 0.01 – 0.10 | Velocidad de aprendizaje (más lento = más conservador y generalizable) |
| `subsample` | 0.6 – 0.9 | Porcentaje de pacientes que cada árbol "ve" durante su construcción |
| `colsample_bytree` | 0.4 – 0.8 | Porcentaje de variables que cada árbol puede usar |
| `min_child_weight` | 3 – 15 | Mínimo de pacientes que debe tener una "hoja" del árbol para existir |
| `gamma` | 1 – 7 | Umbral mínimo de mejora para crear una nueva rama (fuerza la poda) |
| `reg_alpha` | 0 – 5 | Regularización L1 (empuja los pesos de variables inútiles a cero) |
| `reg_lambda` | 1 – 9 | Regularización L2 (suaviza los pesos para evitar valores extremos) |
| `scale_pos_weight` | 1 o ratio de clases | Compensar si hay muchos más pacientes de estancia corta que larga |

**Justificación de los rangos:** Los rangos de `max_depth`, `learning_rate`, `gamma`, `reg_alpha` y `reg_lambda` se heredan del tuning regularizado previo del proyecto (que probó con `n_iter=50` y demostró mejor generalización que el tuning sin regularizar). Limitar `max_depth ≤ 7` y subir el piso de `gamma ≥ 1` reduce el sobreajuste significativamente: en el tuning anterior, la brecha train-test del XGBoost regularizado fue de apenas 0.45 puntos de MAE frente a 2.27 del no regularizado. Bergstra & Bengio (2012) demuestran que RandomizedSearch con 50 iteraciones encuentra combinaciones dentro del 5% del óptimo global con alta probabilidad.

**Validación cruzada:** `StratifiedKFold(n_splits=5)`. Se usa "estratificado" porque la etiqueta binaria (≥14 días o no) puede estar desbalanceada. El *stratified* garantiza que cada fold tenga la misma proporción de pacientes de estancia larga y corta, asegurando que la evaluación del riesgo sea robusta y no se sesgue debido al desbalance natural de clases en admisiones hospitalarias (Stone et al., 2022; Steyerberg, 2019).

```python
# Ejemplo conceptual del Paso 1 (tuning del clasificador):
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

clf = XGBClassifier(objective="binary:logistic", eval_metric="auc")

search_clf = RandomizedSearchCV(
    estimator=clf,
    param_distributions=PARAM_DISTRIBUTIONS_CLF,  # tabla de arriba
    n_iter=50,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1
)
search_clf.fit(X_train, y_train_binario)  # y_train_binario: 1 si LOS>=14, 0 si no

best_params_clf = search_clf.best_params_
# Guardar en best_params_clf.json
```

### 4.3 Generación de Probabilidades OOF — Paso 2 en Detalle

Una vez seleccionados los hiperparámetros del clasificador, se generan probabilidades sin fuga de información para todo el conjunto de entrenamiento mediante `cross_val_predict` de scikit-learn:

```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Crear clasificador con los mejores hiperparámetros del Paso 1
clf_final = XGBClassifier(**best_params_clf, objective="binary:logistic")

# Generar probabilidades OOF (cada paciente recibe su probabilidad
# calculada por un modelo que NUNCA lo vio durante el entrenamiento)
prob_oof = cross_val_predict(
    clf_final,
    X_train, y_train_binario,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    method="predict_proba"
)[:, 1]  # Tomar solo la columna de probabilidad de clase 1 (≥14 días)

# Añadir como columna al dataset de entrenamiento
X_train["prob_los_14"] = prob_oof
```

Después de este paso, el dataset de entrenamiento tiene todas sus variables clínicas originales **más** una columna `prob_los_14` con valores como `[0.12, 0.87, 0.03, 0.95, ...]`.

### 4.4 Tuning del Regresor — Paso 3 en Detalle

**¿Qué estamos buscando?** La mejor combinación de hiperparámetros para que el modelo `XGBRegressor` prediga con la mayor precisión posible los días exactos de estancia, **penalizando especialmente la subestimación**.

**¿Por qué penalizar la subestimación?** Si el modelo predice 5 días pero el paciente realmente se queda 18 días, el hospital se queda sin camas y se genera un problema grave de planificación. En cambio, si predice 18 pero se queda 5, la cama se libera antes de lo esperado y el impacto operacional es menor. Esta asimetría del costo es fundamental en la planificación de camas hospitalarias y ha sido ampliamente documentada en la optimización de recursos y gestión de camas críticas (Harini et al., 2022; Song et al., 2015; Green, 2002).

**Penalización de la subestimación:** Se utiliza una función de evaluación denominada **MAE asimétrico**. Cuando la predicción es inferior al LOS observado, el error se multiplica por un factor α:

$$\text{MAE Asimétrico} = \frac{1}{N}\sum_{i=1}^{N} w_i \cdot |\hat{y}_i - y_i|$$

donde:
*   $w_i = \alpha$ si $\hat{y}_i < y_i$ (el modelo **subestimó**, predijo menos días de los reales → penalización fuerte)
*   $w_i = 1$ si $\hat{y}_i \geq y_i$ (el modelo sobreestimó o acertó → penalización normal)
*   $\alpha = 2$ significa que subestimar pesa el doble que sobreestimar

```python
# Función de scoring personalizada:
def asymmetric_mae(y_true, y_pred, alpha=2.0):
    """MAE que penaliza la subestimación α veces más que la sobreestimación."""
    errors = y_true - y_pred
    weights = np.where(errors > 0, alpha, 1.0)  # Si error>0 → subestimación
    return np.mean(weights * np.abs(errors))

# Crear scorer compatible con scikit-learn:
from sklearn.metrics import make_scorer
scorer_asimetrico = make_scorer(asymmetric_mae, greater_is_better=False, alpha=2.0)
```

**Justificación de α=2:** Los modelos anteriores evaluaban costos asimétricos con factores 2 y 3, pero no los incorporaban en el ajuste. Se adopta α=2 como valor inicial para representar un costo de subestimación superior al de sobreestimación, de acuerdo con la literatura de gestión hospitalaria citada (Harini et al., 2022; Song et al., 2015).

**Espacio de búsqueda del regresor (XGBRegressor):**

Los rangos son los mismos que se usaron en el tuning regularizado previo, ya que demostraron buena generalización:

| Hiperparámetro | Rango de búsqueda | ¿Qué controla? |
|---|---|---|
| `n_estimators` | 300 – 900 | Número de árboles |
| `max_depth` | 3 – 7 | Profundidad máxima de cada árbol |
| `learning_rate` | 0.01 – 0.10 | Velocidad de aprendizaje |
| `subsample` | 0.6 – 0.9 | % de pacientes por árbol |
| `colsample_bytree` | 0.4 – 0.8 | % de variables por árbol |
| `min_child_weight` | 3 – 15 | Mínimo de pacientes por hoja |
| `gamma` | 1 – 7 | Umbral de poda |
| `reg_alpha` | 0 – 5 | Regularización L1 |
| `reg_lambda` | 1 – 9 | Regularización L2 |

**Para Random Forest**, el espacio de búsqueda es:

| Hiperparámetro | Rango de búsqueda | ¿Qué controla? |
|---|---|---|
| `n_estimators` | 200 – 800 | Número de árboles |
| `max_depth` | 8, 10, 12, 15, 20, 25 | Profundidad máxima (sin `None` para evitar sobreajuste) |
| `min_samples_split` | 10 – 60 | Mínimo de pacientes para dividir un nodo |
| `min_samples_leaf` | 5 – 30 | Mínimo de pacientes por hoja |
| `max_features` | sqrt, log2, 0.3, 0.5, 0.7 | Porcentaje de variables por árbol |
| `max_samples` | 0.5 – 0.9 | % de pacientes para bootstrap |

**Transformación del target:** Siguiendo la práctica del tuning anterior y la recomendación de Manning & Mullahy (2001) para datos de distribución asimétrica como el LOS, el regresor se envuelve en `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`. Esto transforma los días a escala logarítmica durante el entrenamiento (comprimiendo la "cola larga" de pacientes con estancias extremas) y los devuelve a escala original para la predicción.

```python
# Ejemplo conceptual del Paso 3 (tuning del regresor):
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor

reg_base = XGBRegressor(objective="reg:squarederror")
reg = TransformedTargetRegressor(regressor=reg_base, func=np.log1p, inverse_func=np.expm1)

search_reg = RandomizedSearchCV(
    estimator=reg,
    param_distributions=PARAM_DISTRIBUTIONS_REG,  # tabla de arriba con prefijo "regressor__"
    n_iter=50,
    scoring=scorer_asimetrico,  # MAE Asimétrico (α=2)
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1
)
# X_train ahora incluye la columna prob_los_14 generada en el Paso 2
search_reg.fit(X_train, y_train_los)  # y_train_los: días exactos de estancia

best_params_reg = search_reg.best_params_
# Guardar en best_params_reg.json
```

### 4.5 ¿Por qué NO se Usa el Mismo Scoring para Ambos Modelos?

Cada modelo responde a una pregunta diferente y, por lo tanto, se evalúa con la métrica más adecuada para su pregunta:

| | Clasificador (Etapa 2) | Regresor (Etapa 3) |
|---|---|---|
| **Pregunta** | ¿Se queda ≥14 días? (Sí/No) | ¿Cuántos días exactos se queda? |
| **Tipo de respuesta** | Probabilidad (0 a 1) | Número continuo (días) |
| **Scoring del tuning** | ROC-AUC | MAE Asimétrico (α=2) |
| **¿Por qué esa métrica?** | Mide la calidad del ranking probabilístico (qué tan bien ordena riesgo alto vs. bajo) | Mide el error promedio en días, penalizando el doble cuando subestima |

---

## 5. Estructura Modular y Plan de Trabajo en la Carpeta `ml_operacional/`

**Preservación de la implementación anterior:** La carpeta original `ml/` se mantiene sin modificaciones. El nuevo desarrollo se organiza en una carpeta independiente denominada `ml_operacional/`.

La estructura interna será:

### 5.1 Comparación de Modelos y Línea Base

Para determinar el mejor método de predicción operacional en el contexto del hospital, se aplicará el mismo flujo de dos etapas (Clasificación + Regresión) y tuning completo a ambos algoritmos principales:
1.  **XGBoost (XGB)**
2.  **Random Forest (RF)**

Al finalizar el entrenamiento y la evaluación en el Holdout (20%), realizaremos una comparación directa entre ambos modelos utilizando todas las métricas de negocio.

**Comparación con el Modelo Base:**
Para tener un punto de referencia (línea base) que justifique la complejidad de estos modelos, compararemos el desempeño de XGBoost y Random Forest con un **Modelo Base de Regresión Lineal/Lasso**. Entrenaremos un modelo de Regresión Lineal/Lasso (LR) equivalente en la subcarpeta `LR/` que actúe como un comparable directo (utilizando las mismas variables de entrada y la misma separación de entrenamiento y holdout). Esto nos permitirá comparar los tres modelos bajo las mismas condiciones y demostrar cuantitativamente cuánto valor agregan los modelos no lineales avanzados con Stacking frente al enfoque clásico (Alsinglawi et al., 2024; Stone et al., 2022).

### 5.2 Estructura del Directorio

```text
ml_operacional/
├── utils/
│   └── metricas_operacionales.py      ← Funciones de evaluación (MAE, RMSE, ME, PUP, Asimétrico, etc.)
│
├── RF/                                ← Directorio para Random Forest
│   ├── tuning_rf.py                   ← Paso 1 (clf) + Paso 2 (OOF) + Paso 3 (reg) con RandomSearch
│   ├── entrenar_rf.py                 ← Paso 4: Entrenamiento final con best_params
│   └── evaluar_rf.py                  ← Paso 5: Evaluación en holdout con todas las métricas
│
├── XGB/                               ← Directorio para XGBoost
│   ├── tuning_xgb.py                  ← Paso 1 (clf) + Paso 2 (OOF) + Paso 3 (reg) con RandomSearch
│   ├── entrenar_xgb.py                ← Paso 4: Entrenamiento final con best_params
│   └── evaluar_xgb.py                 ← Paso 5: Evaluación en holdout con todas las métricas
│
└── LR/                                ← Directorio para Regresión Lineal / Lasso (Línea Base)
    ├── entrenar_lr.py                 ← Entrenamiento directo (LR tiene pocos hiperparámetros)
    └── evaluar_lr.py                  ← Evaluación comparativa
```

### Métricas que se implementarán en `metricas_operacionales.py`:
Para medir objetivamente el rendimiento clínico y de planificación de camas, incluiremos:
*   **MAE (Error Absoluto Medio):** Mide la desviación promedio general.
*   **RMSE (Raíz del Error Cuadrático Medio):** Mide la desviación penalizando los errores grandes.
*   **MedAE (Error Absoluto Mediano):** Desviación típica del 50% de los casos, robusto a outliers.
*   **Error Medio Firmado (ME):** Promedio de `Predicción - Real`. Si es negativo (ej. -2.5 días), indica subestimación sistemática.
*   **Porcentaje de Pacientes Subestimados (PUP):** Qué porcentaje de pacientes tuvo una estancia real mayor que la predicha.
*   **MAE Asimétrico:** Error absoluto medio con penalización α para subestimación (la función de scoring del tuning).
*   **Métricas Segmentadas por Tramos:** Cálculo automático de MAE, RMSE, MedAE, ME, PUP y MAE Asimétrico para los tramos `0-2`, `3-6`, `7-13` y `14+ (PLOS)` días de estancia real.
*   **Definición Operacional de PLOS:** Paciente con estancia prolongada si `LOS >= 14` días. Esta etiqueta se calcula tanto para el valor real como para la predicción:
    - `plos_real_14 = 1` si `los_dias_reales >= 14`.
    - `plos_pred_14 = 1` si `los_dias_predichos >= 14`.
*   **Precision PLOS 14:** Entre los pacientes que el modelo marcó como PLOS, mide qué proporción realmente tuvo `LOS >= 14`. Responde: "cuando el modelo alerta estancia prolongada, ¿qué tan confiable es esa alerta?".
    - Fórmula: `TP / (TP + FP)`.
*   **Recall PLOS 14:** Proporción de pacientes con `LOS >= 14` que fueron identificados por el modelo.
    - Fórmula: `TP / (TP + FN)`.
*   **F1 PLOS 14:** Promedio armónico entre Precision y Recall PLOS. Resume el balance entre evitar falsas alarmas y detectar suficientes pacientes prolongados.
    - Fórmula: `2 * precision * recall / (precision + recall)`.
*   **Accuracy PLOS 14:** Proporción total de pacientes correctamente clasificados como PLOS o no PLOS.
    - Fórmula: `(TP + TN) / N`.
*   **Matriz de Confusión PLOS 14:** Conteo de casos usados para interpretar las métricas de clasificación:
    - `TP`: PLOS real y PLOS predicho.
    - `FP`: no PLOS real, pero PLOS predicho.
    - `FN`: PLOS real, pero no PLOS predicho.
    - `TN`: no PLOS real y no PLOS predicho.
*   **Conteos PLOS:** `n_plos_real` y `n_plos_pred` permiten auditar cuántos pacientes prolongados existen realmente y cuántos pacientes fueron marcados por el modelo como prolongados.

---

## 6. Salidas y Outputs Esperados por Script (Visibilidad del Proceso)

Para asegurar la auditabilidad del proceso y que el equipo clínico y técnico comprenda qué ocurre en cada etapa, cada uno de los archivos de código deberá generar las siguientes salidas y registros obligatorios:

### 6.1 Módulos de Tuning (`tuning_rf.py` y `tuning_xgb.py`)
*   **Mensajes en Consola (Stdout):**
    - Log de inicio indicando subpoblación cargada y dimensiones del dataset de entrenamiento.
    - Progreso de la búsqueda aleatoria mostrando el tiempo de ejecución y el score obtenido en cada fold.
    - Resumen de los mejores hiperparámetros encontrados y su score final de CV.
*   **Archivos Guardados en Disco (en la subcarpeta de cada modelo):**
    - `best_params_clf_urgente.json` y `best_params_clf_programado.json` (parámetros óptimos del clasificador).
    - `best_params_reg_urgente.json` y `best_params_reg_programado.json` (parámetros óptimos del regresor).

### 6.2 Módulos de Entrenamiento (`entrenar_rf.py`, `entrenar_xgb.py`, `entrenar_lr.py`)
*   **Mensajes en Consola (Stdout):**
    - Confirmación de lectura exitosa de los parámetros óptimos JSON.
    - Log detallado del cálculo de probabilidades Out-of-Fold (OOF) para el conjunto de entrenamiento.
    - Registro del tiempo de entrenamiento del clasificador y regresor definitivos.
*   **Archivos Guardados en Disco (en una nueva carpeta `ml_operacional/modelos_guardados/`):**
    - Modelos finales entrenados con el 80% completo, guardados en formato `.joblib`.
    - Ejemplo: `clf_xgb_urgente.joblib`, `reg_xgb_urgente.joblib`, `clf_xgb_programado.joblib`, `reg_xgb_programado.joblib`.

### 6.3 Módulos de Evaluación (`evaluar_rf.py`, `evaluar_xgb.py`, `evaluar_lr.py`)
*   **Mensajes en Consola (Stdout):**
    - Confirmación de carga de los modelos guardados y el subconjunto de Holdout (20%).
    - Tabla impresa con las métricas globales del Holdout.
*   **Archivos Guardados en Disco (en la subcarpeta de cada modelo o en una carpeta `ml_operacional/reports/`):**
    - Reporte técnico de evaluación en Markdown (ej. `reporte_evaluacion_xgb.md`). Este reporte debe incluir:
      - Tabla comparativa de métricas globales.
      - Desglose de error medio (MAE y MedAE) por tramos de días reales.
      - Métricas operacionales claves (ME, PUP y MAE Asimétrico).
    - Archivo de datos de predicciones en CSV (ej. `predicciones_holdout_xgb.csv`) con las columnas: `case_id`, `los_dias_reales`, `prob_riesgo`, `los_dias_predichos`, `error_dias` y `es_urgencia` para permitir la posterior generación de gráficos.

### 6.4 Exportación de Datos Intermedios para Auditoría (Archivos .csv)
Para facilitar que el equipo clínico y técnico audite y entienda el comportamiento de los datos paso a paso, se guardarán obligatoriamente los siguientes conjuntos de datos en la carpeta `ml_operacional/data_splits/`:
*   **Segmentación Inicial (Paso 0):**
    - `datos_train_urgente.csv` y `datos_train_programado.csv` (el 80% de datos de entrenamiento segmentados por vía de ingreso).
    - `datos_holdout_urgente.csv` y `datos_holdout_programado.csv` (el 20% de datos de test/holdout segmentados).
*   **Probabilidades OOF del Clasificador (Paso 2 / Etapa 2):**
    - `train_con_prob_urgente.csv` y `train_con_prob_programado.csv`. Son los conjuntos de entrenamiento que incluyen la columna predictora adicional `prob_los_14` generada por la validación cruzada OOF. Esto permite inspeccionar directamente qué probabilidad de estancia larga (riesgo) se le estimó a cada paciente de entrenamiento antes de pasar a la Etapa 3 de regresión.

---

## 7. Plan de Verificación

Para confirmar que la implementación es correcta y que no afectó el trabajo previo:
1.  **Aislamiento:** Confirmar que no se editó ningún archivo dentro de `ml/`.
2.  **Pruebas Unitarias de Métricas:** Ejecutar un script rápido que valide que `metricas_operacionales.py` calcula correctamente el ME, el PUP, el MAE Asimétrico y los tramos sin errores de tipado o división por cero.
3.  **Flujo de Tuning Completo:** Correr `tuning_xgb.py` y verificar la creación de `best_params_clf.json` y `best_params_reg.json`.
4.  **Flujo de Entrenamiento:** Correr `entrenar_xgb.py` y comprobar la generación de los modelos guardados en formato `.joblib`.
5.  **Evaluación en Holdout:** Correr `evaluar_xgb.py` y verificar la generación del reporte con todas las métricas.

---

## 7. Justificación Académica y Referencias (APA 7)

*   **Alsinglawi, B. S., Alnajjar, F., Alorjani, M. S., Al-Shari, O. M., Munoz, M. N., & Mubin, O.** (2024). Predicting hospital stay length using explainable machine learning. *IEEE Access*, 12, 90571–90585. https://doi.org/10.1109/ACCESS.2024.3421295
*   **Bacchi, S., et al.** (2019). Length of stay, discharge disposition and hospital readmission among stroke patients. *Neurological Sciences*, 40(12), 2447–2454. https://doi.org/10.1007/s10072-019-04004-0
*   **Bergstra, J., & Bengio, Y.** (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(2), 281–305.
*   **Brier, G. W.** (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
*   **Carter, M. W., & Lapierre, S. D.** (2001). Scheduling emergency and elective patients in a hospital. *Health Care Management Science*, 4(2), 141-151. https://doi.org/10.1023/A:1011432420223
*   **Cho, H. N., Ahn, I., Gwon, H., et al.** (2024). Explainable predictions of a machine learning model to forecast the postoperative length of stay for severe patients. *BMC Medical Informatics and Decision Making*, 24, 350. https://doi.org/10.1186/s12911-024-02755-1
*   **Chrusciel, J., Girardon, F., Roquette, L., Laplanche, D., Duclos, A., & Sanchez, S.** (2021). The prediction of hospital length of stay using unstructured data. *BMC Medical Informatics and Decision Making*, 21(1), 351. https://doi.org/10.1186/s12911-021-01722-4
*   **Green, L. V.** (2002). How many hospital beds does a city need? *Inquiry*, 39(3), 224-236. https://doi.org/10.5034/inquiryjrnl_39.3.224
*   **Harini, A., Sivaraman, R., & Sundarraj, R. P.** (2022). Two-stage machine learning and stochastic optimization models for hospital bed management. *European Journal of Operational Research*, 301(2), 654-672. https://doi.org/10.1016/j.ejor.2021.11.002
*   **Manning, W. G., & Mullahy, J.** (2001). Estimating log models and transition models for nonnegative outcomes. *Journal of Health Economics*, 20(4), 461-494. https://doi.org/10.1016/S0167-6296(01)00086-7
*   **Marazzi, A., Paccaud, F., Ruffieux, C., & Beguin, C.** (1998). Fitting the distributions of length of stay by parametric models. *Medical Care*, 36(6), 915–927. https://doi.org/10.1097/00005650-199806000-00014
*   **Quan, H., Sundararajan, V., Halfon, P., Fong, A., Burnand, B., Luthi, J. C., ... & Ghali, W. A.** (2005). Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. *Medical Care*, 43(11), 1130–1139.
*   **Rajkomar, A., et al.** (2018). Scalable and accurate deep learning with electronic health records. *npj Digital Medicine*, 1(1), 18. https://doi.org/10.1038/s41746-018-0029-1
*   **Song, H., Tucker, A. L., & Murrell, K. L.** (2015). The effects of hospital capacity constraints on admitting decisions and patient outcomes. *Journal of Health Economics*, 40, 109-122. https://doi.org/10.1016/j.jhealeco.2014.12.002
*   **Steyerberg, E. W.** (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Update* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0
*   **Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., Pencina, M. J., & Kattan, M. W.** (2010). Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*, 21(1), 128-138. https://doi.org/10.1097/EDE.0b013e3181c30fb2
*   **Stone, K., Zwiggelaar, R., Jones, P., & Mac Parthaláin, N.** (2022). A systematic review of the prediction of hospital length of stay: Towards a unified framework. *PLOS Digital Health*, 1(4), e0000017. https://doi.org/10.1371/journal.pdig.0000017
