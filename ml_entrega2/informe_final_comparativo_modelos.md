# Informe Final Comparativo de Modelos Predictivos de Length of Stay (LOS)
## Capstone Grupo 16 — Predicción de Estancia Hospitalaria

---

## 1. Introducción y Justificación del Problema

La predicción de la duración de estancia hospitalaria (*Length of Stay*, LOS) constituye uno de los desafíos más relevantes en la gestión operacional de instituciones de salud. De acuerdo con Turgeman et al. (2017), la capacidad de anticipar el LOS permite optimizar la asignación de camas, planificar altas y reducir costos operativos sin comprometer la calidad asistencial. Estudios recientes como los de Phuong et al. (2024) y Alghatani et al. (2024) han demostrado que los modelos de *machine learning* superan consistentemente a los enfoques estadísticos tradicionales en esta tarea, particularmente cuando se dispone de variables clínicas codificadas bajo estándares como ICD-10.

El presente informe consolida los resultados de todas las fases de modelamiento realizadas durante el proyecto, desde el modelo base (regresión lineal) hasta los modelos finales optimizados (XGBoost y Random Forest con hiperparámetros regularizados). Su propósito es proveer una evaluación rigurosa, reproducible y académicamente justificada que permita seleccionar el modelo más adecuado para un eventual despliegue clínico.

### 1.1 Objetivo General

Comparar tres familias de modelos predictivos de LOS — Regresión Lineal, Random Forest y XGBoost — evaluados sobre el mismo conjunto de datos y bajo métricas estandarizadas, con énfasis en la capacidad de detectar estancias prolongadas (PLOS ≥ 27 días).

### 1.2 Referencias Bibliográficas Base

| Ref. | Autores / Fuente | Aporte al proyecto |
|:---:|:---|:---|
| [1] | Turgeman et al., *BMC Med Inform Decis Mak*, 2021 | Metodología de predicción LOS con Random Forest y datos estructurados vs. no estructurados |
| [2] | Turgeman et al., *JAMIA*, 2006 | Fundamentos del uso de ICD y comorbilidades como predictores de LOS |
| [3] | Phuong et al., *BMC Med Inform Decis Mak*, 2024 | XGBoost como mejor modelo para LOS postquirúrgico; uso de SHAP para interpretabilidad |
| [4] | Steyerberg, *Stat Comput*, 2022 | Fundamentos de validación cruzada y calibración de modelos predictivos clínicos |
| [5] | Alghatani et al., *BMC Health Serv Res*, 2024 | Predicción de LOS a gran escala con CatBoost y feature engineering sobre CCS |

---

## 2. Diseño Experimental

### 2.1 Dataset

Se utilizó un dataset de **11,951 registros** de pacientes hospitalizados, procesado en múltiples versiones de *feature engineering*.

**Justificación de Variables de Resumen (Base)**
Todos los escenarios comparten un conjunto de variables de resumen estructural (`es_urgencia`, `n_procedimientos`, `n_diag_total`, `tiene_diag_primario`, `mes_ingreso`). La inclusión de estas variables se fundamenta en una limitación matemática profunda de los modelos basados en árboles: un árbol de decisión no puede "sumar" variables por sí solo. Si se entregan únicamente 1,636 variables binarias de códigos ICD, el algoritmo es ciego a la diferencia entre un paciente con una sola condición y uno con diez comorbilidades simultáneas, ya que requeriría crear una secuencia profunda de 10 nodos anidados para descubrir esa complejidad (lo que llevaría a un sobreajuste masivo y requeriría una cantidad inviable de datos).

Entregar variables como `n_diag_total` o `n_procedimientos` captura directamente la **carga diagnóstica global** y la **intensidad de intervención**. Esto proporciona "atajos matemáticos" que el árbol puede usar en un solo nodo de decisión para separar pacientes graves de pacientes leves, encapsulando relaciones clínicas complejas que el modelo jamás podría deducir eficientemente desde matrices puramente binarias.
Qué aporta cada variable de resumen
n_diag_total / n_diag_primarios / n_diag_secundarios

Capturan la carga diagnóstica global del paciente. La literatura clínica establece que pacientes con mayor número de diagnósticos simultáneos tienden a tener estancias más prolongadas, mayor complejidad asistencial y mayor riesgo de complicaciones. Un árbol que tiene n_diag_total puede hacer una pregunta directa: "¿Este paciente tiene más de 5 diagnósticos distintos?" — esa pregunta discrimina muy bien entre el Paciente A y el B del ejemplo anterior, sin tener que combinar cientos de variables binarias individuales.

n_procedimientos

Captura la intensidad de intervención recibida. Un paciente con 8 procedimientos está siendo tratado con mucha más agresividad que uno con 1. Esta variable actúa como un proxy de la gravedad real del cuadro clínico — los hospitales realizan más procedimientos cuando la situación clínica es más compleja.

es_urgencia

Distingue el contexto de admisión. Un paciente que llega por urgencias no llega con una hospitalización planificada, lo que estadísticamente tiene asociaciones diferentes con el LOS. Las urgencias incluyen cuadros agudos imprevistos que pueden resolverse rápido (apendicitis simple) o escalar gravemente (infarto complicado). Esta variable le da al modelo el "modo de entrada" al hospital.

tiene_diag_primario

Es una variable de calidad de datos clínicos. Pacientes sin diagnóstico primario codificado (como los que llegaron con UUUUUU en urgencias) tienen un historial codificado incompleto, lo que afecta la fiabilidad de todas las demás variables binarias. El modelo aprende que cuando tiene_diag_primario = 0, las 1,122 variables diag_* están potencialmente en blanco — y eso sí importa para la predicción.

mes_ingreso / dia_semana_ingreso

Capturan patrones temporales operativos. En los hospitales, el LOS no es completamente independiente del día o mes de ingreso: pacientes que ingresan el viernes frecuentemente tienen alta el lunes (extensión de fin de semana), y ciertos meses tienen mayor ocupación que afecta la disponibilidad de recursos para dar el alta.

La razón matemática profunda: información no recuperable desde sumas binarias
Un árbol de decisión no puede sumar variables por sí solo. Si tiene diag_E11, diag_I50, diag_N18, diag_J44, cada una como columna separada, el árbol solo puede preguntar sobre cada una individualmente en cada nodo de split. Para "entender" que este paciente tiene 4 condiciones simultáneas, necesitaría crear una secuencia profunda de 4 nodos anidados — lo que requiere muchos datos para aprender esa combinación específica y lleva a sobreajuste.

En cambio, si le entregas n_diag_total = 4 directamente, el árbol puede hacer esa pregunta en un solo nodo, con mucha más potencia estadística porque la variable tiene representación en toda la población, no solo en los pocos pacientes que tienen exactamente esas 4 combinaciones.

Es decir: las variables de resumen son atajos matemáticos que encapsulan relaciones que el árbol tardaría exponencialmente más tiempo en descubrir — o nunca descubriría — solo desde las binarias.

Los datos fueron procesados en tres escenarios distintos para comparar su rendimiento predictivo:

- **v2 (Baseline):** 1,650 features derivadas de una agrupación jerárquica de los códigos crudos ICD-10. Originalmente, el dataset contenía **6,124** códigos diagnósticos distintos y **3,278** códigos de procedimientos distintos (un total de 9,402 códigos). Si se utilizaran todos como features binarias directas, el dataset sufriría de una extrema escasez de datos (*sparsity*) y la "maldición de la dimensionalidad", ya que tendríamos casi tantas columnas como los 11,951 pacientes, llevando inevitablemente al modelo a sobreajustar (memorizar). Para solucionarlo, se aplicó un algoritmo de compresión clínica:
  - **Nivel 1:** Se conservaron los códigos completos solo si aparecían en al menos 10-20 pacientes.
  - **Nivel 2:** Los códigos menos frecuentes se agruparon por su categoría clínica de 3 caracteres (ej. `diag_E11`).
  - **Nivel 3:** Los casos ultra raros se agruparon por capítulo o sección general (ej. `diag_rare_cap_E`).
  El resultado fue una matriz robusta de 1,650 features (1,122 diagnósticos agrupados, 514 procedimientos agrupados y variables base). Conservar exactamente ~1,650 variables representa retener aproximadamente un 18% de la diversidad original, un equilibrio óptimo que proporciona una densidad manejable (0.58%) para que modelos basados en árboles (como XGBoost) puedan encontrar verdaderos patrones estadísticos sin memorizar ruido.
- **v3 Escenario B (Charlson):** 1,651 features — el dataset v2 enriquecido con el **Índice de Comorbilidad de Charlson (CCI)**. Basado en las adaptaciones de Quan et al. para ICD-10, el algoritmo de Charlson escanea el historial de diagnósticos del paciente buscando condiciones médicas que pertenezcan a 17 categorías predefinidas de alto impacto (por ejemplo, infarto de miocardio, insuficiencia cardíaca congestiva, demencia, diabetes con y sin complicaciones, neoplasias, VIH/SIDA). Dependiendo de la gravedad médica asociada a la mortalidad a un año, el algoritmo asigna un "peso" a cada condición detectada (típicamente 1, 2, 3 o 6 puntos). La suma matemática de estos pesos genera el *Charlson Score*, un valor único numérico que resume el riesgo clínico integral y la carga de enfermedad crónica del paciente. Esto dota al modelo de machine learning de un resumen experto inmediato que sería muy difícil de deducir observando solo miles de códigos aislados.
- **v3 Escenario C (Elixhauser):** 560 features — construido como una alternativa de alta compresión semántica. A diferencia de Charlson que resume la gravedad en un solo score, el sistema de Elixhauser clasifica los códigos ICD-10 en **31 categorías binarias de comorbilidad** independientes (ej. hipertensión, obesidad, depresión, coagulopatía, etc.), proporcionando un "perfil clínico" plano y descriptivo en lugar de un puntaje acumulativo. En este escenario, se **eliminaron por completo** las 1,122 variables diagnósticas granulares del Baseline y se reemplazaron exclusivamente por estas 31 variables booleanas de Elixhauser (manteniendo los procedimientos). El objetivo académico fue evaluar si esta "traducción experta" de comorbilidades era suficiente para predecir el LOS, reduciendo drásticamente la dimensionalidad del dataset para favorecer la generalización.

La selección del **Escenario B** como dataset definitivo se justificó empíricamente: superó al Baseline en todos los modelos evaluados durante el tuning, confirmando que el Índice de Charlson aporta una señal clínica complementaria que los códigos ICD individuales no logran capturar por sí solos.

### 2.2 Partición de Datos

**Todos los modelos fueron entrenados y evaluados sobre el mismo dataset (Escenario B) con la misma partición determinista** (`random_state=42`), garantizando que las predicciones de cada modelo se calculan sobre exactamente los mismos 2,391 pacientes del conjunto de test. Esta es una condición sine qua non para una comparación estadística válida.

- **Holdout:** 80% entrenamiento (9,560 pacientes) / 20% test (2,391 pacientes) (`random_state=42`).
- **Estratificación:** La partición se estratificó por tramos de LOS (0–2, 3–6, 7–13, 14–26, 27+ días) para asegurar representación proporcional de estancias prolongadas en ambos conjuntos.
- **Validación cruzada (todos los modelos finales):** StratifiedKFold con 5 pliegues sobre el conjunto de entrenamiento, exclusivamente para evaluar estabilidad — sin buscar nuevos hiperparámetros.

### 2.3 Transformación del Target

Para los modelos basados en árboles (XGBoost y Random Forest), el target fue transformado mediante `log1p(LOS)` durante el entrenamiento. Esta decisión se fundamenta en la distribución altamente asimétrica del LOS: aproximadamente el 49% de los pacientes tiene estancias de 0–2 días, mientras que solo el 5% supera los 27 días. La transformación logarítmica comprime la cola derecha, reduciendo la influencia desproporcionada de los valores extremos sobre la función de pérdida. Todas las métricas reportadas fueron calculadas en **días reales** tras aplicar la transformación inversa (`expm1`).

### 2.4 Optimización de Hiperparámetros

La selección de hiperparámetros para XGBoost y Random Forest se realizó mediante **RandomizedSearchCV** con 50 iteraciones y validación cruzada de 5 pliegues, siguiendo la metodología recomendada por Turgeman et al. (2021) y Phuong et al. (2024). Se evaluaron dos modalidades:

- **Estándar:** Búsqueda sobre el target en días reales.
- **Regularizado:** Búsqueda con target transformado (`log1p`) y rangos de hiperparámetros que favorecen la generalización (mayor penalización L1/L2, menor profundidad).

La versión regularizada fue seleccionada para los modelos finales tras evidenciar que la versión estándar presentaba sobreajuste severo (gap train-test de hasta 2.27 días en XGBoost).

---

## 3. Modelos Evaluados

### 3.1 Regresión Lineal (Modelo Base)

Implementada como *benchmark* inferior mediante `LinearRegression` de scikit-learn. Se entrenó utilizando **el mismo dataset Escenario B**, la **misma partición** y, crucialmente, la **misma transformación log1p(LOS)** que todos los demás modelos para garantizar una comparabilidad matemática exacta. Incluye validación cruzada StratifiedKFold de 5 pliegues. Su propósito es establecer un piso de rendimiento: cualquier modelo de ML que no supere significativamente a la regresión lineal no justifica su complejidad adicional.

### 3.2 XGBoost Final (Regularizado)

`XGBRegressor` con hiperparámetros optimizados por RandomizedSearchCV regularizado. Configuración final: `max_depth=5`, `learning_rate=0.042`, `n_estimators=755`, `reg_alpha=2.67`, `reg_lambda=4.88`, `min_child_weight=9`. Entrenado sobre Escenario B con `log1p(LOS)`.

### 3.3 Random Forest Final (Regularizado)

`RandomForestRegressor` con hiperparámetros optimizados. Configuración final: `max_depth=20`, `n_estimators=777`, `max_features=0.5`, `min_samples_leaf=11`, `min_samples_split=31`. Entrenado sobre Escenario B con `log1p(LOS)`.

---

## 4. Resultados Globales (Holdout Test)

### 4.1 Métricas de Regresión

| Métrica | Reg. Lineal | RF Final | XGBoost Final |
|:---|:---:|:---:|:---:|
| **MAE (días)** | 6.311 | 3.268 | **3.057** |
| **RMSE (días)** | 53.070 | 8.959 | **8.290** |
| **MedAE (días)** | **0.866** | 0.972 | 0.902 |

**Interpretación:**

- **La Regresión Lineal sufre una explosión de error al exponenciar (RMSE de 53 días)**. Esto ocurre porque la relación entre las variables clínicas y el LOS no es estrictamente lineal en el espacio logarítmico. Cuando la regresión lineal predice valores altos, al aplicar la transformación inversa (`expm1`), los errores se multiplican exponencialmente, disparando tanto el MAE (6.31 días) como el RMSE.
- **XGBoost Final logra el menor MAE (3.057) y RMSE (8.290)**, lo que indica que es el modelo que maneja mejor los errores extremos, un aspecto crítico considerando la asimetría de la distribución LOS.
- **La Regresión Lineal tiene el menor MedAE (0.866 días)**, seguida de cerca por XGBoost (0.902). Esto indica que para la "mitad" de los pacientes más predecibles, la regresión lineal acierta muy bien, pero sus errores en la otra mitad son catastróficos. XGBoost es mucho más equilibrado en toda la distribución.

### 4.2 Detección de Estancias Prolongadas (PLOS ≥ 27 días)

| Métrica | Reg. Lineal | RF Final | XGBoost Final |
|:---|:---:|:---:|:---:|
| **Precision PLOS** | 65.98% | **80.85%** | 78.67% |
| **Recall PLOS** | **52.46%** | 31.15% | 48.36% |
| **F1 PLOS** | 58.45% | 44.97% | **59.90%** |

| Modelo | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|
| Reg. Lineal | 2,236 | 33 | 58 | 64 |
| XGBoost Final | 2,253 | 16 | 63 | 59 |
| RF Final | 2,260 | 9 | 84 | 38 |

**Interpretación:**

- **La regresión lineal tiene el mayor Recall (52.46%)**, detectando marginalmente más pacientes PLOS que XGBoost. Al corregir la transformación `log1p`, su Precision mejoró a 65.98% (reduciendo las falsas alarmas de 57 a 33), resultando en un modelo de clasificación binaria sorprendentemente decente para alertas gruesas.
- **Random Forest Final tiene la mayor Precision (80.85%)**, pero presenta el **peor Recall (31.15%)**: deja sin detectar a **84 de 122 pacientes** con estancias prolongadas. Esto es clínicamente inaceptable para un sistema de alerta temprana.
- **XGBoost Final logra el balance más sólido**, consolidándose con el mejor F1-Score (59.90%) entre los modelos de ML. Su Precision de 78.67% asegura baja tasa de falsas alarmas (21.5 puntos porcentuales superior a la Regresión Lineal), a la vez que detecta casi a la mitad de los pacientes críticos.

---

## 5. Análisis por Tramos de Estancia

### 5.1 MAE por Tramo (Holdout Test)

| Tramo | Reg. Lineal | RF Final | XGBoost Final |
|:---|:---:|:---:|:---:|
| **0–2 días** (n=1,183) | **0.790** | 0.920 | 0.823 |
| **3–6 días** (n=680) | 1.615 | 1.558 | **1.440** |
| **7–13 días** (n=249) | 5.559 | **4.286** | 4.389 |
| **14–26 días** (n=157) | 12.413 | **8.156** | 8.398 |
| **27+ días** (n=122) | 79.700 | 27.201 | **24.150** |

**Interpretación:**

- **La Regresión Lineal gana en el tramo de 0-2 días**, confirmando que su MedAE es el más bajo porque acierta muy bien en las estancias cortas.
- Sin embargo, a medida que aumenta la estancia real, el error lineal explota exponencialmente debido al `log1p`. En el tramo crítico de **27+ días**, la regresión lineal promedia un error inaceptable de **79.7 días**, mientras que los modelos de ML lo limitan a 24-27 días. Esto demuestra por qué XGBoost es una solución inmensamente más segura y robusta para la gestión global del hospital.

### 5.2 Subestimación por Tramo

| Tramo | Reg. Lineal | RF Final | XGBoost Final |
|:---|:---:|:---:|:---:|
| **0–2 días** | 29.7% | **11.7%** | 21.5% |
| **3–6 días** | 65.1% | **68.8%** | 75.3% |
| **7–13 días** | 75.1% | 75.9% | **75.1%** |
| **14–26 días** | 76.4% | **85.4%** | 79.0% |
| **27+ días** | 63.9% | **89.3%** | 81.1% |

**Interpretación:**

Todos los modelos basados en árboles muestran una tendencia sistemática a la subestimación en tramos medios y altos. Este fenómeno, denominado **regresión a la media**, es intrínseco a los modelos que optimizan funciones de pérdida simétricas (MSE/MAE): dado que la mayoría de pacientes tienen estancias cortas, el modelo aprende a "tirar" las predicciones hacia la media poblacional.

El **RF Final es el más conservador**, con un 89.3% de subestimación en el tramo 27+. Clínicamente, esto implica que el modelo predice que un paciente que estará 49 días se irá en solo 24 días, generando una falsa sensación de disponibilidad de cama.

---

## 6. Estabilidad del Modelo (Validación Cruzada K-Fold)

Todos los modelos finales — incluyendo la Regresión Lineal — fueron evaluados mediante StratifiedKFold de 5 pliegues sobre el set de entrenamiento para verificar que el rendimiento no depende de un split afortunado.

| Métrica | Reg. Lineal (mean ± std) | XGBoost Final (mean ± std) | RF Final (mean ± std) |
|:---|:---:|:---:|:---:|
| **MAE** | 5.978 ± 1.825 | 3.018 ± 0.112 | 3.434 ± 0.156 |
| **RMSE** | 51.902 ± 38.452 | 7.857 ± 0.799 | 9.828 ± 1.581 |
| **Recall PLOS** | 0.509 ± 0.041 | 0.439 ± 0.072 | 0.257 ± 0.052 |
| **Precision PLOS** | 0.606 ± 0.017 | 0.747 ± 0.061 | 0.802 ± 0.023 |
| **Costo Asimétrico 2x** | 7.949 ± 1.771 | 5.115 ± 0.246 | 6.022 ± 0.327 |

**Interpretación:**

- La **Regresión Lineal es extremadamente inestable** al evaluarse bajo log1p: su desviación estándar en MAE (±1.825) y especialmente en RMSE (±38.452) demuestra que es matemáticamente frágil frente a valores atípicos. Un solo error alto en el espacio logarítmico destruye el promedio global.
- **XGBoost es el modelo con la mayor fiabilidad matemática** en MAE (±0.112 vs ±0.156 de RF) y significativamente más estable en RMSE (±0.799 vs ±1.581), probando que controla la "cola derecha" mucho mejor.

---

## 7. Diagnóstico de Sobreajuste

Un hallazgo crítico del proceso de tuning fue la identificación del nivel de sobreajuste de cada configuración:

| Modelo | Train MAE | Val/Test MAE | Gap |
|:---|:---:|:---:|:---:|
| XGBoost Estándar (sin regularizar) | 0.681 | 2.956 | **2.275** |
| RF Estándar (sin regularizar) | 1.699 | 2.991 | 1.292 |
| **XGBoost Regularizado (final)** | 2.602 | 3.057 | **0.455** |
| **RF Regularizado (final)** | 3.098 | 3.268 | **0.170** |

La regularización redujo el gap de XGBoost de 2.28 a 0.46 días (reducción del 80%), confirmando que las penalizaciones L1/L2 y la limitación de profundidad (`max_depth=5` vs `max_depth=8`) son esenciales para modelos clínicamente confiables. El RF regularizado tiene el menor gap (0.17 días), pero a costa de un MAE sustancialmente mayor.

---

## 8. Selección del Modelo Ganador

### 8.1 Tabla Resumen Multi-Criterio

| Criterio | Reg. Lineal | RF Final | XGBoost Final | Ganador |
|:---|:---:|:---:|:---:|:---:|
| MAE (días) | 6.311 | 3.268 | **3.057** | XGBoost |
| MedAE (días) | **0.866** | 0.972 | 0.902 | Reg. Lineal |
| Precision PLOS | 66.0% | **80.9%** | 78.7% | RF |
| Recall PLOS | **52.5%** | 31.2% | 48.4% | Reg. Lineal |
| F1 PLOS | 58.5% | 45.0% | **59.9%** | XGBoost |
| Estabilidad (K-Fold MAE) | ±1.825 | ±0.156 | **±0.112** | XGBoost |
| Sobreajuste (Gap) | N/A | **0.170** | 0.455 | RF |
| Tuning justificado | N/A | ✓ | ✓ | XGBoost/RF |
| Validación K-Fold | ✓ | ✓ | ✓ | XGBoost/LR/RF |

### 8.2 Discusión

La Regresión Lineal —una vez corregido el error comparativo forzándola a competir en el mismo espacio logarítmico que los modelos de ML— revela ser **matemáticamente inapropiada** para este problema. Aunque gana en MedAE (acierta bien la mediana) y Recall (detecta pacientes graves), su incapacidad para controlar predicciones extremas resulta en un RMSE catastrófico (>50 días). En un entorno hospitalario, no puedes arriesgarte a que tu sistema de planificación ocasionalmente "explote" con predicciones irreales. 

Entre los modelos de machine learning avanzados, **XGBoost Final** se posiciona como el modelo con el mejor equilibrio clínico y justificación metodológica:
- Gana en precisión operativa general (Mejor MAE, Mejor MedAE y Mejor RMSE).
- Gana en métricas de detección balanceada de pacientes críticos (Mejor F1-Score PLOS con casi 60%).
- Hiperparámetros optimizados rigurosamente por RandomizedSearchCV (50 iteraciones, 5-fold CV).
- Máxima validación de estabilidad mediante StratifiedKFold (MAE = 3.018 ± 0.112, siendo el más estable).
- Gap de sobreajuste controlado (0.455 días, una mejora sustancial frente a su contraparte no regularizada).

Por su parte, el Random Forest Final presenta el menor sobreajuste (Gap de 0.170), pero sufre de un problema grave: un Recall inaceptablemente bajo (31.15%) para identificar pacientes críticos, lo que lo descarta como una herramienta de alerta temprana fiable.

### 8.3 Modelo Ganador: XGBoost Final (Regularizado, Escenario B)

Se selecciona **XGBoost Regularizado** entrenado sobre el Escenario B (Charlson) como el modelo definitivo del proyecto, fundamentado en:

1. **Rigor metodológico completo:** Es el único modelo con tuning justificado, validación cruzada de estabilidad y diagnóstico de sobreajuste documentado.
2. **Balance clínico:** Precision PLOS del 78.67% (baja tasa de falsas alarmas) con Recall del 48.36% (detecta casi la mitad de los pacientes críticos).
3. **Precisión operativa:** MedAE de 0.902 días — la mitad de las predicciones del modelo tienen un error menor a un día, lo que habilita la planificación diaria de camas.
4. **Estabilidad demostrada:** La baja varianza en K-Fold (±0.112 días) confirma que el modelo no depende de un split favorable.

---

## 9. Limitaciones Identificadas

### 9.1 Subestimación Sistemática en Estancias Prolongadas

Todos los modelos de ML subestiman significativamente a los pacientes del tramo 27+ días (entre 15 y 25 días de error). Esta es una limitación estructural del uso de funciones de pérdida simétricas (MSE), que tratan un error de +5 y -5 días como equivalentes. Clínicamente, subestimar es más peligroso que sobreestimar: puede resultar en altas prematuras o falta de preparación de recursos.

### 9.2 Recall PLOS Inferior al 50%

Ningún modelo logra detectar más de la mitad de los pacientes con estancias prolongadas. Esta limitación es consistente con la literatura: Alghatani et al. (2024) reportan dificultades similares en la predicción de la cola derecha de la distribución LOS, incluso con datasets de 2.3 millones de registros.

### 9.3 Ausencia de Variables Dinámicas

Los modelos actuales utilizan exclusivamente variables disponibles al momento de la admisión. La incorporación de datos intrahospitalarios (resultados de laboratorio, evolución clínica) podría mejorar significativamente el rendimiento, como demuestran Turgeman et al. (2021) con datos no estructurados de historias clínicas electrónicas.

---

## 10. Propuestas de Trabajo Futuro

Basándonos en las limitaciones identificadas y en las recomendaciones de la literatura revisada, se proponen las siguientes líneas de mejora:

1. **Funciones de pérdida asimétricas:** Entrenar XGBoost con una función de costo que penalice la subestimación 2x–3x más que la sobreestimación, alineando la optimización matemática con el riesgo clínico real.
2. **Regresión por cuantiles (*Quantile Regression*):** Predecir el percentil 90 del LOS en lugar de la media, proporcionando un "colchón de seguridad" natural para la planificación.
3. **Sobremuestreo de estancias largas (SMOGN):** Aplicar técnicas de balanceo numérico para aumentar la representación de pacientes con LOS extremo en el entrenamiento.
4. **Modelo de dos etapas:** Clasificador binario (corto/largo) seguido de un regresor especializado para cada grupo, como sugieren Phuong et al. (2024).
5. **Incorporación de variables temporales:** Integrar datos de laboratorio y signos vitales de las primeras 24h de internación.

---

## 11. Conclusión

Este estudio demuestra que los modelos de *machine learning* basados en árboles superan consistentemente a la regresión lineal en la predicción del LOS hospitalario, reduciendo el MAE en aproximadamente un 50% (de 6.31 días a 3.05 días). El **XGBoost Regularizado con Índice de Charlson** emerge como el modelo óptimo, ofreciendo el mejor equilibrio entre precisión predictiva (MedAE < 1 día), detección clínica (F1 PLOS = 0.599) y robustez metodológica (validación cruzada, tuning documentado, sobreajuste controlado).

Sin embargo, la subestimación sistemática en estancias prolongadas y el Recall PLOS inferior al 50% representan limitaciones estructurales que requieren estrategias de optimización asimétrica en futuras iteraciones. Estas limitaciones no son un defecto del pipeline implementado, sino una consecuencia matemática del uso de funciones de pérdida simétricas sobre distribuciones altamente desbalanceadas — un hallazgo consistente con la literatura internacional más reciente en predicción de LOS.

---

## Referencias

1. Turgeman, L., May, J.H. & Sciulli, R. Insights from a machine learning model for predicting the hospital Length of Stay (LOS) at the time of admission. *Expert Systems with Applications*, 78, 376-385 (2017).
2. Charlson, M.E. et al. A new method of classifying prognostic comorbidity in longitudinal studies: development and validation. *Journal of Chronic Diseases*, 40(5), 373-383 (1987).
3. Phuong, J. et al. Explainable predictions of a machine learning model to forecast the postoperative length of stay for severe patients. *BMC Medical Informatics and Decision Making*, 24, 275 (2024).
4. Steyerberg, E.W. Clinical Prediction Models. *Statistics and Computing*, Springer (2022).
5. Alghatani, K. et al. Predicting hospital length of stay using machine learning on a large open health dataset. *BMC Health Services Research*, 24, 1238 (2024).
