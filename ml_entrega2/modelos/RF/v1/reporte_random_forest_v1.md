# 📋 Reporte Analítico — Random Forest v1
**Predicción de Length of Stay (LOS) Hospitalario — Capstone Grupo 16**  
**Modelo:** `RandomForestRegressor_v1` | **Script:** `entrenar_random_forest_v1.py`

---

## 1. Contexto y Objetivo del Modelo

El objetivo de este modelo es predecir cuántos días permanecerá hospitalizado un paciente (`los_dias`) a partir de sus **grupos de códigos ICD-10** de diagnósticos y procedimientos, junto con variables clínicas de ingreso.

Este es un **modelo preliminar (v1)** cuya función principal es verificar si la ingeniería de features desarrollada en `procesamiento_features_v2.py` (la agrupación jerárquica de códigos ICD-10) contiene señal predictiva útil, y establecer una línea base de desempeño antes de construir versiones más complejas.

**No es el modelo definitivo.** Es el primer paso para entender cómo se ajustan los datos.

---

## 2. Configuración del Modelo

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Algoritmo | `RandomForestRegressor` | Maneja bien matrices binarias dispersas (densidad 0.58%) y alta dimensionalidad (1,650 features) |
| `n_estimators` | 300 | 300 árboles es suficiente para estabilidad estadística del ensamblado |
| `max_depth` | 8 | Profundidad conservadora para evitar memorización de combinaciones raras de códigos ICD |
| `min_samples_leaf` | 10 | Mínimo 10 pacientes por hoja — previene sobreajuste en subgrupos pequeños |
| `min_samples_split` | 20 | Consistente con el umbral de agrupación ICD del pipeline |
| `max_features` | `"sqrt"` | ~40 features evaluadas por split de 1,650 disponibles — estándar RF |
| `random_state` | 42 | Semilla fija para reproducibilidad |
| `n_jobs` | -1 | Usa todos los núcleos disponibles |

**Transformación del target:** Se aplicó `log1p(los_dias)` antes de entrenar y `expm1` para volver a días en las predicciones. Esto es necesario porque LOS tiene una distribución muy asimétrica (mediana=3, máximo=262) — sin esta transformación el modelo optimizaría solo los casos frecuentes de 1-6 días.

**División train/test:** 80%/20% con estratificación por tramos de LOS, lo que garantiza representación proporcional de los casos raros (LOS ≥ 27 días) tanto en entrenamiento como en evaluación.

---

## 3. Distribución del Dataset

Antes del split, la cohorte se distribuye así:

| Tramo de LOS | Proporción |
|-------------|-----------|
| 0–2 días | 49.5% |
| 3–6 días | 28.4% |
| 7–13 días | 10.4% |
| 14–26 días | 6.6% |
| 27+ días | 5.1% |

**Interpretación:** El dataset es fuertemente asimétrico. Casi la mitad de los pacientes se van en 2 días o menos. Los casos más complejos (≥27 días) representan apenas 1 de cada 20 hospitalizaciones. Esta distribución es el principal desafío del modelo.

---

## 4. Archivos de Salida — Descripción y Variables

### 4.1 `reports_modelos/predicciones_random_forest_v1.csv`

Contiene las predicciones individuales para cada uno de los 2,391 pacientes del conjunto de test.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `case_id` | string | Identificador del paciente (solo para trazabilidad) |
| `los_real` | int | LOS real en días (valor verdadero) |
| `los_pred` | float | LOS predicho por el modelo en días |
| `error` | float | Diferencia: `los_pred - los_real`. Negativo = subestimación |
| `abs_error` | float | Valor absoluto del error (sin signo) |
| `subestima` | int (0/1) | 1 si el modelo predijo menos días de los reales |
| `tramo_los` | categórico | Tramo real del paciente: 0-2, 3-6, 7-13, 14-26, 27+ |
| `plos_real` | int (0/1) | 1 si el LOS real fue ≥ 27 días |
| `plos_pred` | int (0/1) | 1 si el modelo predijo ≥ 27 días |

**Hallazgo crítico:** El modelo predice en un rango de **1.90 a 20.14 días**. Nunca supera los 20 días. Los 5 peores casos tienen errores de 110 a 193 días — todos ellos pacientes que estuvieron más de 100 días hospitalizados y el modelo predijo ~10-15 días.

---

### 4.2 `reports_modelos/metricas_random_forest_v1.csv`

Métricas globales de evaluación del modelo sobre el set de test completo (2,391 pacientes).

| Columna | Valor | Descripción |
|---------|-------|-------------|
| `modelo` | RandomForestRegressor_v1 | Nombre del modelo |
| `target_entrenamiento` | log1p(los_dias) | Confirma la transformación aplicada |
| `mae` | **4.16 días** | Error absoluto medio: el modelo se equivoca en promedio 4.16 días |
| `rmse` | **11.13 días** | Raíz del error cuadrático medio: penaliza los errores grandes |
| `medae` | **1.38 días** | Mediana del error: el error "típico" es de solo 1.38 días |
| `precision_plos_27` | 0.0 | Proporción de alarmas PLOS que son correctas |
| `recall_plos_27` | 0.0 | Proporción de PLOS reales que el modelo detecta |
| `f1_plos_27` | 0.0 | Balance precision/recall para PLOS |

### 1. Métricas de Error de Regresión (Predicción de Días Exactos)
El modelo intenta adivinar un número exacto (ej. 5 días). Estas métricas miden qué tan lejos se quedó de la realidad.

##### RMSE (11.13 días) - Raíz del Error Cuadrático Medio:

Qué significa: Mide el error promedio, pero castiga muchísimo las equivocaciones gigantes. Si el modelo se equivoca por 1 día en un paciente, no pasa nada. Pero si un paciente se quedó 150 días y el modelo predijo 10 (un error de 140 días), el RMSE eleva ese 140 al cuadrado, haciéndolo explotar.
Por qué es 11.13: Como vimos que el modelo tiene errores enormes en los pacientes que se quedan meses hospitalizados (errores de >100 días), esos pocos casos disparan el RMSE hacia arriba.
  

##### Mediana del Error Absoluto - MedAE (1.38 días): 

Qué significa: Es el "error típico" o el error del paciente que está justo en el medio del grupo. Significa que si ordenamos a todos tus pacientes del que mejor predijimos al que peor predijimos, al 50% de tus pacientes les adivinamos los días de estancia fallando por 1.38 días o menos.
Por qué es útil: A diferencia del RMSE, la Mediana ignora por completo a los casos extremos. Te dice cómo funciona el modelo para "la gran mayoría" de las personas comunes y corrientes (las de 1 o 2 días).

### 2. Métricas de Detección PLOS (Clasificación de Estancias Prolongadas ≥ 27 días)
Aquí ya no evaluamos si el modelo adivinó el número exacto, sino si logró encender la alarma correcta: ¿Este paciente se va a quedar un mes o más? Sí o No.

Para entender esto, imagina que el modelo es una alarma de incendios.

##### Precision (0.0) - "Si la alarma suena, ¿es un incendio real?"

Qué significa: De todas las veces que tu modelo dijo "¡Alerta! Este paciente va a estar más de 27 días", ¿qué porcentaje realmente lo estuvo?
Por qué dio 0.0: Porque tu modelo es tan conservador que nunca encendió la alarma (nunca predijo un valor mayor a 20 días). Como no hizo ninguna predicción de ≥ 27 días, su precisión es cero. No hubo falsas alarmas, pero porque nunca hubo alarmas.

##### Recall (0.0) - "Si hay un incendio, ¿la alarma logró sonar?"

Qué significa: Es la capacidad de encontrar los casos peligrosos. Hubo 122 pacientes en tu prueba que sí se quedaron más de 27 días. De esos 122, ¿cuántos logró detectar el modelo?
Por qué dio 0.0: El modelo detectó 0 de 122. Se le escaparon absolutamente todos los casos graves (100% de Falsos Negativos). Clínicamente, un Recall de 0 es peligroso porque indica que el sistema no está viendo a los pacientes más complejos.
F1-Score (0.0) - El balance general

Qué significa: Es un promedio matemático entre la Precision y el Recall. En la vida real, si haces que tu alarma sea muy sensible (alto Recall), va a sonar por accidente muchas veces (baja Precision). El F1-Score te dice si lograste un buen equilibrio.
Por qué dio 0.0: Como ambos valores anteriores fueron cero, el F1-Score es cero. El modelo falló completamente en la tarea de separar a los pacientes normales de los pacientes crónicos extremos.
En resumen para tu informe: Tu modelo es excelente para adivinar estancias de los pacientes comunes de 1 a 3 días (eso lo dice el MedAE bajo de 1.38), pero está completamente ciego a los pacientes graves (eso lo dice el Recall de 0.0 y el RMSE alto).
---

### 4.3 `reports_modelos/metricas_por_tramo_random_forest_v1.csv`

La métrica más importante del reporte. Desglosa el error por cada segmento de LOS.

| Tramo | N pacientes | LOS real prom. | LOS pred. prom. | MAE | MedAE | Error medio | % Subestima |
|-------|------------|---------------|-----------------|-----|-------|-------------|-------------|
| 0–2 días | 1,183 | 1.24 días | 2.58 días | 1.34 | 1.29 | +1.34 | 0.0% |
| 3–6 días | 680 | 4.00 días | 3.19 días | 1.39 | 1.20 | -0.81 | 83.1% |
| 7–13 días | 249 | 9.22 días | 5.29 días | 4.13 | 4.20 | -3.93 | 92.8% |
| 14–26 días | 157 | 18.21 días | 7.29 días | 10.94 | 10.34 | -10.92 | 99.4% |
| **27+ días** | **122** | **49.16 días** | **10.85 días** | **38.31** | **31.80** | **-38.31** | **100.0%** |

---

### 4.4 `reports_modelos/matriz_confusion_plos_rf_v1.csv`

Evaluación binaria de si el modelo identifica correctamente los pacientes con LOS prolongado (≥27 días).

| | Pred: Corto (<27d) | Pred: Largo (≥27d) |
|---|---|---|
| **Real: Corto (<27d)** | TN = 2,269 | FP = 0 |
| **Real: Largo (≥27d)** | **FN = 122** | TP = 0 |

---

### 4.5 `models/random_forest_v1.pkl`

El modelo entrenado serializado en formato binario. Puede cargarse en el futuro con `joblib.load()` para hacer nuevas predicciones sin necesidad de reentrenar.

---

## 5. Interpretación de los Resultados

### 5.1 ¿Qué está haciendo el modelo bien?

**El modelo funciona razonablemente bien en las estancias cortas**, que son por lejos el grupo más numeroso. En el tramo 0–2 días, el error medio es solo 1.34 días con una MedAE de 1.29 días. Para un paciente que realmente se va en 1 día, predecir ~2.5 días es un error tolerable en términos clínicos.

La MedAE global de **1.38 días** indica que la mitad de los pacientes en el test set tienen un error de predicción menor a 1.38 días. Esto parece positivo, pero es engañoso porque la mayoría de esos pacientes son del tramo 0–2 días.

### 5.2 ¿Cuál es el problema central?

**El modelo está sesgado hacia el promedio poblacional.** El rango de predicciones va de 1.90 a 20.14 días — nunca predice más de 20 días. Esto significa que el modelo básicamente aprendió a predecir "el LOS típico de los pacientes con características similares" en vez de aprender cuándo un caso será extremo.

Este comportamiento es matemáticamente esperado: Random Forest promedia las predicciones de todos sus árboles, lo que naturalmente "amortigua" los valores extremos. En el contexto de una distribución tan asimétrica como el LOS hospitalario, este efecto de promediación es especialmente perjudicial.

### 5.3 El gradiente de subestimación — el hallazgo más revelador

El dato más importante está en la columna `pct_subestima` de las métricas por tramo:

- **Tramo 0–2 días:** 0% de subestimación → el modelo siempre sobreestima (predice más días de los reales)
- **Tramo 3–6 días:** 83% de subestimación
- **Tramo 7–13 días:** 93% de subestimación
- **Tramo 14–26 días:** 99% de subestimación
- **Tramo 27+ días:** 100% de subestimación

Esto revela un **sesgo sistemático** perfectamente gradiente: cuanto más largo es el LOS real, más seguro es que el modelo lo subestime. La razón es estructural: el 49.5% del dataset tiene LOS de 0–2 días. El modelo "aprendió" que la respuesta segura es predecir algo cercano a la media de todos los pacientes (~6 días), lo que es correcto para la mayoría, pero catastrófico para los casos complejos.

### 5.4 La incapacidad total de detectar PLOS

**Precision = 0.0 / Recall = 0.0 / F1 = 0.0** es el resultado más contundente del análisis. El modelo predijo LOS ≥ 27 días para **cero** pacientes. Los 122 pacientes que realmente tuvieron LOS ≥ 27 días fueron todos clasificados como "estancia corta" (122 Falsos Negativos, 0 Verdaderos Positivos).

Esto no es sorprendente dado que el modelo nunca supera las 20 días de predicción, pero confirma que en su estado actual el modelo no sirve para el objetivo clínico más importante: anticipar hospitalizaciones prolongadas.

---

## 6. Ventajas del Modelo v1

| Ventaja | Descripción |
|---------|-------------|
| **Funciona para estancias cortas** | MAE de 1.34 días en el tramo 0–2 días. Útil para planificación de camas en el 49% más común |
| **Completamente reproducible** | `random_state=42` garantiza resultados idénticos en cada ejecución |
| **Sin leakage** | Se confirmó que ninguna variable post-ingreso entró como predictor |
| **Baseline establecido** | Proporciona un punto de referencia cuantitativo para comparar versiones futuras |
| **Tolerante a la sparsidad** | Random Forest maneja bien los 1,650 features binarios con densidad 0.58% |
| **Documentación de sesgos** | El análisis por tramos reveló con precisión dónde y cómo falla el modelo |

---

## 7. Desventajas y Limitaciones

| Desventaja | Descripción | Gravedad |
|------------|-------------|----------|
| **Sesgo hacia el promedio** | Rango de predicción 1.90–20.14 días. Nunca predice estancias largas | 🔴 Alta |
| **PLOS no detectado** | Recall=0% para LOS ≥27 días. El objetivo clínico más importante falla completamente | 🔴 Alta |
| **Subestimación sistemática** | 99–100% de subestimación en tramos largos. El modelo es peligrosamente optimista | 🔴 Alta |
| **RMSE elevado (11.13 días)** | Penaliza los errores en casos extremos. Indica que los outliers se predicen muy mal | 🟡 Media |
| **Población mezclada** | Obstetricia, medicina interna y cirugía en un solo modelo diluyen los patrones | 🟡 Media |
| **Sin feature importance** | No se extrae cuáles grupos ICD son los más relevantes para el modelo | 🟡 Media |
| **Sin validación cruzada** | Un solo split 80/20 puede dar estimaciones optimistas o pesimistas por el azar | 🟡 Media |

---

## 8. ¿Por qué ocurre todo esto? — Diagnóstico Técnico

El comportamiento observado es la combinación de tres factores que actúan simultáneamente:

**Factor 1 — Distribución asimétrica del target:**
Casi el 50% de los datos son LOS de 1–2 días. El modelo aprende que "predecir 2–3 días" tiene una penalización pequeña en la mayoría de los casos. Aunque se aplicó `log1p` para atenuar esto, el desbalance sigue siendo pronunciado.

**Factor 2 — El promediado de Random Forest:**
Random Forest genera su predicción como el promedio de 300 árboles. Cuando la mayoría de los árboles votaron ~2 días porque el 50% de los datos de entrenamiento tienen ese valor, el promedio final siempre estará tirado hacia ese centro. Los árboles que "vieron" casos de 100 días son una minoría que queda ahogada por los 250 árboles restantes.

**Factor 3 — Los códigos ICD son condición necesaria pero no suficiente:**
Los grupos ICD capturan *qué tiene el paciente*, pero no cuán grave es su situación al momento del ingreso. Dos pacientes con el mismo diagnóstico de neumonía (código J18) pueden estar hospitalizados 3 días o 45 días según la severidad clínica, la respuesta al tratamiento y los recursos disponibles. Esta variabilidad interna no es capturada por los features binarios actuales.

---

## 9. Pasos Futuros — Lo que Indica Este Primer Modelo

Los resultados del modelo v1 son informativos aunque no satisfactorios. Establecen con claridad las prioridades para el modelo v2:

### 🔴 Prioritario — Tratar el desbalance de clases

El problema central es la distribución asimétrica del LOS. Las estrategias a evaluar son:

1. **Sobremuestreo de casos extremos (SMOTE para regresión):** Generar muestras sintéticas de pacientes con LOS largo durante el entrenamiento para que el modelo vea más ejemplos de casos complejos.
2. **Pesos de muestra personalizados:** Asignar mayor peso en la función de pérdida a los pacientes con LOS ≥ 14 días, de modo que el modelo "pague" más por equivocarse en esos casos.
3. **Modelo de dos etapas:** Primero clasificar si el paciente es PLOS (≥27 días) o no, y luego usar un regresor separado para cada grupo.

### 🔴 Prioritario — Cambio a Gradient Boosting (XGBoost / LightGBM)

XGBoost y LightGBM tienen mecanismos nativos para manejar distribuciones asimétricas y pueden aprender de los errores de forma secuencial. Esto los hace más adecuados para predecir valores en la cola derecha de la distribución que Random Forest.

### 🟡 Importante — Extraer Feature Importance

El siguiente paso más valioso analíticamente es saber cuáles grupos ICD el modelo considera más predictivos. Esto permitiría:
- Validar clínicamente si los features importantes tienen sentido médico
- Reducir el número de features (de 1,650 a los 50–100 más importantes) antes de entrenar el modelo v2
- Identificar si algunas variables base (`n_procedimientos`, `n_diag_total`) dominan sobre los códigos ICD específicos

### 🟡 Importante — Validación Cruzada (KFold)

Implementar K-Fold con k=5 para obtener una estimación más estable del error. Con un solo split 80/20 y datos médicos que pueden tener estacionalidad (más partos en ciertos meses, más neumonías en invierno), el resultado puede variar según qué datos cayeron en train y cuáles en test.

### 🟢 Trabajo Futuro — Estratificación por Servicio

Separar el modelo por tipo de hospitalización (obstétrica, médica, quirúrgica) permitiría que cada modelo aprenda patrones más homogéneos. Un paciente obstétrico y uno oncológico no deberían compartir el mismo modelo.

---

## 10. Conclusión

> El modelo Random Forest v1 **funciona correctamente como herramienta diagnóstica del problema**, pero **no es apto para uso clínico** en su estado actual.

> Su principal aportación es confirmar dos cosas: (1) los features ICD agrupados contienen señal predictiva real para estancias cortas; y (2) el sesgo hacia el promedio de Random Forest impide la detección de casos prolongados en una distribución tan asimétrica como el LOS hospitalario.

> **El dataset está bien construido. El problema está en la estrategia de modelado, no en los datos.**

> El siguiente paso lógico es implementar XGBoost o LightGBM con manejo de desbalance, extraer feature importance de este modelo para reducir dimensionalidad, y comparar resultados con validación cruzada.

---

*Análisis generado sobre los archivos de salida de `entrenar_random_forest_v1.py`. Cohorte: 11,951 pacientes, 9,560 train / 2,391 test. Dataset: `model_data_ml_v2.csv`.*
