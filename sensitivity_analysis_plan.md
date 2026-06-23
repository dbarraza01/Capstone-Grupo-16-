# Plan de Implementación — Análisis de Sensibilidad a Posteriori
**Capstone Grupo 16 — Evaluación de Robustez del Pipeline de Predicción de LOS**

---

## 1. Objetivo del Análisis de Sensibilidad

El objetivo de este análisis es responder la pregunta central:

> **¿Qué tan robusto es nuestro pipeline de predicción de LOS frente a cambios en los supuestos, datos y configuraciones que asumimos durante su construcción?**

En un contexto hospitalario real, las condiciones cambian constantemente: la mezcla de pacientes varía entre estaciones, nuevas patologías aparecen, el umbral de "estancia prolongada" puede ajustarse según la capacidad del hospital, y los datos pueden tener calidad variable. Si nuestro modelo solo funciona bien bajo las condiciones exactas del entrenamiento, su utilidad operacional es limitada.

El análisis de sensibilidad nos permite:
1. **Demostrar robustez:** Comprobar que las métricas del modelo no colapsan ante perturbaciones razonables.
2. **Identificar limitaciones honestas:** Documentar bajo qué condiciones el modelo deja de funcionar bien.
3. **Apoyar la toma de decisiones:** Dar al equipo clínico y gerencial confianza sobre cuándo confiar en las predicciones y cuándo ser cautelosos.
4. **Proponer mejoras concretas:** Basándonos en las debilidades encontradas, sugerir líneas de mejora futuras.

> [!IMPORTANT]
> **Criterio de selección de parámetros a variar:** No se varían parámetros arbitrariamente. Cada escenario tiene una justificación operacional o estadística concreta que demuestra robustez frente a situaciones realistas que un hospital enfrentaría. Los escenarios se seleccionan para cubrir las tres dimensiones de incertidumbre: **datos** (volumen y calidad), **supuestos del modelo** (umbrales y transformaciones) y **configuración de evaluación** (métricas y segmentación).

---

## 2. Estructura del Análisis — Los 6 Escenarios

El análisis se organiza en **6 escenarios de sensibilidad**, cada uno variando un supuesto clave del pipeline. Para cada escenario se re-ejecuta el pipeline completo (o la porción afectada) y se comparan las métricas resultantes contra la **línea base** (los resultados actuales del modelo XGBoost con la configuración por defecto).

```text
╔═══════════════════════════════════════════════════════════════════════════════════╗
║  ANÁLISIS DE SENSIBILIDAD — 6 ESCENARIOS                                        ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  LÍNEA BASE (Escenario 0)                                                        ║
║  └── Resultados actuales: XGB con PLOS≥14, stacking, log1p, split 80/20          ║
║                                                                                   ║
║  DIMENSIÓN: SUPUESTOS DEL MODELO                                                 ║
║  ├── Escenario 1: Variación del umbral de PLOS (7, 14, 21, 28 días)             ║
║  └── Escenario 2: Sin transformación log1p del target (regresión directa)        ║
║                                                                                   ║
║  DIMENSIÓN: DATOS                                                                 ║
║  ├── Escenario 3: Reducción del tamaño de entrenamiento (40%, 60%, 80%)          ║
║  └── Escenario 4: Inyección de ruido en variables clínicas (5%, 10%, 20%)        ║
║                                                                                   ║
║  DIMENSIÓN: CONFIGURACIÓN DE EVALUACIÓN                                          ║
║  ├── Escenario 5: Sin segmentación urgente/programado (modelo unificado)         ║
║  └── Escenario 6: Sin stacking (regresor directo sin probabilidad)               ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Descripción Detallada de Cada Escenario

### Escenario 0 — Línea Base (Baseline)

**¿Qué es?** Los resultados actuales del pipeline tal como fue entrenado y evaluado. Sirve como punto de comparación para todos los demás escenarios.

**Valores de referencia (extraídos de los resultados actuales del holdout combinado):**

| Métrica | XGBoost | Random Forest | LR (Baseline) |
|---|---|---|---|
| MAE | 3.56 | 3.80 | 6.68 |
| RMSE | 9.46 | 10.29 | 69.57 |
| MedAE | 1.17 | 1.13 | 0.85 |
| ME | -2.03 | -2.26 | +2.76 |
| PUP | 45.2% | 43.3% | 49.7% |
| MAE Asim. (α=2) | 6.35 | 6.83 | 8.63 |

**No requiere código nuevo.** Se toman directamente los reportes de `ml_operacional/reports/`.

---

### Escenario 1 — Variación del Umbral de PLOS (Prolonged Length of Stay)

**¿Qué supuesto se varía?** El umbral que define cuándo una estancia se considera "prolongada", actualmente fijado en **≥14 días**.

**¿Por qué es relevante?** El umbral de 14 días es una decisión de diseño clínico-operacional, no una constante física. Diferentes hospitales o servicios podrían considerar como estancia prolongada desde los 7 días (en servicios de cirugía ambulatoria o pediátricos) hasta los 28 días (en unidades de cuidados intensivos o rehabilitación) (Stone et al., 2022; Carter & Lapierre, 2001). Si el modelo solo funciona bien con el umbral de 14 días y colapsa con otros umbrales, sería una limitación importante.

**¿Qué se hace?**
1. Se re-entrena todo el pipeline (Etapas 2 y 3) con los umbrales: **7, 14 (base), 21 y 28 días**.
2. Para cada umbral:
   - Se recalcula la etiqueta binaria (`LOS >= umbral`).
   - Se re-ejecuta el tuning del clasificador con los hiperparámetros óptimos ya encontrados (no se hace un nuevo RandomizedSearch; se **re-usan** los `best_params_clf` ya guardados para mantener la comparación justa).
   - Se generan nuevas probabilidades OOF con el nuevo umbral.
   - Se re-entrena el regresor con los `best_params_reg` ya guardados + la nueva columna de probabilidad.
   - Se evalúa en el mismo holdout (20%) con las mismas métricas.
3. Se comparan las métricas de regresión (MAE, RMSE, ME, PUP, MAE Asimétrico) entre los cuatro umbrales.

**Hipótesis esperada:** El MAE debería ser razonablemente estable entre umbrales de 7-21 días, con posible degradación en el umbral de 28 días debido a la escasez de pacientes con estancias tan largas (la clase positiva se vuelve extremadamente minoritaria, lo que dificulta la estimación de la probabilidad).

> [!NOTE]
> **¿Por qué no re-hacer el tuning completo?** Porque el objetivo es aislar el efecto del umbral, no confundirlo con el efecto de tener hiperparámetros diferentes. Si re-tuneáramos, no sabríamos si un cambio en MAE se debe al umbral o a que encontramos hiperparámetros distintos.

**Métricas a comparar:** MAE, RMSE, ME, PUP, MAE Asimétrico (α=2), y especialmente MAE en el tramo 14-26 y 27+ (donde la probabilidad tiene mayor impacto).

---

### Escenario 2 — Sin Transformación log1p del Target

**¿Qué supuesto se varía?** Eliminamos la transformación logarítmica (`log1p`) que se aplica al target (días de estancia) durante el entrenamiento del regresor.

**¿Por qué es relevante?** La transformación `log1p` comprime la distribución asimétrica del LOS (que tiene una "cola larga" de pacientes con estancias extremas). Esta decisión se basa en Manning & Mullahy (2001). Sin embargo, la transformación también puede hacer que el modelo sea más conservador en sus predicciones de estancias largas (porque en escala logarítmica, la diferencia entre 20 y 40 días es mucho menor que en escala original). Este escenario evalúa si la transformación realmente ayuda o si perjudica la predicción de pacientes de estancia prolongada.

**¿Qué se hace?**
1. Se re-entrena el regresor (Etapa 3) **sin** `TransformedTargetRegressor` (regresión directa sobre días reales).
2. Se usan los mismos `best_params_reg` y las mismas probabilidades OOF del Escenario 0.
3. Se evalúa en el mismo holdout.

**Hipótesis esperada:** Sin la transformación, el MAE global podría empeorar levemente (más sensible a outliers), pero el MAE en los tramos de estancias largas (14-26, 27+) podría mejorar al no comprimir las predicciones altas.

**Métricas a comparar:** MAE global, MAE por tramo (especialmente 14-26 y 27+), RMSE (más sensible a errores grandes), ME.

---

### Escenario 3 — Reducción del Tamaño de Entrenamiento (Curva de Aprendizaje)

**¿Qué supuesto se varía?** El volumen de datos de entrenamiento disponible.

**¿Por qué es relevante?** En un hospital real, los datos pueden ser limitados — quizás solo tienen un año de registros, o un hospital más pequeño tiene menos admisiones. Este escenario responde: ¿cuántos datos necesita nuestro modelo para funcionar razonablemente bien? Si el modelo alcanza un buen rendimiento con solo el 40-60% de los datos, es un indicador de robustez y generalizabilidad. Si necesita el 100% y cualquier reducción lo degrada severamente, es una limitación importante (Rajkomar et al., 2018).

**¿Qué se hace?**
1. Del 80% de entrenamiento actual, se toman submuestras estratificadas de **40%, 60% y 80%** (conservando la proporción de urgentes/programados y tramos de LOS).
2. Para cada submuestra:
   - Se re-entrena el pipeline completo (Etapas 2 y 3) con los `best_params` existentes.
   - Se evalúa en el **mismo holdout de siempre** (que no cambia, garantizando comparabilidad).
3. Se construye una **curva de aprendizaje** que muestra cómo las métricas mejoran con más datos.

> [!IMPORTANT]
> **El holdout NUNCA cambia.** Solo se reduce la porción de entrenamiento. Esto asegura que las métricas son comparables entre sí.

**Hipótesis esperada:** Degradación progresiva pero no catastrófica. Si el MAE con 40% de datos es solo un 15-25% peor que con 80%, el modelo es robusto. Si se duplica, es una limitación seria.

**Métricas a comparar:** MAE, RMSE, MAE Asimétrico, y PUP en función del porcentaje de datos de entrenamiento.

---

### Escenario 4 — Inyección de Ruido en Variables Clínicas

**¿Qué supuesto se varía?** La calidad y precisión de los datos de entrada.

**¿Por qué es relevante?** Los datos clínicos reales no son perfectos. Hay errores de codificación ICD-10, diagnósticos faltantes, procedimientos registrados tardíamente, o errores humanos al ingresar datos. Este escenario simula esas imperfecciones de manera controlada para evaluar si nuestro modelo es robusto frente a datos "sucios" (Chrusciel et al., 2021).

**¿Qué se hace?**
1. Se toma el holdout (20%) y se le inyecta ruido a las variables **numéricas continuas** (las variables binarias/one-hot de diagnósticos y procedimientos NO se tocan, ya que un diagnóstico no puede ser "0.85 presente"):
   - **Variables afectadas:** `n_procedimientos`, `n_diag_primarios`, `n_diag_secundarios`, `n_diag_total`, `charlson_index`, `grupos_unicos_diag`, `max_repeticion_diag_grupo`, `n_proc_codigos_repetidos`, `grupos_unicos_proc`, `max_repeticion_proc_grupo`.
   - **Tipo de ruido:** Ruido gaussiano aditivo con desviación estándar proporcional a la desviación estándar original de cada variable.
   - **Niveles:** σ_ruido = 5%, 10% y 20% de la σ original.
2. Se usa el modelo **ya entrenado** (sin re-entrenar) para predecir sobre el holdout ruidoso.
3. Se compara contra la predicción sobre el holdout limpio.

> [!NOTE]
> **¿Por qué solo en el holdout y no en el entrenamiento?** Porque queremos simular la situación real: el modelo fue entrenado con datos históricos de calidad razonable, pero cuando llega un paciente nuevo, sus datos pueden tener errores. Estamos midiendo la **tolerancia a errores de entrada** del modelo ya desplegado.

**Hipótesis esperada:** Con 5% de ruido, las métricas deberían ser prácticamente iguales. Con 20%, podría haber una degradación moderada del MAE (10-20% peor), especialmente si el modelo depende mucho de variables como `charlson_index`.

**Métricas a comparar:** Diferencia absoluta y porcentual en MAE, RMSE, PUP respecto al escenario sin ruido.

---

### Escenario 5 — Sin Segmentación Urgente/Programado (Modelo Unificado)

**¿Qué supuesto se varía?** La decisión de separar a los pacientes en dos subpoblaciones (urgentes vs. programados) y entrenar modelos independientes.

**¿Por qué es relevante?** La segmentación por vía de ingreso es una decisión de diseño basada en la hipótesis de que urgentes y programados tienen dinámicas de estancia fundamentalmente diferentes. Si un modelo unificado (sin separar) funciona igual de bien, la segmentación no aporta valor y simplifica la implementación. Si funciona significativamente peor, la segmentación es una decisión de diseño valiosa y justificada (Carter & Lapierre, 2001; Alsinglawi et al., 2024).

**¿Qué se hace?**
1. Se re-entrena el pipeline completo sobre **todos los pacientes juntos** (sin segmentar por `es_urgencia`).
2. Se usan los mismos `best_params` del XGBoost (se pueden usar los del segmento urgente como punto de partida, ya que no hay segmentación).
3. Se evalúa en el holdout completo y también separando las métricas por urgentes y programados para ver si algún grupo sufre más.

**Hipótesis esperada:** El modelo unificado debería tener un MAE global similar o ligeramente peor, pero con peor rendimiento específico en el grupo de urgentes (donde la distribución del LOS tiene una cola más larga y más variabilidad).

**Métricas a comparar:** MAE global, MAE por segmento (urgente vs. programado por separado), ME por segmento, PUP por segmento.

---

### Escenario 6 — Sin Stacking (Regresor Directo)

**¿Qué supuesto se varía?** Eliminamos la Etapa 2 completa (clasificación de riesgo). El regresor predice los días directamente sin la columna `prob_los_14`.

**¿Por qué es relevante?** El stacking de dos etapas es la innovación central del pipeline. Si un regresor simple (sin probabilidad de riesgo) funciona igual de bien, todo el trabajo de clasificación y OOF es innecesario. Si funciona significativamente peor, especialmente para estancias largas, el stacking queda justificado como una decisión de diseño que resuelve el **sesgo hacia el promedio** (Harini et al., 2022).

**¿Qué se hace?**
1. Se re-entrena el `XGBRegressor` usando los mismos `best_params_reg`, pero **sin la columna `prob_los_14`** (solo variables clínicas originales).
2. Se evalúa en el mismo holdout.

**Hipótesis esperada:** El MAE global podría ser similar (porque la mayoría de pacientes tienen estancias cortas y el promedio funciona bien para ellos). Pero el **MAE en los tramos 14-26 y 27+** debería empeorar notablemente, y el **ME** debería ser más negativo (mayor subestimación sistemática de estancias largas). Esto es lo que el stacking fue diseñado para resolver.

**Métricas a comparar:** MAE global, MAE por tramo (especialmente 14-26 y 27+), ME, PUP, MAE Asimétrico.

---

## 4. Estructura del Directorio y Archivos

```text
ml_operacional/
├── sensitivity/                           ← Nueva carpeta principal del análisis
│   ├── run_sensitivity.py                 ← Script maestro que ejecuta todos los escenarios
│   │                                        y genera el reporte consolidado
│   │
│   ├── escenario_1_umbrales.py            ← Variación del umbral PLOS (7, 14, 21, 28)
│   ├── escenario_2_sin_log1p.py           ← Sin transformación logarítmica del target
│   ├── escenario_3_curva_aprendizaje.py   ← Reducción del tamaño de entrenamiento
│   ├── escenario_4_ruido.py               ← Inyección de ruido en variables numéricas
│   ├── escenario_5_sin_segmentacion.py    ← Modelo unificado (sin split urgente/programado)
│   ├── escenario_6_sin_stacking.py        ← Regresor directo sin probabilidad de riesgo
│   │
│   └── results/                           ← Outputs de cada escenario
│       ├── escenario_1_resultados.csv
│       ├── escenario_2_resultados.csv
│       ├── escenario_3_resultados.csv
│       ├── escenario_4_resultados.csv
│       ├── escenario_5_resultados.csv
│       ├── escenario_6_resultados.csv
│       └── reporte_sensibilidad_consolidado.md  ← Reporte final con tablas comparativas
```

---

## 5. Salidas y Outputs Esperados

### 5.1 Por Cada Escenario Individual (`escenario_X_*.py`)

**Mensajes en Consola (Stdout):**
- Log de inicio indicando el nombre del escenario y los parámetros que se varían.
- Para cada variante dentro del escenario, imprimir:
  - La configuración específica (ej. "Umbral PLOS = 7 días" o "Ruido = 10%").
  - Las métricas clave: MAE, ME, PUP, MAE Asimétrico.
- Al final, imprimir una tabla resumen comparando todas las variantes del escenario.

**Archivos Guardados en Disco (`sensitivity/results/`):**
- `escenario_X_resultados.csv`: Tabla con una fila por variante y columnas para todas las métricas.
- Para escenarios con métricas por tramo (1, 2, 5, 6): archivo adicional `escenario_X_resultados_por_tramo.csv`.

### 5.2 Script Maestro (`run_sensitivity.py`)

**Función:** Ejecuta secuencialmente todos los 6 escenarios (importando cada módulo). Al final, genera el reporte consolidado.

**Mensajes en Consola:**
- Progreso general: "Ejecutando Escenario 1/6...", "Escenario 1 completado.", etc.
- Al final: resumen ejecutivo con las conclusiones principales.

**Reporte Consolidado (`reporte_sensibilidad_consolidado.md`):**
Este es el entregable principal del análisis. Contendrá:

1. **Tabla resumen ejecutiva:** Una tabla con todos los escenarios y su impacto en las métricas clave.
2. **Para cada escenario:**
   - Descripción del supuesto variado.
   - Tabla de resultados.
   - Interpretación de los resultados (¿el modelo es robusto o no frente a este cambio?).
3. **Sección de Limitaciones:** Listado explícito de las limitaciones identificadas con el análisis.
4. **Sección de Mejoras Propuestas:** Basadas en los hallazgos, ¿qué se podría mejorar en el futuro?
5. **Sección de Apoyo a la Toma de Decisiones:** ¿Cómo ayudan estos resultados a la gestión hospitalaria?

---

## 6. Detalle de Implementación — Cómo se Re-usa el Código Existente

> [!TIP]
> **No se re-inventa la rueda.** Cada escenario re-usa las funciones existentes en `ml_operacional/utils/pipeline_operacional.py` y `metricas_operacionales.py`. Solo se modifican los parámetros de entrada.

| Escenario | Funciones reutilizadas | ¿Qué cambia? |
|---|---|---|
| 1 (Umbrales) | `binary_target_los14` → se crea una versión parametrizable `binary_target(y, threshold)`, `generate_oof_probabilities`, `make_regressor`, `calcular_metricas_globales` | El umbral de clasificación (7, 21, 28) |
| 2 (Sin log1p) | `make_regressor` → se modifica para NO envolver con `TransformedTargetRegressor` | La transformación del target |
| 3 (Curva de aprendizaje) | `prepare_xy`, `generate_oof_probabilities`, `make_classifier`, `make_regressor`, `calcular_metricas_globales` | El tamaño del dataset de entrenamiento (submuestra) |
| 4 (Ruido) | Solo `calcular_metricas_globales` y los modelos guardados en `.joblib` | Se inyecta ruido al holdout antes de predecir |
| 5 (Sin segmentación) | Todo el pipeline pero sin filtrar por `es_urgencia` | Se entrena con todos los pacientes juntos |
| 6 (Sin stacking) | `make_regressor`, `prepare_xy(include_prob=False)`, `calcular_metricas_globales` | Se excluye la columna `prob_los_14` |

---

## 7. Consideraciones Importantes para la Implementación

> [!WARNING]
> **Tiempo de ejecución:** Los escenarios 1, 3 y 5 requieren re-entrenamiento completo del pipeline. Dependiendo de la máquina, cada uno puede tardar entre 5-15 minutos. El Escenario 4 es el más rápido (solo inferencia). Se recomienda ejecutar `run_sensitivity.py` y dejar que corra secuencialmente.

### 7.1 Sobre el Random State
Todos los escenarios usarán `RANDOM_STATE = 42` (el mismo del pipeline original) para garantizar reproducibilidad. Las submuestras del Escenario 3 usarán `random_state=42` en `train_test_split` para ser determinísticas.

### 7.2 Sobre Qué Modelo se Usa como Base para el Análisis
El análisis de sensibilidad se ejecutará **únicamente sobre XGBoost**, que es el modelo ganador según la comparación final. No se repite para Random Forest ni LR, porque el objetivo no es comparar modelos (eso ya se hizo), sino evaluar la robustez de la solución elegida.

### 7.3 Sobre la Interpretación
Para cada escenario, se calcula la **variación porcentual** del MAE respecto a la línea base:

$$\Delta\text{MAE}\% = \frac{\text{MAE}_{\text{escenario}} - \text{MAE}_{\text{base}}}{\text{MAE}_{\text{base}}} \times 100$$

Criterios de interpretación:
- **|ΔMAE%| < 5%:** El modelo es **robusto** frente a este cambio.
- **5% ≤ |ΔMAE%| < 15%:** Sensibilidad **moderada**. Documentar como limitación menor.
- **|ΔMAE%| ≥ 15%:** Sensibilidad **alta**. Documentar como limitación importante y proponer mejora.

---

## 8. Sección de Limitaciones y Mejoras (Estructura Anticipada)

El reporte final incluirá estas secciones basadas en los resultados:

### 8.1 Limitaciones Identificadas
*(Se completarán con los resultados reales, pero anticipamos las siguientes categorías:)*

1. **Dependencia del volumen de datos:** Si el Escenario 3 muestra degradación severa con 40% de datos, documentar el tamaño mínimo de dataset necesario.
2. **Sesgo de subestimación en estancias largas:** Ya observado en los resultados actuales (tramo 27+: MAE=30.16, PUP=100%). Limitación estructural del enfoque.
3. **Sensibilidad al umbral de PLOS:** Si el Escenario 1 muestra variación significativa entre umbrales.
4. **Calidad de datos de entrada:** Si el Escenario 4 muestra degradación con 10-20% de ruido.
5. **Generalización temporal:** Limitación no evaluable con los datos actuales (datos de un solo período).

### 8.2 Mejoras Propuestas
*(Basadas en las limitaciones encontradas:)*

1. **Modelo especializado para estancias extremas (27+ días):** Crear un tercer modelo que solo se active cuando la probabilidad de PLOS es extremadamente alta (>0.90).
2. **Validación temporal (Rolling window):** Implementar validación por ventanas temporales para simular el paso del tiempo.
3. **Re-entrenamiento periódico:** Protocolo para re-entrenar el modelo cada 6-12 meses con datos nuevos.
4. **Integración de datos no estructurados:** Notas clínicas de texto libre como features adicionales (Chrusciel et al., 2021).
5. **Ensemble de umbrales:** Usar múltiples clasificadores con diferentes umbrales y promediar las probabilidades.

### 8.3 Apoyo a la Toma de Decisiones
*(Estructura anticipada del análisis:)*

1. **¿Cuándo confiar en la predicción?** Rango de LOS predicho donde la confiabilidad es alta (tramos 0-2 y 3-6).
2. **¿Cuándo ser cauteloso?** Predicciones de estancias > 14 días tienen mayor incertidumbre.
3. **Uso operacional recomendado:** El modelo como herramienta de **priorización** (no como predicción exacta) para la gestión de camas.

---

## 9. Plan de Verificación

1. **Consistencia de la línea base:** Verificar que el Escenario 0 reproduce exactamente los resultados ya reportados en `ml_operacional/reports/`.
2. **Reproducibilidad:** Ejecutar `run_sensitivity.py` dos veces y confirmar que los resultados son idénticos (mismo random_state).
3. **Coherencia lógica:** El Escenario 6 (sin stacking) debe tener peor MAE en tramos altos pero similar MAE global (si no, hay un error).
4. **Formato de salida:** Verificar que todos los CSV se generan correctamente y que el reporte `.md` es legible.

---

## 10. Prompt para Codex (Instrucciones de Implementación)

> [!IMPORTANT]
> **Este prompt se le entregará a Codex para que implemente todo el código.**

```
CONTEXTO DEL PROYECTO:
- Proyecto Capstone de predicción de estancia hospitalaria (LOS).
- Pipeline en dos etapas: Clasificador (XGBClassifier) → Regresor (XGBRegressor) con stacking.
- Segmentación por urgente/programado.
- PLOS ≥ 14 días. Umbral actual: 14 días.
- Código existente en ml_operacional/ con utils compartidas.
- Modelo ganador: XGBoost.

TAREA:
Implementar un análisis de sensibilidad completo en ml_operacional/sensitivity/.
Seguir EXACTAMENTE la estructura descrita en el plan de implementación 
(sensitivity_analysis_plan.md) que se adjunta.

REGLAS:
1. Re-usar las funciones de ml_operacional/utils/pipeline_operacional.py y 
   metricas_operacionales.py. No duplicar código.
2. Cada escenario debe ser un script independiente que se pueda ejecutar por separado.
3. run_sensitivity.py debe ejecutar todos los escenarios secuencialmente.
4. Todos los escenarios se ejecutan solo sobre XGBoost (el modelo ganador).
5. Usar los best_params ya guardados en ml_operacional/XGB/*.json. No re-tunear.
6. El holdout NUNCA cambia. Solo se modifica el entrenamiento o los datos de entrada.
7. Cada script debe imprimir un log claro en consola con las métricas resultantes.
8. Generar archivos .csv en sensitivity/results/ con los resultados de cada escenario.
9. Al final, generar reporte_sensibilidad_consolidado.md con tablas comparativas,
   interpretación, limitaciones, mejoras propuestas y apoyo a la toma de decisiones.
10. RANDOM_STATE = 42 en todo momento.

ESCENARIOS A IMPLEMENTAR:
1. Variación del umbral PLOS: 7, 14, 21, 28 días.
2. Sin transformación log1p del target.
3. Curva de aprendizaje: submuestras de 40%, 60%, 80% del entrenamiento.
4. Inyección de ruido gaussiano: 5%, 10%, 20% de sigma en variables numéricas continuas.
5. Sin segmentación urgente/programado (modelo unificado).
6. Sin stacking (regresor directo sin prob_los_14).

DATOS:
- Dataset principal: ml/feature_engineering/processed_v3/model_data_v3_escenario_B_charlson.csv
- Splits ya creados en: ml_operacional/data_splits/
- Modelos guardados en: ml_operacional/modelos_guardados/
- Hiperparámetros en: ml_operacional/XGB/best_params_*.json
- Target: los_dias. ID: case_id. Urgencia: es_urgencia.

VARIABLES NUMÉRICAS PARA ESCENARIO 4 (RUIDO):
n_procedimientos, n_diag_primarios, n_diag_secundarios, n_diag_total, 
charlson_index, grupos_unicos_diag, max_repeticion_diag_grupo, 
n_diag_codigos_repetidos, grupos_unicos_proc, max_repeticion_proc_grupo

CRITERIO DE INTERPRETACIÓN:
- |ΔMAE%| < 5%: Robusto
- 5% ≤ |ΔMAE%| < 15%: Sensibilidad moderada
- |ΔMAE%| ≥ 15%: Sensibilidad alta
```

---

## 11. Referencias (APA 7)

*   **Alsinglawi, B. S., et al.** (2024). Predicting hospital stay length using explainable machine learning. *IEEE Access*, 12, 90571–90585. https://doi.org/10.1109/ACCESS.2024.3421295
*   **Bergstra, J., & Bengio, Y.** (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(2), 281–305.
*   **Carter, M. W., & Lapierre, S. D.** (2001). Scheduling emergency and elective patients in a hospital. *Health Care Management Science*, 4(2), 141-151.
*   **Chrusciel, J., et al.** (2021). The prediction of hospital length of stay using unstructured data. *BMC Medical Informatics and Decision Making*, 21(1), 351.
*   **Harini, A., Sivaraman, R., & Sundarraj, R. P.** (2022). Two-stage machine learning and stochastic optimization models for hospital bed management. *European Journal of Operational Research*, 301(2), 654-672.
*   **Manning, W. G., & Mullahy, J.** (2001). Estimating log models and transition models for nonnegative outcomes. *Journal of Health Economics*, 20(4), 461-494.
*   **Rajkomar, A., et al.** (2018). Scalable and accurate deep learning with electronic health records. *npj Digital Medicine*, 1(1), 18.
*   **Song, H., Tucker, A. L., & Murrell, K. L.** (2015). The effects of hospital capacity constraints on admitting decisions and patient outcomes. *Journal of Health Economics*, 40, 109-122.
*   **Stone, K., Zwiggelaar, R., Jones, P., & Mac Parthaláin, N.** (2022). A systematic review of the prediction of hospital length of stay: Towards a unified framework. *PLOS Digital Health*, 1(4), e0000017.
