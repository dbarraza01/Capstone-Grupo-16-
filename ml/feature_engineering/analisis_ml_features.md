# 📊 Análisis de Features de Machine Learning — Capstone Grupo 16
**LOS (Length of Stay) Prediction — Ingeniería de Features Clínicas**

---

## 1. Descripción General del Repositorio `ml/`

La carpeta `ml/` contiene la **pipeline completa de ingeniería de features** para construir el dataset final que alimentará los modelos de predicción de Estancia Hospitalaria (LOS). El flujo va desde datos clínicos brutos (diagnósticos y procedimientos en códigos ICD-10-CM / ICD-10-PCS) hasta una matriz binaria lista para Random Forest y Gradient Boosting.

### Estructura del directorio

```text
ml/
├── procesamiento_features_v2.py               ← Script de ingeniería de features
├── analisis_ml_features.md                    ← Este documento
├── processed/
│   ├── features_diagnosticos_agrupados_v2.csv ← Matriz binaria diagnósticos
│   ├── features_procedimientos_agrupados_v2.csv ← Matriz binaria procedimientos
│   └── model_data_ml_v2.csv                   ← Dataset final (LISTO PARA USAR)
└── reports/
    ├── mapeo_reemplazos_diagnosticos_v2.csv        ← Trazabilidad diagnósticos
    ├── mapeo_reemplazos_procedimientos_v2.csv      ← Trazabilidad procedimientos
    ├── reporte_frecuencias_diagnosticos_v2.csv     ← Frecuencias de diagnósticos
    ├── reporte_frecuencias_procedimientos_v2.csv   ← Frecuencias de procedimientos
    ├── reporte_repeticiones_codigos_v2.csv         ← Features de repetición
    ├── reporte_target_los_v2.csv                   ← Estadísticas del target (LOS)
    └── reporte_codigos_outliers_v2.csv             ← Códigos en pacientes outliers LOS
```

---

## 2. Qué Cambió de v1 a v2

| # | Aspecto | v1 | v2 |
|---|---------|----|----|
| 1 | **`UUUUUU`** | Incluido como feature normal | **Excluido** de la matriz predictiva. No es un diagnóstico clínico; representa urgencias sin código asignado |
| 2 | **Fallback diagnósticos** | Capítulo directo (`diag_E`, `diag_I`) | Prefijo `diag_rare_cap_E`, `diag_rare_cap_I` para distinguir de categorías reales |
| 3 | **Umbral procedimientos** | 20 para código completo | **10** para código completo (conserva más procedimientos específicos) |
| 4 | **Fallback procedimientos** | Sección directa (`proc_0`) | Prefijo `proc_rare_sec_0` para distinguir de categorías reales |
| 5 | **Features de repetición** | No existían | **6 columnas nuevas**: conteos de repeticiones de códigos y grupos |
| 6 | **Validación de LOS** | No existía | Reporte de percentiles, outliers (P95), códigos frecuentes en outliers |
| 7 | **Features procedimentales** | 360 | **514** (+154 por umbral más bajo) |
| 8 | **Columnas totales** | 1,493 | **1,652** |

---

## 3. Descripción de Cada Archivo v2

### 3.1 `procesamiento_features_v2.py`

Script principal v2. Implementa la misma lógica jerárquica de agrupación con estas mejoras:

| Nivel | Diagnósticos | Procedimientos |
|-------|-------------|---------------|
| 1 (preferido) | Código completo si ≥ 20 pacientes | Código completo si ≥ **10** pacientes |
| 2 (intermedio) | Categoría 3 chars si ≥ 20 pacientes | Categoría 3 chars si ≥ 20 pacientes |
| 3 (fallback) | `rare_cap_{capítulo}` | `rare_sec_{sección}` |

Además: excluye `UUUUUU`, genera features de repetición, valida LOS, analiza outliers.

### 3.2 `processed/features_diagnosticos_agrupados_v2.csv`

Matriz binaria de **11,951 pacientes × 1,122 features diagnósticas**. Diferencia con v1: `UUUUUU` excluido (-1 feature), y las features de fallback ahora tienen prefijo `diag_rare_cap_*` en vez de `diag_E`, `diag_I`, etc.

### 3.3 `processed/features_procedimientos_agrupados_v2.csv`

Matriz binaria de **11,951 pacientes × 514 features procedimentales**. Diferencia con v1: +154 features porque el umbral de código completo bajó de 20 a 10, conservando más procedimientos específicos antes de colapsar.

### 3.4 `processed/model_data_ml_v2.csv` — **DATASET PARA ENTRENAR**

Dataset integrado final:
- **11,951 filas** (pacientes)
- **1,652 columnas** totales:
  - 1,122 features diagnósticas (`diag_*`)
  - 514 features procedimentales (`proc_*`)
  - 10 columnas base
  - 6 features de repetición

**Para entrenar:**
- `X` = todas las columnas **EXCEPTO** `case_id` y `los_dias`
- `y` = `los_dias`
- `case_id` se mantiene solo como identificador

### 3.5 Archivos de reportes v2

| Archivo | Contenido |
|---------|-----------|
| `mapeo_reemplazos_diagnosticos_v2.csv` | Trazabilidad de 6,124 códigos diagnósticos (sin UUUUUU) → grupo asignado y nivel |
| `mapeo_reemplazos_procedimientos_v2.csv` | Trazabilidad de 3,278 códigos procedimentales → grupo asignado y nivel |
| `reporte_frecuencias_diagnosticos_v2.csv` | Frecuencias por pacientes únicos de cada grupo diagnóstico final |
| `reporte_frecuencias_procedimientos_v2.csv` | Frecuencias por pacientes únicos de cada grupo procedimental final |
| `reporte_repeticiones_codigos_v2.csv` | Para cada paciente: cuántos códigos repetidos tiene, cuántos grupos únicos, máxima repetición de un grupo |
| `reporte_target_los_v2.csv` | Estadísticas descriptivas de `los_dias`: media, mediana, std, percentiles, outliers |
| `reporte_codigos_outliers_v2.csv` | Top 30 diagnósticos y top 30 procedimientos más frecuentes en pacientes con LOS ≥ P95 |

---

## 4. Estadísticas Cuantitativas v2

### 4.1 Diagnósticos

| Métrica | v1 | v2 |
|---------|----|----|
| Códigos ICD-10-CM únicos | 6,125 | **6,124** (sin UUUUUU) |
| Grupos finales generados | 1,123 | **1,122** |
| Nivel `codigo_completo` | 703 códigos (11.5%) | **702** códigos |
| Nivel `categoria_3_char` | 3,277 códigos (53.5%) | **3,277** códigos |
| Nivel `capitulo_1_char` / `capitulo_rare` | 2,145 códigos (35.0%) | **2,145** códigos (ahora con prefijo `rare_cap_`) |

### 4.2 Procedimientos

| Métrica | v1 | v2 |
|---------|----|----|
| Códigos ICD-10-PCS únicos | 3,278 | **3,278** |
| Grupos finales generados | 360 | **514** |
| Nivel `codigo_completo` | 215 códigos (6.6%) | **371** códigos (11.3%) ← más conservado |
| Nivel `categoria_3_char` | 2,120 códigos (64.7%) | **1,974** códigos (60.2%) |
| Nivel `seccion_1_char` / `seccion_rare` | 943 códigos (28.8%) | **933** códigos (28.5%) (ahora con prefijo `rare_sec_`) |

**Impacto del umbral más bajo en procedimientos:** Al bajar el umbral de código completo de 20 a 10, se conservaron **156 procedimientos adicionales** como códigos específicos en vez de colapsarse a categoría de 3 chars. Esto permite al modelo distinguir entre procedimientos quirúrgicos concretos.

### 4.3 Features de Repetición (nuevas en v2)

| Feature | Descripción | Tipo |
|---------|-------------|------|
| `n_diag_codigos_repetidos` | N° de registros diagnósticos duplicados (mismo código) | Conteo bruto |
| `grupos_unicos_diag` | N° de grupos diagnósticos distintos del paciente | Conteo bruto |
| `max_repeticion_diag_grupo` | Máxima cantidad de veces que un grupo diagnóstico aparece | Conteo bruto |
| `n_proc_codigos_repetidos` | N° de registros de procedimientos duplicados | Conteo bruto |
| `grupos_unicos_proc` | N° de grupos procedimentales distintos del paciente | Conteo bruto |
| `max_repeticion_proc_grupo` | Máxima cantidad de veces que un grupo procedimental aparece | Conteo bruto |

**¿Por qué conteos brutos y no log1p?** Random Forest y Gradient Boosting trabajan con splits binarios sobre valores, no con distancias. La transformación logarítmica no aporta beneficio y dificulta la interpretación de feature importance.

- Pacientes con diagnósticos repetidos: 1,757 de 11,951 (14.7%)
- Pacientes con procedimientos repetidos: 200 de 11,951 (1.7%)

### 4.4 Variable Objetivo: `los_dias`

| Estadística | Valor |
|-------------|-------|
| N total | 11,951 |
| Nulos | 0 |
| Mínimo | 0 días |
| P5 | 1 día |
| P25 | 1 día |
| **Mediana** | **3 días** |
| **Media** | **6.47 días** |
| P75 | 6 días |
| P90 | 15 días |
| **P95 (umbral outlier)** | **27 días** |
| P99 | 60 días |
| Máximo | 262 días |
| Desviación estándar | 13.23 días |
| **Outliers (LOS ≥ P95)** | **609 pacientes (5.1%)** |

**Interpretación:** La distribución es fuertemente asimétrica (right-skewed). La mediana (3 días) y la media (6.47) difieren significativamente, confirmando que hay una cola larga de estancias prolongadas. El P99 de 60 días y el máximo de 262 días sugieren casos extremos que podrían requerir transformación del target (log) o estratificación al entrenar.

### 4.5 Dataset Final v2

| Métrica | Valor |
|---------|-------|
| Pacientes | **11,951** |
| Columnas totales | **1,652** |
| Features diagnósticas (`diag_*`) | **1,122** |
| Features procedimentales (`proc_*`) | **514** |
| Features base | **10** |
| Features de repetición | **6** |
| **Densidad de la matriz binaria** | **0.58%** |
| Nulos restantes | **0** |

---

## 5. Decisiones Documentadas

### 5.1 Tratamiento de `UUUUUU`

**Hallazgo:** `UUUUUU` aparece en **3,486 pacientes (29.2%)**, siempre como **diagnóstico primario (tipo P)**. Representa ingresos por urgencias donde no se asigna un código ICD-10 de diagnóstico al momento del ingreso.

**Decisión:** Excluirlo de la matriz diagnóstica predictiva.

**Justificación:** No es un código clínico interpretable — es un código operativo de urgencias. Si se incluye como predictor, el modelo aprendería que "este paciente llegó por urgencias" pero sin saber por qué (puede ser infarto, traumatismo, intoxicación, etc.). El LOS de estas hospitalizaciones tiene altísima varianza dependiendo del diagnóstico real. Incluirlo generaría predicciones con alta incertidumbre y sesgos hacia el LOS promedio de urgencias.

**Nota:** La información de urgencia ya está parcialmente capturada en la feature `es_urgencia` del dataset maestro.

### 5.2 Diagnósticos Primarios y Secundarios

**Decisión:** Se usan ambos tipos de diagnósticos (primarios y secundarios) en esta prueba de concepto.

**Justificación:** El profesor indicó que pueden usarse. No existe información clara sobre el timing de codificación en el sistema fuente — no se puede confirmar si los diagnósticos secundarios se codifican al ingreso o al egreso.

> ⚠️ **Advertencia futura:** Si los diagnósticos secundarios se codifican al egreso, constituirían leakage temporal (el modelo vería información que solo existe después de conocer el LOS). Para un modelo de producción, sería necesario separar diagnósticos conocidos al ingreso vs. diagnósticos codificados al egreso.

### 5.3 Mezcla de Poblaciones / Servicio Hospitalario

**Decisión:** No se implementa estratificación por servicio hospitalario en v2.

**Justificación:** El objetivo actual es capturar patrones predictivos internos de esta base de datos, no construir un modelo universal transferible. Las asociaciones encontradas son **predictivas, no causales**. La estratificación por servicio es trabajo futuro para un modelo más completo.

### 5.4 Pérdida de Especificidad en Procedimientos

**Problema en v1:** Muchos procedimientos colapsaban rápidamente a sección (1 char), generando solo 360 grupos con una compresión de 9:1. La sección `0` (médico-quirúrgica) agrupaba desde apendicectomías hasta bypasses coronarios.

**Solución v2:**
1. **Umbral de código completo bajado a 10** (era 20). Esto conserva 156 procedimientos específicos adicionales.
2. **Prefijo `proc_rare_sec_*`** para los que aún caen a sección, distinguiéndolos de categorías reales.

**Resultado:** 514 grupos procedimentales (era 360). La compresión pasó de 9.1:1 a 6.4:1.

**¿Por qué umbral 10 y no 5?** Con 10 pacientes, un procedimiento tiene representación mínima para que un árbol de decisión pueda hacer splits informativos (mínimo ~10 muestras en un nodo hoja es configuración estándar de RF). Bajar a 5 introduciría demasiado ruido.

### 5.5 Capítulos Diagnósticos Amplios

**Problema en v1:** El fallback a capítulo (1 char) generaba features como `diag_E` que mezclaban toda la endocrinología sin distinción. Esto se confundía visualmente con categorías legítimas de 3 chars.

**Solución v2:** Prefijo `diag_rare_cap_E` para estos fallbacks. Esto:
- Permite al modelo **distinguir** entre un diagnóstico específico (`diag_E119`) y un "cajón de sastre" (`diag_rare_cap_E`).
- Mejora la interpretabilidad del feature importance.
- No cambia la cantidad de features, solo renombra.

### 5.6 Repetición de Códigos

**Problema en v1:** Las features binarias solo capturaban presencia/ausencia. Un paciente con 1 procedimiento y otro con 10 procedimientos del mismo tipo eran indistinguibles.

**Solución v2:** 6 features numéricas de repetición/carga que capturan la "intensidad" del uso de servicios clínicos, sin duplicar las features binarias.

### 5.7 Outliers de LOS

Se definió outlier como LOS ≥ P95 (27 días). Se identificaron 609 pacientes outlier. Los diagnósticos y procedimientos más frecuentes en estos pacientes se guardaron en `reporte_codigos_outliers_v2.csv`.

**No se crearon features de outlier en el dataset principal** — hacerlo implicaría riesgo de leakage (la condición de outlier depende del target `los_dias`).

### 5.8 Leakage — Columnas Excluidas

Las siguientes columnas están **excluidas** de los predictores:

| Columna | Razón de exclusión |
|---------|-------------------|
| `los_dias` | Es el target, no predictor |
| `fecha_egreso` | Se conoce al final de la estancia |
| `fecha_ingreso` | Se transformó en `mes_ingreso` y `dia_semana_ingreso` |
| `los_cero` | Derivada del target |
| `los_negativo` | Derivada del target |
| `fechas_invalidas` | Flag de calidad de datos |
| `procedimientos` | Texto crudo, ya transformado en features binarias |
| `diagnosticos_primarios` | Texto crudo, ya transformado |
| `diagnosticos_secundarios` | Texto crudo, ya transformado |
| `case_id` | Identificador, mantenido solo para trazabilidad |

---

## 6. Ventajas y Riesgos de v2

### 6.1 Ventajas ✅

| Aspecto | Descripción |
|---------|-------------|
| **UUUUUU tratado** | Eliminado de predictores; la feature `es_urgencia` ya captura parcialmente esa información |
| **Más procedimientos específicos** | 514 vs 360 features procedimentales — menos pérdida de información clínica |
| **Fallbacks distinguibles** | `rare_cap_*` y `rare_sec_*` se distinguen de categorías legítimas en feature importance |
| **Features de repetición** | Capturan intensidad de uso de servicios sin romper la lógica binaria |
| **LOS validado** | Se conoce la distribución del target antes de entrenar |
| **Outliers analizados** | Se sabe qué diagnósticos/procedimientos prevalecen en estancias largas |
| **0 nulos** | Dataset completamente limpio |

### 6.2 Riesgos Pendientes ⚠️

| Riesgo | Severidad | Estado |
|--------|-----------|--------|
| **Leakage temporal en diagnósticos secundarios** | 🟡 Media | Documentado; se permite por indicación del profesor |
| **Leakage potencial en `n_procedimientos`** | 🟡 Media | El n° total de procedimientos podría correlacionar con LOS por definición |
| **Alta dimensionalidad** | 🟡 Media | 1,652 features para 12K pacientes. RF y GBM lo manejan, pero conviene feature selection posterior |
| **Mezcla de poblaciones** | 🟡 Media | Obstetricia + medicina interna + cirugía en un solo modelo. Trabajo futuro |
| **Distribución asimétrica del target** | 🟡 Media | Mediana=3, media=6.47, max=262. Considerar transformación log al entrenar |
| **Densidad muy baja (0.58%)** | 🟢 Baja | RF y GBM manejan sparsity nativamente |

---

## 7. Columnas del Dataset Final `model_data_ml_v2.csv`

### Columnas base (10)

| Columna | Tipo | Rol |
|---------|------|-----|
| `case_id` | string | **Identificador** — NO usar como predictor |
| `los_dias` | int | **TARGET** — NO usar como predictor |
| `es_urgencia` | int (0/1) | Predictor |
| `n_procedimientos` | int | Predictor |
| `n_diag_primarios` | int | Predictor |
| `n_diag_secundarios` | int | Predictor |
| `n_diag_total` | int | Predictor |
| `tiene_diag_primario` | int (0/1) | Predictor |
| `mes_ingreso` | int (1-12) | Predictor |
| `dia_semana_ingreso` | int (0-6) | Predictor |

### Features de repetición (6)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `n_diag_codigos_repetidos` | int | Códigos diagnósticos repetidos en el paciente |
| `grupos_unicos_diag` | int | Grupos diagnósticos distintos del paciente |
| `max_repeticion_diag_grupo` | int | Máx. repeticiones de un solo grupo diagnóstico |
| `n_proc_codigos_repetidos` | int | Códigos de procedimiento repetidos |
| `grupos_unicos_proc` | int | Grupos procedimentales distintos |
| `max_repeticion_proc_grupo` | int | Máx. repeticiones de un solo grupo procedimental |

### Features diagnósticas (1,122 columnas `diag_*`)

Binarias (0/1). Incluyen:
- `diag_{codigo}` — diagnóstico específico (ej: `diag_E119`)
- `diag_{cat3}` — categoría de 3 chars (ej: `diag_E11`)
- `diag_rare_cap_{capitulo}` — fallback a capítulo con prefijo (ej: `diag_rare_cap_E`)

### Features procedimentales (514 columnas `proc_*`)

Binarias (0/1). Incluyen:
- `proc_{codigo}` — procedimiento específico (ej: `proc_4A1ZXQZ`)
- `proc_{cat3}` — categoría de 3 chars (ej: `proc_0DB`)
- `proc_rare_sec_{seccion}` — fallback a sección con prefijo (ej: `proc_rare_sec_0`)

---

1. Conteo de códigos (Diagnósticos y Procedimientos)
Diagnósticos:
Códigos originales (ICD-10 completos): 6,124
Grupos finales (después de aplicar jerarquías): 1,321
Procedimientos:
Códigos originales (completos): 3,278
Grupos finales: 433
Total de códigos analizados: 9,402 códigos distintos.

2. Porcentaje que quedó en el dataset del modelo
Si comparamos los códigos originales con los grupos finales:

Diagnósticos: Solo el 21.5% de los nombres originales se convirtieron en columnas (el resto se agrupó por jerarquía).
Procedimientos: Solo el 13.2% de los nombres originales se convirtieron en columnas.
¿Es muy pequeño? Al contrario, es el tamaño ideal.

Si hubiéramos usado los 9,402 códigos originales como columnas:

Dataset vacío: La gran mayoría de las columnas tendrían puros ceros (porque un código específico quizás solo lo tiene 1 paciente en 12,000). El modelo no podría aprender nada de una columna que casi siempre es cero.
Maldición de la dimensionalidad: Tendrías más columnas que pacientes, lo que garantiza que el modelo se aprenda los datos de memoria (overfitting) en lugar de aprender medicina.
Conclusión: Tener 1,652 features (1,321 diag + 433 proc + variables base) para 11,951 pacientes es una proporción muy sana. Estás usando aproximadamente el 18% de la diversidad original de códigos, pero agrupados de forma que cada columna tiene suficientes "1s" para que el modelo (XGBoost) pueda encontrar patrones estadísticos reales.

## 8. Conclusión

> **`model_data_ml_v2.csv` es el dataset recomendado para entrenar Random Forest y Gradient Boosting.** Resuelve los problemas principales de v1: UUUUUU excluido, fallbacks diferenciados, más procedimientos conservados, features de repetición añadidas, y target validado.

> **Para entrenar:** `X = todas las columnas excepto 'case_id' y 'los_dias'`; `y = 'los_dias'`.

> **Trabajo futuro:** Estratificación por servicio hospitalario, verificación del timing de diagnósticos secundarios, y evaluación de transformación logarítmica del target.
