# Plan de Implementación — Mejora de Modelos de Predicción de LOS
**Capstone Grupo 16 — Predicción de Length of Stay Hospitalario**

---

## 1. Diagnóstico del Estado Actual

El modelo Random Forest v1 confirmó que los features ICD agrupados contienen señal predictiva para estancias cortas (MAE=1.34 días en tramo 0–2), pero presenta tres problemas estructurales:

1. **Sesgo hacia el promedio:** Rango de predicción limitado a 1.90–20.14 días. Nunca predice estancias largas.
2. **Subestimación sistemática:** 100% de subestimación en pacientes con LOS ≥27 días (Recall PLOS = 0%).
3. **Features insuficientes para capturar severidad:** Los códigos ICD binarios indican *qué tiene* el paciente, pero no *qué tan grave* es su condición.

**Conclusión clave del v1:** El dataset está bien construido. El problema está en la estrategia de modelado y en la falta de features de severidad clínica.

---

## 2. Estrategia General — Fases de Trabajo

```
Fase 1: XGBoost v1 (comparación directa con RF v1)
   ↓
Fase 2: Ingeniería de Features v3 (capturar severidad clínica)
   ↓
Fase 3: Random Forest v2 + XGBoost v2 (con SMOTE, pesos, dos etapas)
   ↓
Fase 4: Comparación final y selección del modelo
```

---

## 3. Fase 1 — Modelo XGBoost v1 (Comparación Directa)

### 3.1 Objetivo
Entrenar un XGBoost con **exactamente los mismos datos y configuración** que el RF v1 para aislar el efecto del algoritmo. Si XGBoost mejora significativamente con los mismos features, confirma que el problema del RF v1 es algorítmico (el promediado de árboles). Si no mejora, confirma que el problema principal son los features.

### 3.2 Configuración Propuesta

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `n_estimators` | 300 | Igual que RF v1 para comparación justa |
| `max_depth` | 6 | XGBoost es más propenso a overfitting que RF; empezar más conservador |
| `learning_rate` | 0.1 | Tasa de aprendizaje estándar para primera iteración |
| `subsample` | 0.8 | Usar 80% de los datos por árbol para reducir overfitting |
| `colsample_bytree` | 0.8 | Usar 80% de las features por árbol |
| `reg_alpha` | 0.1 | Regularización L1 — ayuda con la sparsidad del dataset (0.58% densidad) |
| `reg_lambda` | 1.0 | Regularización L2 estándar |
| `objective` | `reg:squarederror` | Mismo objetivo de regresión |
| `random_state` | 42 | Misma semilla para reproducibilidad |

### 3.3 ¿Por qué XGBoost debería funcionar mejor?

A diferencia de Random Forest (que promedia 300 árboles independientes), XGBoost construye árboles **secuencialmente**: cada nuevo árbol se enfoca en los errores que dejaron los árboles anteriores. Esto significa que después de los primeros 50 árboles que aprenden a predecir estancias cortas, los siguientes 250 árboles se concentran en los residuos — es decir, en los pacientes que el modelo está prediciendo mal (los de estancias largas).

**Referencia bibliográfica:** Estudios comparativos recientes (2023–2024) publicados en NIH PubMed reportan que XGBoost frecuentemente logra RMSE y MAE inferiores a Random Forest en predicción de LOS hospitalario, especialmente en la cola derecha de la distribución (Amegroups, 2024; MDPI, 2024).

### 3.4 Métricas de Comparación RF v1 vs XGBoost v1

Para que la comparación sea válida, ambos modelos deben evaluarse con:
- **Mismos datos de test** (mismo `random_state=42`, mismo split 80/20)
- **Mismas métricas:** MAE, RMSE, MedAE (globales) + MAE por tramo + Precision/Recall/F1 PLOS
- **Misma transformación del target:** `log1p(los_dias)`

### 3.5 Resultado Esperado
- XGBoost debería reducir el RMSE (de 11.13 a ~8–9 días)
- Debería mejorar parcialmente la detección de PLOS (Recall > 0%, aunque probablemente bajo)
- Si la mejora es mínima, esto confirma que los features son el cuello de botella → priorizar Fase 2

---

## 4. Fase 2 — Ingeniería de Features v3 (Capturar Severidad Clínica)

### 4.1 El Problema Fundamental

Los features actuales son binarios: `diag_I10 = 1` significa "el paciente tiene hipertensión". Pero no distingue entre un hipertenso controlado (LOS corto) y un hipertenso con crisis que necesita UCI (LOS largo). **Necesitamos features que capturen la carga de enfermedad y la complejidad del caso.**

### 4.2 Estrategia A — Índice de Comorbilidad de Charlson (CCI)

#### ¿Qué es?
El Índice de Charlson es un puntaje numérico estándar en medicina que asigna pesos a 17 condiciones médicas según su impacto en la mortalidad a un año. Fue desarrollado por Mary Charlson en 1987 y es el índice de comorbilidad más citado en la literatura médica mundial.

#### ¿Cómo funciona?
Cada condición tiene un peso de 1 a 6 puntos:

| Peso | Condiciones |
|------|------------|
| 1 punto | Infarto de miocardio, insuficiencia cardíaca, enfermedad vascular periférica, enfermedad cerebrovascular, demencia, EPOC, úlcera péptica, enfermedad hepática leve, diabetes sin complicaciones |
| 2 puntos | Hemiplejia, enfermedad renal moderada/severa, diabetes con complicaciones, tumores sólidos, leucemia, linfoma |
| 3 puntos | Enfermedad hepática moderada/severa |
| 6 puntos | Tumor metastásico, SIDA |

**Ejemplo:** Un paciente con diabetes tipo 2 (1 punto) + insuficiencia renal crónica (2 puntos) + EPOC (1 punto) = CCI de 4. Un CCI ≥ 3 se considera alta carga de comorbilidad.

#### ¿Cómo implementarlo con nuestros datos ICD-10?
Existen mapeos validados de códigos ICD-10 a categorías Charlson (algoritmo de Quan et al., 2005). Nuestro pipeline leería los códigos diagnósticos de cada paciente, los mapearía a las 17 categorías, asignaría los pesos, y sumaría un puntaje total.

**Feature resultante:** Una columna numérica `charlson_index` con valores de 0 a ~20.

**Referencia:** Quan H, Sundararajan V, Halfon P, et al. *"Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data."* Medical Care, 2005;43(11):1130-1139.

### 4.3 Estrategia B — Índice de Elixhauser

#### ¿Qué es?
Similar al Charlson pero más completo: incluye 31 categorías de comorbilidad (vs. 17 del Charlson). Estudios recientes sugieren que el Elixhauser tiene **superior poder predictivo para LOS** que el Charlson, porque incluye condiciones como obesidad, trastornos psiquiátricos y abuso de sustancias que afectan la estancia pero no la mortalidad.

#### Implementación
Usar los pesos de van Walraven, que convierten las 31 categorías binarias en un puntaje continuo. Alternativamente, dejar las 31 categorías como features binarias individuales y dejar que el modelo aprenda sus pesos — esto es lo que la literatura recomienda para ML.

**Features resultantes:** 31 columnas binarias (una por categoría Elixhauser) + 1 columna de puntaje total.

**Referencia:** van Walraven C, et al. *"A modification of the Elixhauser comorbidity measures into a point system for hospital death using administrative data."* Medical Care, 2009;47(6):626-633.

### 4.4 Estrategia C — Features de Interacción Clínica

Crear features derivadas que capturen combinaciones peligrosas que los códigos individuales no pueden capturar:

| Feature | Fórmula | Lógica Clínica |
|---------|---------|----------------|
| `n_comorbilidades_graves` | Conteo de códigos ICD que caen en categorías Charlson peso ≥ 2 | Pacientes con múltiples condiciones severas simultáneas |
| `tiene_procedimiento_mayor` | 1 si tiene ≥1 procedimiento clasificado como quirúrgico mayor | Cirugías extensas prolongan la estancia |
| `ratio_diag_proc` | `n_diag_total / n_procedimientos` | Ratio alto puede indicar complejidad diagnóstica vs. intervención |
| `es_multimorbido` | 1 si tiene ≥ 3 diagnósticos secundarios de capítulos ICD distintos | Pacientes con enfermedades en múltiples sistemas del cuerpo |
| `tiene_infeccion` | 1 si tiene códigos del capítulo A00-B99 (infecciones) | Infecciones hospitalarias prolongan significativamente el LOS |

### 4.5 Estrategia D — Estratificación por Tipo de Hospitalización

#### El problema actual
El modelo actual mezcla pacientes obstétricos (partos normales de 1–2 días) con pacientes oncológicos (quimioterapia de semanas). Esto diluye los patrones.

#### Propuesta de implementación
Crear una columna `tipo_hospitalizacion` basada en el diagnóstico primario:

| Tipo | Capítulos ICD-10 | LOS Esperado |
|------|-----------------|--------------|
| Obstétrica | O00-O9A (Embarazo, parto y puerperio) | 1–3 días típicamente |
| Quirúrgica | Pacientes con ≥1 procedimiento registrado | Variable, según complejidad |
| Médica | Todos los demás | Variable |

#### Dos opciones de uso

**Opción A — Feature adicional:** Agregar `tipo_hospitalizacion` como una columna categórica codificada (one-hot encoding). El modelo global aprende con esta señal adicional.

**Opción B — Modelos separados:** Entrenar un modelo independiente para cada tipo. Esto es más potente pero requiere que cada subgrupo tenga suficientes datos para entrenar.

**Recomendación:** Empezar con la Opción A (más simple, un solo modelo) y evaluar. Si los resultados siguen siendo insatisfactorios en algún subgrupo, pasar a la Opción B.

### 4.6 Entregable de la Fase 2

Un nuevo script `procesamiento_features_v3.py` que genere `model_data_ml_v3.csv` con:
- Todas las columnas actuales de la v2 (features binarias ICD agrupadas)
- `charlson_index` (numérica continua)
- 31 features binarias Elixhauser (o las que apliquen a nuestros datos)
- Features de interacción (n_comorbilidades_graves, es_multimorbido, etc.)
- `tipo_hospitalizacion` (one-hot encoded)

---

## Fase 2.5: Hyperparameter Tuning
Paso 1: Crear el script de Tuning Crearé un archivo llamado ml/modelos/XGB/tuning_xgboost.py. Este archivo no guardará un modelo final, su único trabajo será correr simulaciones.

Paso 2: Definir el "Espacio de Búsqueda" (Grid) Le diremos al algoritmo que busque en estos rangos lógicos para datos tabulares asimétricos:

n_estimators: [100, 300, 500, 800] (¿Cuántos árboles?)
max_depth: [3, 5, 7, 9] (¿Qué tan complejos?)
learning_rate: [0.01, 0.05, 0.1, 0.2] (¿Qué tan rápido aprende?)
subsample / colsample_bytree: [0.6, 0.8, 1.0] (Evita el overfitting)
Paso 3: Validacion Cruzada (5-Fold CV) Para cada combinación que intente, el algoritmo partirá la data de entrenamiento en 5 trozos. Entrenará con 4 y validará con 1. Repetirá esto 5 veces. Esto garantiza que el modelo es robusto y no tuvo "suerte" con los datos.

Paso 4: Métrica de Selección Le diremos que busque la combinación que logre el menor MAE (Error Absoluto Medio) en los 5 folds.

Paso 5: Ejecución y Registro Al terminar, el script imprimirá algo como: Mejores parámetros encontrados: {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05} Nosotros copiaremos exactamente esos parámetros y los pegaremos en nuestro script final de entrenamiento (entrenar_xgboost_v3.py), citando que fueron obtenidos mediante RandomizedSearchCV.

## 5. Fase 3 — Random Forest v2 + XGBoost v2 (Técnicas Avanzadas)

### 5.1 Técnica 1 — SMOGN (Sobremuestreo para Regresión)

#### ¿Qué es?
SMOTE (Synthetic Minority Oversampling TEchnique) fue diseñado para clasificación. Para regresión se usa **SMOGN** (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise), que es la extensión de SMOTE para variables continuas.

#### ¿Cómo funciona?
1. Define una **función de relevancia** que indica qué valores del target son "raros" (en nuestro caso, LOS ≥ 14 días)
2. **Sub-muestrea** los casos frecuentes (LOS 0–6 días) para reducir su dominio
3. **Genera muestras sintéticas** de los casos raros interpolando entre pacientes reales con LOS largo y añadiendo ruido gaussiano controlado

#### Implementación
```
pip install smogn
```
Se aplica **solo sobre el set de entrenamiento** (nunca sobre el test). Esto es crítico para evitar leakage.

#### Resultado esperado
El dataset de entrenamiento pasaría de ~9,500 filas a ~12,000–15,000 filas, con una distribución más equilibrada entre estancias cortas y largas. Esto forzaría al modelo a ver más ejemplos de pacientes complejos.

**Referencia:** Branco P, Torgo L, Ribeiro RP. *"SMOGN: a Pre-processing Approach for Imbalanced Regression."* Proceedings of Machine Learning Research, 2017;74:36-50.

### 5.2 Técnica 2 — Pesos de Muestra Personalizados

#### ¿Qué es?
En lugar de generar datos sintéticos, se le dice al algoritmo: *"Cuando te equivoques en un paciente de LOS largo, considéralo como si te hubieras equivocado en 5 pacientes."* Esto se hace asignando un peso (sample_weight) a cada fila de entrenamiento.

#### Esquema de pesos propuesto

| Tramo LOS | Peso | Justificación |
|-----------|------|---------------|
| 0–2 días | 1.0 | Peso base (grupo mayoritario) |
| 3–6 días | 1.5 | Leve incremento |
| 7–13 días | 3.0 | Estancias intermedias donde el modelo empieza a fallar |
| 14–26 días | 5.0 | Estancias largas, clínicamente importantes |
| 27+ días | 10.0 | Casos extremos, máxima prioridad para el hospital |

#### Implementación
Tanto `RandomForestRegressor` como `XGBRegressor` de scikit-learn aceptan el parámetro `sample_weight` en el método `.fit()`. No requiere librerías adicionales.

#### Ventaja sobre SMOGN
- No genera datos artificiales (más transparente académicamente)
- No altera el tamaño del dataset
- Más fácil de justificar en un informe

### 5.3 Técnica 3 — Modelo de Dos Etapas

#### Concepto
Dividir el problema en dos subproblemas más simples:

```
                    ┌─────────────────────┐
Paciente nuevo  →   │ Etapa 1: Clasificador│
                    │ ¿LOS ≥ 14 días?      │
                    │ (Sí / No)            │
                    └──────┬──────┬────────┘
                           │      │
                      No   │      │  Sí
                           ▼      ▼
                    ┌──────────┐ ┌──────────┐
                    │ Regresor │ │ Regresor │
                    │ Corto    │ │ Largo    │
                    │ (0-13 d) │ │ (14+ d)  │
                    └──────────┘ └──────────┘
```

#### Etapa 1 — Clasificador Binario
- **Objetivo:** Predecir si el paciente tendrá LOS ≥ 14 días (Sí/No)
- **Algoritmo:** XGBoost Classifier o Random Forest Classifier
- **Métricas:** Precision, Recall, F1-Score (aquí sí es clasificación pura)
- **Umbral de decisión:** Ajustar el threshold para maximizar Recall (preferimos falsas alarmas a pacientes no detectados)

#### Etapa 2 — Regresores Especializados
- **Regresor Corto:** Entrenado solo con pacientes de LOS < 14 días (~88% de la cohorte). Este modelo tiene una distribución mucho más homogénea.
- **Regresor Largo:** Entrenado solo con pacientes de LOS ≥ 14 días (~12% de la cohorte). Al no estar "contaminado" por los miles de pacientes de 1–2 días, puede aprender patrones específicos de estancias prolongadas.

#### ¿Por qué 14 días como umbral de corte?
- 14 días es el punto medio de la distribución "difícil" (tramo 14–26 + tramo 27+)
- Agrupa ~12% de los pacientes en el grupo largo, lo que da ~1,400 pacientes para entrenar el regresor largo — suficiente para un modelo robusto
- Clínicamente, 2 semanas de hospitalización es el punto donde se activan protocolos de revisión en muchos hospitales

**Referencia:** La literatura en predicción de LOS (NIH PubMed, 2023) documenta que los modelos de dos etapas (clasificación + regresión) frecuentemente superan a los modelos de una sola etapa en la detección de estancias prolongadas, precisamente porque permiten optimizar el recall en la etapa de clasificación sin sacrificar la precisión de la regresión.

### 5.4 Función de Pérdida Asimétrica (Solo XGBoost)

#### Concepto
XGBoost permite definir funciones de pérdida personalizadas. Se puede crear una función que **castigue más la subestimación que la sobreestimación**. Clínicamente tiene sentido: es peor decirle al hospital "este paciente se va en 5 días" cuando en realidad se queda 30 (el hospital no planifica recursos) que decir "se queda 10 días" cuando se va en 5 (el hospital tiene una cama extra).

#### Implementación conceptual
```
Si error < 0 (subestimación): penalización = 3 × |error|²
Si error > 0 (sobreestimación): penalización = 1 × |error|²
```

Esto haría que XGBoost prefiera sobreestimar antes que subestimar, lo cual es clínicamente más seguro.

---

## 6. Fase 4 — Comparación Final y Selección

### 6.1 Tabla de Comparación Esperada

| Modelo | MAE | RMSE | Recall PLOS | Notas |
|--------|-----|------|-------------|-------|
| RF v1 (baseline) | 4.16 | 11.13 | 0.0% | Solo features v2 |
| XGBoost v1 | ¿? | ¿? | ¿? | Mismos features v2, diferente algoritmo |
| RF v2 (con pesos) | ¿? | ¿? | ¿? | Features v3 + sample weights |
| XGBoost v2 (con pesos) | ¿? | ¿? | ¿? | Features v3 + sample weights |
| Dos Etapas (XGBoost) | ¿? | ¿? | ¿? | Clasificador + 2 regresores |

### 6.2 Criterio de Selección del Modelo Final

El modelo final se selecciona en base a un **balance entre tres objetivos**:

1. **MAE global bajo** → El modelo predice bien en general
2. **Recall PLOS alto** → El modelo detecta pacientes de estancia prolongada
3. **MAE en tramo 27+ razonable** → El modelo no se desploma en los casos extremos

No existe un modelo perfecto. La decisión final depende de qué prioriza el hospital:
- Si prioriza **planificación de camas diaria** → optimizar MAE global
- Si prioriza **detección temprana de casos complejos** → optimizar Recall PLOS

### 6.3 Validación Cruzada

En la Fase 4, todos los modelos finalistas deben evaluarse con **K-Fold Cross-Validation (k=5)** estratificada por tramos de LOS. Esto garantiza que los resultados no dependan de un solo split afortunado.

---

## 7. Orden de Ejecución Recomendado

| Paso | Tarea | Prioridad | Dependencias |
|------|-------|-----------|-------------|
| 1 | Entrenar XGBoost v1 con features v2 actuales | 🔴 Alta | Ninguna |
| 2 | Comparar RF v1 vs XGBoost v1 | 🔴 Alta | Paso 1 |
| 3 | Implementar Charlson Index en features v3 | 🔴 Alta | Ninguna (paralelo a paso 1) |
| 4 | Implementar features Elixhauser | 🟡 Media | Paso 3 |
| 5 | Implementar features de interacción | 🟡 Media | Paso 3 |
| 6 | Agregar tipo_hospitalizacion | 🟡 Media | Paso 3 |
| 7 | Entrenar RF v2 con pesos + features v3 | 🔴 Alta | Pasos 3–6 |
| 8 | Entrenar XGBoost v2 con pesos + features v3 | 🔴 Alta | Pasos 3–6 |
| 9 | Implementar modelo de dos etapas | 🟡 Media | Paso 8 |
| 10 | Validación cruzada K-Fold de modelos finalistas | 🔴 Alta | Pasos 7–9 |
| 11 | Comparación final y selección | 🔴 Alta | Paso 10 |

---

## 8. Referencias Bibliográficas

1. **Quan H, Sundararajan V, Halfon P, et al.** *"Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data."* Medical Care, 2005;43(11):1130-1139. — Algoritmo estándar para mapear códigos ICD a categorías Charlson y Elixhauser.

2. **van Walraven C, Austin PC, Jennings A, Quan H, Forster AJ.** *"A modification of the Elixhauser comorbidity measures into a point system for hospital death using administrative data."* Medical Care, 2009;47(6):626-633. — Sistema de pesos para convertir categorías Elixhauser en puntaje continuo.

3. **Branco P, Torgo L, Ribeiro RP.** *"SMOGN: a Pre-processing Approach for Imbalanced Regression."* Proceedings of Machine Learning Research, 2017;74:36-50. — Técnica de sobremuestreo sintético adaptada para regresión con distribuciones asimétricas.

4. **Charlson ME, Pompei P, Ales KL, MacKenzie CR.** *"A new method of classifying prognostic comorbidity in longitudinal studies: development and validation."* Journal of Chronic Diseases, 1987;40(5):373-383. — Paper original del Índice de Comorbilidad de Charlson.

5. **Elixhauser A, Steiner C, Harris DR, Coffey RM.** *"Comorbidity measures for use with administrative data."* Medical Care, 1998;36(1):8-27. — Paper original del Índice de Elixhauser.

6. **Chen T, Guestrin C.** *"XGBoost: A Scalable Tree Boosting System."* Proceedings of the 22nd ACM SIGKDD, 2016:785-794. — Paper original de XGBoost. Describe el mecanismo de boosting secuencial y funciones de pérdida personalizadas.

8. **Daghistani TA, Elshoush HT, et al.** *"Predictors of in-hospital length of stay among cardiac patients: A machine learning approach."* International Journal of Medical Informatics, 2020;140:104162. — Estudio que demuestra cómo la combinación de índices de comorbilidad con Random Forest y XGBoost mejora significativamente la predicción de estancias largas.

9. **Amato ACM, et al.** *"Machine learning in prediction of individual patient readmissions and length of stay."* SAGE Open Medicine, 2020. — Utiliza machine learning para predecir hospitalizaciones prolongadas aplicando la codificación de Elixhauser como feature clave.

10. **Caetano N, et al.** *"Predicting Length of Stay of Hospitalized Patients using Machine Learning."* Procedia Computer Science, 2014;37:522-527. — Compara modelos de ensamblado para predecir LOS usando variables demográficas y clínicas agrupadas.

---

*Plan de implementación generado el 2026-05-04. Basado en los resultados del modelo RF v1 y revisión bibliográfica de mejores prácticas en predicción de LOS hospitalario.*
