# Material adjunto - Entrega 3

## Ejecución del pipeline `ml_operacional_entrega3`

Todos los comandos de esta sección deben ejecutarse desde la raíz de la carpeta entregada. Esta secuencia ejecuta únicamente el pipeline de `ml_operacional_entrega3` y comprende la preparación de datos, entrenamiento, evaluación y comparación de XGBoost, Random Forest y regresión lineal.

### Requisitos

- Python 3.11 o una versión compatible con los modelos.
- `numpy`, `pandas`, `scikit-learn`, `xgboost` y `joblib`.
- Archivo de entrada incluido en:
  `ml_entrega2/feature_engineering/processed_v3/model_data_v3_escenario_B_charlson.csv`.

### Ejecución con los hiperparámetros incluidos

Esta es la forma recomendada para reproducir los modelos finales sin repetir la búsqueda de hiperparámetros. Los archivos `best_params_*.json` incluidos contienen el resultado del tuning completo previamente ejecutado.

```bash
python3 ml_operacional_entrega3/preparar_datos.py

python3 ml_operacional_entrega3/XGB/entrenar_xgb.py
python3 ml_operacional_entrega3/RF/entrenar_rf.py
python3 ml_operacional_entrega3/LR/entrenar_lr.py

python3 ml_operacional_entrega3/XGB/evaluar_xgb.py
python3 ml_operacional_entrega3/RF/evaluar_rf.py
python3 ml_operacional_entrega3/LR/evaluar_lr.py

python3 ml_operacional_entrega3/comparar_modelos.py
python3 ml_operacional_entrega3/analizar_impacto_clasificador.py
```

El orden es obligatorio: los datos se preparan antes del entrenamiento, los modelos se entrenan antes de evaluarse y la comparación final requiere las predicciones y métricas de los tres modelos.

### Ejecución repitiendo el tuning completo

Esta secuencia recalcula los hiperparámetros de XGBoost y Random Forest antes de entrenar los modelos finales. Los scripts tienen `FAST_RUN = False`, equivalente a 50 configuraciones por búsqueda con validación cruzada. Su ejecución puede tardar varias horas según el equipo disponible.

```bash
python3 ml_operacional_entrega3/preparar_datos.py

python3 ml_operacional_entrega3/XGB/tuning_xgb.py
python3 ml_operacional_entrega3/RF/tuning_rf.py

python3 ml_operacional_entrega3/XGB/entrenar_xgb.py
python3 ml_operacional_entrega3/RF/entrenar_rf.py
python3 ml_operacional_entrega3/LR/entrenar_lr.py

python3 ml_operacional_entrega3/XGB/evaluar_xgb.py
python3 ml_operacional_entrega3/RF/evaluar_rf.py
python3 ml_operacional_entrega3/LR/evaluar_lr.py

python3 ml_operacional_entrega3/comparar_modelos.py
python3 ml_operacional_entrega3/analizar_impacto_clasificador.py
```

La regresión lineal no tiene tuning. Se utiliza como línea base segmentada, sin regularización y con transformación logarítmica de la variable objetivo.

### Salidas principales del pipeline

| Ruta | Contenido |
|---|---|
| `ml_operacional_entrega3/data_splits/` | Particiones train y holdout para pacientes urgentes y programados. |
| `ml_operacional_entrega3/modelos_guardados/` | Clasificadores y regresores entrenados en formato `joblib`. |
| `ml_operacional_entrega3/reports/` | Predicciones, métricas train/holdout, estabilidad y comparación final. |
| `ml_operacional_entrega3/reports/comparacion_final_modelos.md` | Informe comparativo principal de XGB, RF y LR. |

### Análisis complementarios

Estos comandos no son necesarios para entrenar los modelos, pero regeneran análisis derivados del pipeline final:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --status
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py

python3 ml_operacional_entrega3/generar_graficos_xgb_vs_lr.py
python3 ml_operacional_entrega3/generar_graficos_xgb_vs_lr_por_segmento.py
python3 ml_operacional_entrega3/generar_graficos_segmentos_juntos_xgb_vs_lr.py
python3 ml_operacional_entrega3/generar_dashboard_segmentos_compacto.py
python3 ml_operacional_entrega3/generar_grafico_error_xgb_por_tramo_segmento.py
```

---

## Descripción general del material

El proyecto estudia la predicción de la duración de estancia hospitalaria o LOS (*Length of Stay*) a partir de diagnósticos ICD-10-CM, procedimientos ICD-10-PCS, comorbilidad de Charlson y variables operacionales de ingreso.

La Entrega 3 introduce tres decisiones metodológicas principales:

1. Separar pacientes urgentes y programados, debido a sus diferencias en distribución de LOS.
2. Definir estancia prolongada como `LOS >= 14 días`.
3. Utilizar una arquitectura de dos etapas para XGBoost y Random Forest: un clasificador estima `prob_los_14` y un regresor utiliza esa probabilidad para estimar los días exactos.

La regresión lineal se mantiene como caso base y se entrena de forma independiente para cada segmento.

## Estructura de la entrega

```text
E3/
├── README.md
├── data/
├── LOS_0/
├── graficos/
├── ml_entrega2/
├── ml_operacional_entrega3/
└── visualizacion_avanzada/
```

## `data/` - Datos y preparación inicial

Contiene los datos originales, el proceso de limpieza y los conjuntos tabulares normalizados que alimentan las etapas posteriores.

| Archivo o carpeta | Descripción |
|---|---|
| `Datos proyecto LOS.xlsx` | Fuente original de hospitalizaciones utilizada en el proyecto. |
| `datos_diagnostico.csv` | Registros crudos de diagnósticos asociados con cada caso. |
| `procedimiento_pacientes.csv` | Registros crudos de procedimientos asociados con cada caso. |
| `limpieza_datos.py` | Limpia fechas y códigos, valida casos, calcula LOS y construye las tablas procesadas. |
| `analisis.py` | Genera estadísticas descriptivas y reportes de calidad sobre los datos procesados. |
| `processed/dataset_maestro.csv` | Una fila por hospitalización con LOS y variables administrativas. |
| `processed/caso_diagnostico.csv` | Relación normalizada entre casos y diagnósticos. |
| `processed/caso_procedimiento.csv` | Relación normalizada entre casos y procedimientos. |
| `processed/pacientes_rechazados.csv` | Casos excluidos durante la limpieza y su información de auditoría. |
| `reports/reporte_limpieza.csv` | Resumen de registros aceptados, rechazados y transformaciones realizadas. |
| `reports/reporte_estadistico_*.csv` | Estadísticas descriptivas del dataset maestro, diagnósticos y casos rechazados. |

## `LOS_0/` - Análisis de estancias de cero días

Estudia los 250 casos con `LOS = 0` para determinar si corresponden a registros válidos, procedimientos ambulatorios o situaciones administrativas.

| Archivo | Descripción |
|---|---|
| `analisis_los_0.py` | Analiza diagnósticos y procedimientos frecuentes en pacientes sin pernoctación. |
| `analisis_correlaciones_los_0.py` | Calcula coocurrencias entre diagnósticos y procedimientos del grupo LOS = 0. |
| `RESUMEN_ANALISIS_LOS_0.md` | Síntesis descriptiva e interpretación clínica del grupo. |
| `RESUMEN_CORRELACIONES_LOS_0.md` | Síntesis de asociaciones diagnóstico-procedimiento. |
| `diagnosticos_detallado_los_0.csv` | Frecuencias diagnósticas del subconjunto. |
| `procedimientos_detallado_los_0.csv` | Frecuencias de procedimientos del subconjunto. |
| `correlaciones_diagnostico_procedimiento_los_0.csv` | Tabla completa de coocurrencias. |
| `01_*.png` a `06_*.png` | Gráficos descriptivos, distribuciones y mapas de coocurrencia. |

## `graficos/` - Análisis exploratorio general

Contiene el análisis estadístico y visual de la distribución de LOS en la cohorte completa.

| Archivo | Descripción |
|---|---|
| `visualizacion_los.py` | Genera el histograma lineal, la transformación logarítmica y el boxplot por percentiles. |
| `analisis_distribucion_los.py` | Ajusta distribuciones Normal, Log-Normal, Gamma, Weibull, Exponencial y modelos de mezcla. |
| `visualizar_weight_verosimilitud.py` | Grafica la verosimilitud del modelo de mezcla en función de su ponderación. |
| `README_GRAFICOS.md` | Documentación metodológica e interpretación de los gráficos generales. |
| `explicacion_grafico_verosimilitud_mezcla.md` | Explicación específica del gráfico de verosimilitud de la mezcla. |
| `01_*.png` a `07_*.png` | Distribuciones, Q-Q plots, CDF, PDF y verosimilitud de la mezcla. |

### `graficos/diag_proc/`

| Archivo o carpeta | Descripción |
|---|---|
| `analisis_codigos_outliers.py` | Identifica códigos clínicos asociados con estancias extremas y códigos de alta frecuencia. |
| `analisis_complejidad_los.py` | Evalúa la relación entre LOS y número de diagnósticos, diagnósticos secundarios y procedimientos. |
| `diagnosticos/` | Gráficos, estadísticas y conclusiones del análisis de códigos diagnósticos. |
| `procedimientos/` | Gráficos, estadísticas y conclusiones del análisis de procedimientos. |
| `estadisticas_outliers.csv` | Frecuencia y LOS de códigos asociados con valores extremos. |
| `estadisticas_frecuencia.csv` | Códigos más frecuentes y proporción de pacientes afectados. |
| `CONCLUSIONES.md` | Interpretación de los resultados de cada tipo de código. |

## `ml_entrega2/` - Antecedente metodológico

Contiene la ingeniería de variables y los modelos desarrollados en la Entrega 2. Sus resultados son históricos y se utilizan para documentar la evolución hacia el pipeline operacional de la Entrega 3.

### Documentos principales

| Archivo | Descripción |
|---|---|
| `plan_implementacion_modelos.md` | Plan original de comparación entre regresión lineal, Random Forest y XGBoost. |
| `informe_final_comparativo_modelos.md` | Informe de resultados y selección del modelo en la Entrega 2. |
| `modelos/reporte_comparativo_tuning.md` | Comparación de hiperparámetros y escenarios de variables. |

### `feature_engineering/`

| Archivo o carpeta | Descripción |
|---|---|
| `procesamiento_features_v2.py` | Agrupa diagnósticos y procedimientos mediante una jerarquía de frecuencia. |
| `procesamiento_features_v3.py` | Añade Charlson, Elixhauser y construye los escenarios B y C. |
| `comordibipy.py` | Implementación local utilizada para calcular índices de comorbilidad. |
| `analisis_ml_features.md` | Explica las variables construidas, sus fuentes y riesgos de fuga temporal. |
| `processed_v2/` | Matrices de variables y dataset final de la versión 2. |
| `processed_v3/model_data_v3_escenario_B_charlson.csv` | Dataset con Charlson utilizado por la Entrega 3. |
| `processed_v3/model_data_v3_escenario_C_elixhauser.csv` | Dataset alternativo basado en categorías Elixhauser. |
| `processed_v3/reporte_analisis_comorbilidades_v3.md` | Análisis de los índices de comorbilidad. |
| `reports_features/` | Trazabilidad de agrupaciones, frecuencias, repeticiones y distribución del objetivo. |

### `modelos/`

| Ruta | Descripción |
|---|---|
| `modelos/LR/` | Entrenamiento, modelo y métricas de la regresión lineal histórica. |
| `modelos/RF/` | Versiones, tuning, parámetros, predicciones y métricas de Random Forest. |
| `modelos/XGB/` | Versiones, tuning, parámetros, predicciones y métricas de XGBoost. |
| `modelos/*/v1/` | Primeras versiones utilizadas para establecer referencias de desempeño. |
| `modelos/*/final/` | Modelos seleccionados y evaluaciones finales de la Entrega 2. |
| `graficos_comparativos/` | Comparaciones visuales entre los modelos históricos. |

Los archivos `.pkl` son modelos serializados; los `.csv` contienen predicciones o métricas; los `.png` son visualizaciones derivadas de esos resultados.

## `ml_operacional_entrega3/` - Pipeline principal

Esta es la implementación vigente. El split global es 80/20 y está estratificado por combinación de vía de admisión y tramo de LOS. Después del split, cada conjunto se divide en pacientes urgentes y programados.

### Archivos principales

| Archivo | Descripción |
|---|---|
| `preparar_datos.py` | Regenera los cuatro splits de entrenamiento y holdout. |
| `comparar_modelos.py` | Consolida XGB, RF y LR y ejecuta el estudio de estabilidad K-Fold. |
| `analizar_impacto_clasificador.py` | Evalúa el clasificador PLOS y la importancia de `prob_los_14` dentro del regresor XGB. |
| `implementation_plan.md` | Metodología completa del pipeline de dos etapas. |
| `sensitivity_analysis_plan.md` | Diseño experimental de los cuatro escenarios de sensibilidad. |

### Modelos

| Ruta | Descripción |
|---|---|
| `XGB/tuning_xgb.py` | Ajusta clasificadores y regresores XGBoost por segmento. |
| `XGB/entrenar_xgb.py` | Genera probabilidades OOF y entrena los modelos XGBoost finales. |
| `XGB/evaluar_xgb.py` | Evalúa XGBoost en train y holdout. |
| `XGB/best_params_*.json` | Mejores hiperparámetros del clasificador y regresor por segmento. |
| `RF/tuning_rf.py` | Ajusta clasificadores y regresores Random Forest por segmento. |
| `RF/entrenar_rf.py` | Genera probabilidades OOF y entrena los modelos Random Forest finales. |
| `RF/evaluar_rf.py` | Evalúa Random Forest en train y holdout. |
| `RF/best_params_*.json` | Mejores hiperparámetros del clasificador y regresor por segmento. |
| `LR/entrenar_lr.py` | Entrena dos regresiones lineales, una urgente y otra programada. |
| `LR/evaluar_lr.py` | Evalúa la regresión lineal en train y holdout. |
| `LR/generar_grafico_regresion_lineal.py` | Genera el gráfico observado frente a predicho del baseline. |
| `LR/informe_regresion_lineal_baseline.md` | Explicación académica de la formulación lineal y sus variables. |

### Utilidades compartidas

| Archivo | Descripción |
|---|---|
| `utils/pipeline_operacional.py` | Rutas, split estratificado, preparación de variables, constructores de modelos y stacking OOF. |
| `utils/model_workflows.py` | Flujos reutilizables de entrenamiento, evaluación, estabilidad y comparación. |
| `utils/metricas_operacionales.py` | MAE, RMSE, MedAE, sesgo, subestimación, MAE asimétrico y métricas PLOS. |

### Datos, modelos y reportes

| Ruta o patrón | Descripción |
|---|---|
| `data_splits/datos_train_*.csv` | 80% destinado al entrenamiento de cada segmento. |
| `data_splits/datos_holdout_*.csv` | 20% independiente destinado a evaluación final. |
| `data_splits/train_con_prob_*.csv` | Conjuntos de entrenamiento enriquecidos con probabilidades OOF. |
| `modelos_guardados/clf_*.joblib` | Clasificadores PLOS de XGB y RF. |
| `modelos_guardados/reg_*.joblib` | Regresores XGB, RF y LR por segmento. |
| `reports/predicciones_{train,holdout}_*.csv` | Predicción individual, LOS observado, error y segmento. |
| `reports/metricas_{train,holdout}_*.csv` | Métricas globales de cada modelo. |
| `reports/metricas_por_segmento_*.csv` | Métricas separadas para urgentes y programados. |
| `reports/metricas_por_tramo_*.csv` | Métricas por 0-2, 3-6, 7-13 y 14+ días. |
| `reports/comparacion_train_holdout_*.csv` | Brecha entre entrenamiento y holdout. |
| `reports/estabilidad_kfold_*.csv` | Resultados de estabilidad mediante cinco pliegues. |
| `reports/comparacion_final_*.csv` | Comparaciones finales globales, por segmento y por tramo. |
| `reports/reporte_evaluacion_*.md` | Informes legibles de evaluación por modelo y split. |
| `reports/reporte_impacto_clasificador_en_regresor.md` | Interpretación del aporte de la primera etapa. |
| `reports/comparacion_final_modelos.md` | Informe principal de selección del modelo. |

### Gráficos derivados

| Archivo o carpeta | Descripción |
|---|---|
| `generar_graficos_xgb_vs_lr.py` | Genera comparación global XGB-LR, IC95 bootstrap e intervalos IP90. |
| `generar_graficos_xgb_vs_lr_por_segmento.py` | Genera las mismas comparaciones por separado para cada segmento. |
| `generar_graficos_segmentos_juntos_xgb_vs_lr.py` | Combina urgentes y programados dentro de cada tipo de gráfico. |
| `generar_dashboard_segmentos_compacto.py` | Produce un resumen compacto para presentación. |
| `generar_grafico_error_xgb_por_tramo_segmento.py` | Grafica la diferencia promedio entre LOS predicho y real por tramo y segmento. |
| `graficos_png/` | Gráficos globales XGB frente a LR. |
| `graficos_csv/` | Datos e intervalos utilizados para construir esos gráficos. |
| `graficos_png_por_segmento/` | Gráficos individuales para urgentes y programados. |
| `graficos_csv_por_segmento/` | Datos e intervalos de los gráficos individuales por segmento. |
| `graficos_png_presentacion/` | Figuras compactas para exposición. |
| `graficos_csv_presentacion/` | Datos de respaldo de las figuras de presentación. |
| `graficos_png_segmentos_juntos/` | Destino creado al ejecutar el generador de figuras con ambos segmentos en una imagen por métrica. |
| `graficos_csv_segmentos_juntos/` | Tablas incluidas para construir los gráficos conjuntos. |

### Urgencias

| Archivo o carpeta | Descripción |
|---|---|
| `urgencias/generar_graficos_urgencias.py` | Compara distribución, tramos y tasa PLOS entre vías de admisión. |
| `urgencias/urgencias.md` | Evidencia estadística que justifica entrenar modelos separados. |
| `urgencias/01_*.png` a `06_*.png` | Histogramas, boxplot, ECDF, tramos y tasa PLOS. |

### Análisis de sensibilidad

| Archivo | Descripción |
|---|---|
| `sensitivity/run_sensitivity.py` | Orquestador de escenarios, validación de resultados y generación del informe. |
| `sensitivity/common.py` | Carga de datos, entrenamiento perturbado, métricas y bootstrap compartidos. |
| `sensitivity/escenario_1_umbrales_plos.py` | Reentrena el pipeline con umbrales PLOS de 7, 14, 21 y 27 días. |
| `sensitivity/escenario_2_ablation_features.py` | Retira variables o la etapa clasificadora para medir su aporte. |
| `sensitivity/escenario_3_punto_operacion.py` | Evalúa políticas de alerta según umbral probabilístico. |
| `sensitivity/escenario_4_hiperparametros.py` | Evalúa configuraciones cercanas a los hiperparámetros seleccionados. |
| `sensitivity/reporting.py` | Construye el reporte consolidado a partir de los CSV. |
| `sensitivity/smoke_test.py` | Comprueba rápidamente que imports, modelos y datos estén disponibles. |
| `sensitivity/README.md` | Comandos de ejecución y recuperación del análisis. |
| `sensitivity/results/` | CSV de los escenarios, manifiesto e informe consolidado. |

## `visualizacion_avanzada/` - Explicabilidad y seguimiento

| Archivo | Descripción |
|---|---|
| `analisis_shap.py` | Calcula valores SHAP de XGBoost y genera gráficos globales, locales y de dependencia. |
| `app_streamlit.py` | Interfaz interactiva para predicción, exploración del holdout y explicabilidad. |
| `registro_wandb.py` | Registra métricas y artefactos de los modelos en Weights & Biases. |
| `EXPLICACION_COMPONENTES.md` | Describe inferencia, SHAP y telemetría de modelos. |
| `shap_summary_*.png` | Importancia y dirección global de las variables. |
| `shap_waterfall_*.png` | Explicación local de una predicción individual. |
| `shap_dependence_*.png` | Relación entre `prob_los_14` y su contribución SHAP. |

La aplicación Streamlit se inicia desde la raíz con:

```bash
streamlit run visualizacion_avanzada/app_streamlit.py
```

El análisis SHAP estático se regenera con:

```bash
python3 visualizacion_avanzada/analisis_shap.py
```

## Convenciones de archivos

| Extensión | Significado |
|---|---|
| `.py` | Código ejecutable o utilidades del pipeline. |
| `.csv` | Datos procesados, predicciones, métricas o trazabilidad. |
| `.json` | Hiperparámetros seleccionados. |
| `.joblib` / `.pkl` | Modelos entrenados serializados. |
| `.md` | Metodología, informes, resúmenes o conclusiones. |
| `.png` | Visualizaciones generadas por los scripts. |

Los directorios `__pycache__/` y los archivos `.DS_Store` son artefactos locales del sistema y no forman parte del análisis.
