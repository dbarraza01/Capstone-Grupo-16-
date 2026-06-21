# Predicción de Estancia Hospitalaria (LOS)
### Proyecto Capstone - Guía de Estructura, Ejecución y Modelado V2 (Final Regularizado)

---

## 📋 Descripción del Proyecto

Este proyecto implementa una solución de **Machine Learning** de extremo a extremo para predecir la duración de la estancia hospitalaria (**Length of Stay - LOS**) de pacientes a partir de datos clínicos estructurados. Utiliza información de diagnósticos (ICD-10-CM), procedimientos (ICD-10-PCS), medidas de comorbilidad clínica (Índices de Charlson y Elixhauser), y factores de ingreso. 

El modelo definitivo es un **XGBoost Regularizado (Escenario B - Charlson)** que opera bajo una transformación logarítmica `log1p(LOS)` para mitigar la asimetría de la cola larga, logrando un **MAE de 3.057 días** en el Holdout Test Set.

---

## 📂 Estructura Completa del Proyecto

```
Capstone-Grupo-16/
├── README.md                                          # 💡 Esta guía rápida de entrada
├── RF_1.png, RF_2.png                                 # 📊 Gráficos de residuos de Random Forest
├── XGB_1.png, XGB_2.png                               # 📊 Gráficos de residuos de XGBoost
├── resumen_modelamiento_los_para_equipo.md            # 📄 Resumen de metodología y resultados V2
├── urgencias.md                                       # 📄 Análisis de urgencias y código UUUUUU
│
├── LOS_0/                                             # 🔍 Análisis de egresos inmediatos (LOS = 0)
│   ├── analisis_los_0.py, analisis_correlaciones_los_0.py
│   ├── correlaciones_diagnostico_procedimiento_los_0.csv, *.csv
│   ├── RESUMEN_ANALISIS_LOS_0.md, RESUMEN_CORRELACIONES_LOS_0.md
│   └── 01_*.png, 02_*.png, 03_*.png, 04_*.png, 05_*.png, 06_*.png
│
├── data/                                              # 💾 Datos crudos y preprocesamiento
│   ├── Datos proyecto LOS.xlsx                        # Datos crudos en Excel
│   ├── datos_diagnostico.csv, procedimiento_pacientes.csv
│   ├── limpieza_datos.py, analisis.py                 # Limpieza y reportes descriptivos
│   ├── processed/                                     # CSVs limpios generados
│   │   ├── dataset_maestro.csv, caso_diagnostico.csv, caso_procedimiento.csv, pacientes_rechazados.csv
│   └── reports/                                       # CSVs de auditoría y métricas
│       └── reporte_limpieza.csv, reporte_estadistico_*.csv
│
├── graficos/                                          # 📈 Análisis y visualización de la variable target
│   ├── analisis_distribucion_los.py, visualizacion_los.py, visualizar_weight_verosimilitud.py
│   ├── README_GRAFICOS.md, explicacion_grafico_verosimilitud_mezcla.md
│   ├── 01_*.png, 02_*.png, 03_*.png, 04_*.png, 05_*.png, 06_*.png, 07_*.png
│   └── diag_proc/                                     # Visualización de impacto de códigos
│       ├── analisis_codigos_outliers.py, analisis_complejidad_los.py
│       ├── diagnosticos/                              # Plots, conclusiones y CSVs de outliers Dx
│       └── procedimientos/                            # Plots, conclusiones y CSVs de outliers Px
│
├── ml/                                                # 🤖 Modelado Predictivo V2 (Final Regularizado)
│   ├── plan_implementacion_modelos.md                 # Plan inicial de diseño de variables
│   ├── informe_final_comparativo_modelos.md           # Reporte de resultados de test holdout
│   ├── feature_engineering/                           # Ingeniería de variables
│   │   ├── comordibipy.py, procesamiento_features_v2.py, procesamiento_features_v3.py, *.md
│   │   ├── processed_v2/                              # Datos agrupados binarios v2
│   │   ├── processed_v3/                              # Escenarios B y C con target log1p
│   │   └── reports_features/                          # Auditoría de agrupaciones de soporte (v2/v3)
│   └── modelos/                                       # Algoritmos entrenados y evaluados
│       ├── reporte_comparativo_tuning.md              # Comparativa de hiperparámetros
│       ├── LR/                                        # Regresión Lineal Baseline (entrenamiento y final/)
│       ├── RF/                                        # Random Forest (tuning, entrenar y final/plots/CSVs)
│       ├── XGB/                                       # XGBoost (tuning, entrenar, final/reporte/plots/CSVs)
│       └── graficos_comparativos/                     # Comparación cruzada de modelos
│           ├── generar_graficos*.py                   # Scripts de visualización
│           ├── comparativa_3_modelos/                 # Gráficos comparativos LR vs RF vs XGB
│           └── xgb_vs_rf/                             # Gráficos comparativos RF vs XGB
│
└── ml2/                                               # ⚠️ Versión 1 de Modelado (Obsolescente)
    ├── entrenar_*_v1.py                               # Scripts sin regularizar
    ├── models/                                        # Serialización de pkl v1
    └── reports_modelos/                               # CSVs de predicciones y métricas v1
```

---

## 🛠️ Guía de Referencia Rápida para Modificaciones

Para realizar cambios o buscar información, dirígirse a los siguientes archivos clave (rutas relativas):

*   **Modificar la limpieza o cruce de datos iniciales:** Modifica el script [limpieza_datos.py](data/limpieza_datos.py).
*   **Modificar variables predictoras (Charlson, Elixhauser o soporte):** Modifica el script [procesamiento_features_v3.py](ml/feature_engineering/procesamiento_features_v3.py).
*   **Cambiar la grilla de búsqueda de hiperparámetros de XGBoost:** Modifica el script [tuning_xgboost_regularizado.py](ml/modelos/XGB/tuning_xgboost_regularizado.py).
*   **Re-entrenar el modelo final XGBoost ganador y generar sus métricas:** Ejecuta o modifica [entrenar_xgboost_final.py](ml/modelos/XGB/entrenar_xgboost_final.py).
*   **Modificar los gráficos comparativos de rendimiento:** Modifica el script [generar_graficos_3_modelos.py](ml/graficos_comparativos/generar_graficos_3_modelos.py).
*   **Consultar la justificación de modelado y resultados de Holdout:** Consulta el archivo [informe_final_comparativo_modelos.md](ml/informe_final_comparativo_modelos.md).

---

## 📖 Catálogo Detallado de Archivos

### 1. Directorio Raíz del Proyecto

*   **[resumen_modelamiento_los_para_equipo.md](resumen_modelamiento_los_para_equipo.md)**:
    - *Función:* Resumen técnico de la metodología, justificaciones y resultados V2.
    - *Entrada/Salida:* Documento markdown descriptivo con métricas cruzadas y holdout.
    - *Rol:* Guía técnica para el equipo sobre el tuning y regularización aplicada.
    - *Edición:* Actualizar al re-calcular modelos o agregar algoritmos adicionales.
*   **[urgencias.md](urgencias.md)**:
    - *Función:* Documento explicativo sobre el impacto del tipo de ingreso (Urgencia vs Electivo).
    - *Entrada/Salida:* Justificación teórica sobre el código administrativo general `UUUUUU`.
    - *Rol:* Respalda metodológicamente la separación de urgencias de patologías clínicas.
    - *Edición:* Modificar si se incorporan otras marcas administrativas especiales.
*   **Gráficos de dispersión y residuos en la raíz ([RF_1.png](RF_1.png), [RF_2.png](RF_2.png), [XGB_1.png](XGB_1.png), [XGB_2.png](XGB_2.png))**:
    - *Función:* Visualizan la estancia real vs predicha y la distribución de errores de RF y XGB.
    - *Entrada/Salida:* Gráficos PNG autogenerados por los scripts de entrenamiento final.
    - *Rol:* Diagnóstico visual rápido del comportamiento de residuos sobre holdout.
    - *Edición:* Se re-generan automáticamente al re-entrenar los modelos correspondientes.

### 2. Directorio `data/` (Datos y Preprocesamiento)

*   **[Datos proyecto LOS.xlsx](data/Datos%20proyecto%20LOS.xlsx)**:
    - *Función:* Hoja Excel original con los datos clínicos crudos de la institución.
    - *Entrada/Salida:* Archivo estático inicial que actúa como fuente de origen del proyecto.
    - *Rol:* Datos de partida históricos de diagnósticos y procedimientos.
    - *Edición:* No modificar. Reemplazar solo con nuevos lotes manteniendo las columnas de origen.
*   **CSV de Entrada ([datos_diagnostico.csv](data/datos_diagnostico.csv), [procedimiento_pacientes.csv](data/procedimiento_pacientes.csv))**:
    - *Función:* Datos en bruto delimitados por punto y coma de diagnósticos y procedimientos.
    - *Entrada/Salida:* Insumos directos que consume el script de limpieza inicial.
    - *Rol:* Proveen el historial clínico y las fechas de hospitalización del paciente.
    - *Edición:* Actualizar al ingresar nuevos registros clínicos para preprocesamiento.
*   **[limpieza_datos.py](data/limpieza_datos.py)**:
    - *Función:* Carga, valida formatos de códigos, depura fechas incorrectas y cruza datos.
    - *Entrada/Salida:* Lee CSVs crudos y escribe datasets limpios en `data/processed/`.
    - *Rol:* Script primario encargado de consolidar y curar la muestra de pacientes.
    - *Edición:* Modificar si cambian los límites temporales (ej: LOS < 0) o reglas de validación.
*   **[analisis.py](data/analisis.py)**:
    - *Función:* Genera reportes estadísticos descriptivos estilo `summary` de R para los datos limpios.
    - *Entrada/Salida:* Lee datos maestros y escribe métricas CSV en `data/reports/`.
    - *Rol:* Automatiza la auditoría de calidad de datos y las estadísticas generales.
    - *Edición:* Modificar si se requiere reportar cuantiles o correlaciones adicionales.

#### Subcarpeta `data/processed/` (Datos Limpios)
*   **Tablas de salida ([dataset_maestro.csv](data/processed/dataset_maestro.csv), [caso_diagnostico.csv](data/processed/caso_diagnostico.csv), [caso_procedimiento.csv](data/processed/caso_procedimiento.csv), [pacientes_rechazados.csv](data/processed/pacientes_rechazados.csv))**:
    - *Función:* CSVs limpios a nivel de paciente, a nivel granular de códigos y de registros excluidos.
    - *Entrada/Salida:* Generados automáticamente tras ejecutar `limpieza_datos.py`.
    - *Rol:* Base de datos curada para análisis exploratorio, gráficos y modelamiento predictivo.
    - *Edición:* No modificar manualmente. Re-generar ejecutando la tubería de limpieza.

#### Subcarpeta `data/reports/` (Reportes Estadísticos)
*   **Métricas resultantes ([reporte_limpieza.csv](data/reports/reporte_limpieza.csv), [reporte_estadistico_maestro.csv](data/reports/reporte_estadistico_maestro.csv), [reporte_estadistico_diagnostico.csv](data/reports/reporte_estadistico_diagnostico.csv), [reporte_estadistico_rechazados.csv](data/reports/reporte_estadistico_rechazados.csv))**:
    - *Función:* Tablas estructuradas con tasa de aceptación, promedios, percentiles y motivos de rechazo.
    - *Entrada/Salida:* Generados de forma automática como resultado de ejecutar `analisis.py`.
    - *Rol:* Insumos estadísticos tabulares para el capítulo de caracterización clínica.
    - *Edición:* Consultar para obtener métricas; no editar directamente.

### 3. Directorio `LOS_0/` (Pacientes con Estancia Cero)

*   **Scripts descriptivos ([analisis_los_0.py](LOS_0/analisis_los_0.py), [analisis_correlaciones_los_0.py](LOS_0/analisis_correlaciones_los_0.py))**:
    - *Función:* Caracterizan diagnósticos y evalúan correlaciones estadísticas en casos de alta inmediata.
    - *Entrada/Salida:* Leen `dataset_maestro.csv` y guardan reportes y figuras de coexistencia en `/LOS_0`.
    - *Rol:* Explican los determinantes clínicos detrás de pacientes con estancia de 0 días.
    - *Edición:* Modificar si se requiere cambiar el tipo de coeficiente o gráficos de asociación.
*   **Resultados de soporte ([correlaciones_diagnostico_procedimiento_los_0.csv](LOS_0/correlaciones_diagnostico_procedimiento_los_0.csv), [diagnosticos_detallado_los_0.csv](LOS_0/diagnosticos_detallado_los_0.csv), [procedimientos_detallado_los_0.csv](LOS_0/procedimientos_detallado_los_0.csv))**:
    - *Función:* Tablas de frecuencias y coeficientes de correlación de códigos en LOS = 0.
    - *Entrada/Salida:* Generados por los scripts de análisis de estancia cero.
    - *Rol:* Respaldo cuantitativo de las patologías e intervenciones del subgrupo.
    - *Edición:* Se re-generan corriendo automáticamente los scripts asociados.
*   **Documentación de conclusiones ([RESUMEN_ANALISIS_LOS_0.md](LOS_0/RESUMEN_ANALISIS_LOS_0.md), [RESUMEN_CORRELACIONES_LOS_0.md](LOS_0/RESUMEN_CORRELACIONES_LOS_0.md))**:
    - *Función:* Síntesis explicativa markdown con la interpretación de las altas del mismo día.
    - *Entrada/Salida:* Textos estáticos descriptivos de soporte clínico.
    - *Rol:* Insumos de análisis cualitativo y discusión para el manuscrito final.
    - *Edición:* Editar para actualizar las interpretaciones de los hallazgos médicos.
*   **Gráficos descriptivos de LOS=0 ([01_diagnosticos_procedimientos_los_0.png](LOS_0/01_diagnosticos_procedimientos_los_0.png) al [06_diagnosticos_por_procedimiento_los_0.png](LOS_0/06_diagnosticos_por_procedimiento_los_0.png))**:
    - *Función:* Visualizan el ranking de códigos, el heatmap de coexistencia y distribución de tipos en LOS = 0.
    - *Entrada/Salida:* Gráficos PNG autogenerados por la suite descriptiva de la carpeta.
    - *Rol:* Ilustraciones que complementan la caracterización de altas rápidas.
    - *Edición:* Se actualizan de forma automática al ejecutar los scripts descriptivos correspondientes.

### 4. Directorio `graficos/` (Análisis de Distribución de LOS)

*   **Scripts de modelamiento distributivo ([analisis_distribucion_los.py](graficos/analisis_distribucion_los.py), [visualizacion_los.py](graficos/visualizacion_los.py), [visualizar_weight_verosimilitud.py](graficos/visualizar_weight_verosimilitud.py))**:
    - *Función:* Ajustan mezclas de densidades (Weibull/Log-Normal) y grafican histogramas descriptivos.
    - *Entrada/Salida:* Leen `dataset_maestro.csv` y guardan curvas de ajuste y verosimilitud en `/graficos`.
    - *Rol:* Validan el sesgo del target y ubican la verosimilitud máxima en el weight 0.42.
    - *Edición:* Editar si se desea ensayar otras familias teóricas de ajuste continuo.
*   **Conceptos matemáticos ([README_GRAFICOS.md](graficos/README_GRAFICOS.md), [explicacion_grafico_verosimilitud_mezcla.md](graficos/explicacion_grafico_verosimilitud_mezcla.md))**:
    - *Función:* Textos markdown que explican las fórmulas, la escala logarítmica dual y la mezcla.
    - *Entrada/Salida:* Explicaciones estáticas de soporte estadístico.
    - *Rol:* Facilitan la correcta lectura metodológica de los histogramas de estancia.
    - *Edición:* Modificar si se incorporan nuevos análisis descriptivos de la variable target.
*   **Gráficos de distribución general ([01_distribucion_los_escala_lineal.png](graficos/01_distribucion_los_escala_lineal.png) al [07_verosimilitud_vs_weight.png](graficos/07_verosimilitud_vs_weight.png))**:
    - *Función:* Histogramas (lineal y log1p), percentiles boxplot, Q-Q plots, curvas CDF y PDF.
    - *Entrada/Salida:* PNGs generadas por los scripts de distribución de la carpeta.
    - *Rol:* Exposición visual de la asimetría y el modelado de mezclas teóricas.
    - *Edición:* Se actualizan automáticamente al re-correr la suite de visualización distributiva.

#### Subcarpeta `graficos/diag_proc/` (Outliers y Complejidad Clínica)
*   **Scripts descriptivos ([analisis_codigos_outliers.py](graficos/diag_proc/analisis_codigos_outliers.py), [analisis_complejidad_los.py](graficos/diag_proc/analisis_complejidad_los.py))**:
    - *Función:* Identifican códigos causantes de estancias largas y estudian el LOS vs volumen de códigos.
    - *Entrada/Salida:* Leen `dataset_maestro.csv` y guardan reportes en subcarpetas `/diagnosticos` y `/procedimientos`.
    - *Rol:* Sustentan el impacto de la complejidad clínica sobre el aumento de la estancia.
    - *Edición:* Editar si cambian los umbrales estadísticos para definir registros outliers.
*   **Visualización de Diagnósticos (en `/diagnosticos/`: [CONCLUSIONES.md](graficos/diag_proc/diagnosticos/CONCLUSIONES.md), CSVs e imágenes 01 a 06)**:
    - *Función:* Gráficos descriptivos de patologías outliers, violines de frecuentes y tablas de dispersión.
    - *Entrada/Salida:* Producidos por los scripts de outliers y complejidad clínica.
    - *Rol:* Identifican visualmente qué enfermedades específicas incrementan el LOS.
    - *Edición:* Se re-generan automáticamente al ejecutar el script de análisis de outliers.
*   **Visualización de Procedimientos (en `/procedimientos/`: [CONCLUSIONES.md](graficos/diag_proc/procedimientos/CONCLUSIONES.md), CSVs e imágenes 01 a 05)**:
    - *Función:* Gráficos descriptivos de cirugías complejas (incluyendo copias _pro) y tablas de dispersión.
    - *Entrada/Salida:* Producidos en el pipeline descriptivo quirúrgico de la carpeta.
    - *Rol:* Documentan la variabilidad quirúrgica asociada a colas largas.
    - *Edición:* Se actualizan de forma automatizada al correr el script general de outliers.

### 5. Directorio `ml/` (Modelado Predictivo - Versión 2)

*   **[informe_final_comparativo_modelos.md](ml/informe_final_comparativo_modelos.md)**:
    - *Función:* Comparación del rendimiento de los tres modelos finales en el set de holdout.
    - *Entrada/Salida:* Texto estructurado con métricas oficiales (MAE, RMSE, MedAE, PLOS).
    - *Rol:* Reporte definitivo de la fase final de modelamiento predictivo.
    - *Edición:* Modificar si se actualizan los estimadores o se recalculan resultados en test.
*   **[plan_implementacion_modelos.md](ml/plan_implementacion_modelos.md)**:
    - *Función:* Planificación inicial del diseño de variables y esquemas de validación cruzada.
    - *Entrada/Salida:* Pautas markdown de diseño de ingeniería de características.
    - *Rol:* Estableció los lineamientos de modelamiento de la v2 antes de codificar.
    - *Edición:* Documento histórico; modificar únicamente si se varía la metodología base.

#### Subcarpeta `ml/feature_engineering/` (Ingeniería de Variables)
*   **[comordibipy.py](ml/feature_engineering/comordibipy.py)**:
    - *Función:* Script interactivo que demuestra el funcionamiento de la librería `comorbidipy`.
    - *Entrada/Salida:* Imprime la tabla de pesos y traduce diagnósticos de ejemplo.
    - *Rol:* Auditoría y aprendizaje clínico para validar el cálculo de Charlson y Elixhauser.
    - *Edición:* Modificar los códigos de prueba para auditar el comportamiento con otros diagnósticos.
*   **Scripts de procesamiento ([procesamiento_features_v2.py](ml/feature_engineering/procesamiento_features_v2.py), [procesamiento_features_v3.py](ml/feature_engineering/procesamiento_features_v3.py))**:
    - *Función:* Codifican variables One-Hot, aplican soporte jerárquico y calculan comorbilidades.
    - *Entrada/Salida:* Leen datos maestros limpios y guardan CSVs estructurados en `processed_v2/` y `processed_v3/`.
    - *Rol:* El script v3 construye las variables y escenarios finales de modelado (A, B y C).
    - *Edición:* Editar v3 si se varía el soporte mínimo o la definición de escenarios predictivos.
*   **[analisis_ml_features.md](ml/feature_engineering/analisis_ml_features.md)**:
    - *Función:* Análisis estadístico de la dimensionalidad de las variables binarias resultantes.
    - *Entrada/Salida:* Justificación markdown sobre la agrupación y el soporte mínimo de 20 casos.
    - *Rol:* Respaldo metodológico sobre el control de sobreajuste reduciendo características dispersas.
    - *Edición:* Editar si cambian las justificaciones teóricas del agrupamiento.
*   **Datasets estructurados resultantes (en `/processed_v2/` y `/processed_v3/`)**:
    - *Función:* Matrices predictoras One-Hot por escenarios listas para entrenar algoritmos.
    - *Entrada/Salida:** CSVs resultantes de ejecutar los scripts de ingeniería de variables.
    - *Rol:* Insumos de modelamiento; `model_data_v3_escenario_B_charlson.csv` alimenta al mejor modelo.
    - *Edición:* No modificar a mano. Re-generar corriendo el script de variables definitivo.
*   **[reporte_analisis_comorbilidades_v3.md](ml/feature_engineering/processed_v3/reporte_analisis_comorbilidades_v3.md)**:
    - *Función:* Describe y evalúa el impacto de los índices Charlson y Elixhauser en la estancia.
    - *Entrada/Salida:* Análisis descriptivo del cruce de índices contra días de hospitalización.
    - *Rol:* Respalda la elección de incorporar Charlson en el escenario predictivo óptimo.
    - *Edición:* Modificar si cambia el cálculo o categorización de comorbilidades.
*   **Tablas de trazabilidad y reportes (en `/reports_features/`)**:
    - *Función:* CSVs con mapeos de reemplazo, frecuencias intermedias y repetición de códigos.
    - *Entrada/Salida:* Generados por `procesamiento_features_v2.py` para control de calidad.
    - *Rol:* Auditoría de agregación jerárquica para comprobar el soporte de 20 casos mínimos.
    - *Edición:* Consultar para trazar agrupamientos; no deben modificarse manualmente.

#### Subcarpeta `ml/modelos/` (Entrenamiento, Tuning y Métricas)
*   **[reporte_comparativo_tuning.md](ml/modelos/reporte_comparativo_tuning.md)**:
    - *Función:* Documenta las configuraciones de parámetros y grillas de búsqueda evaluadas.
    - *Entrada/Salida:* Resumen markdown técnico comparativo del tuning.
    - *Rol:* Sustento de la selección hiperparamétrica y regularización del proyecto.
    - *Edición:* Actualizar al probar nuevas grillas de búsqueda en validación cruzada.
*   **Scripts de Tuning ([tuning_random_forest.py](ml/modelos/RF/tuning_random_forest.py), [tuning_rf_regularizado.py](ml/modelos/RF/tuning_rf_regularizado.py), [tuning_xgboost.py](ml/modelos/XGB/tuning_xgboost.py), [tuning_xgboost_regularizado.py](ml/modelos/XGB/tuning_xgboost_regularizado.py))**:
    - *Función:* Realizan Randomized Search CV de 5 folds para optimizar MAE continuo.
    - *Entrada/Salida:* Leen escenario B y escriben los hiperparámetros óptimos en archivos JSON.
    - *Rol:* Los de grilla regularizada son críticos para frenar el sobreajuste masivo en test.
    - *Edición:* Modificar para ensayar rangos o distribuciones paramétricas distintas.
*   **Scripts de Entrenamiento ([entrenar_lr_final.py](ml/modelos/LR/entrenar_lr_final.py), [entrenar_rf_final.py](ml/modelos/RF/entrenar_rf_final.py), [entrenar_xgboost_final.py](ml/modelos/XGB/entrenar_xgboost_final.py))**:
    - *Función:* Ajustan los modelos en train (80%) y evalúan la generalización en holdout (20%).
    - *Entrada/Salida:* Leen escenario B e hiperparámetros y escriben métricas y pkl en subcarpetas `/final`.
    - *Rol:* Generan los estimadores oficiales evaluados sobre datos nunca antes vistos.
    - *Edición:* Editar para ensayar entrenamientos sobre otros escenarios o particiones.
*   **Evaluación del Holdout (en las carpetas `/final/` de LR, RF y XGB)**:
    - *Función:* CSVs de matriz de confusión PLOS, métricas kfold, predicciones y resumen de validación.
    - *Entrada/Salida:* Generados automáticamente tras ejecutar el entrenamiento final correspondiente.
    - *Rol:* Respaldo numérico de las métricas en holdout para armar la comparativa del informe.
    - *Edición:* Re-generar corriendo los scripts de ajuste de modelos definitivos.
*   **Gráficos diagnósticos de holdout (en `/RF/final/` y `/XGB/final/`)**:
    - *Función:* PNGs de MAE/RMSE por tramo, porcentaje de subestimación, residuos histograma y scatter.
    - *Entrada/Salida:* Visualizaciones de diagnóstico resultantes de evaluar los modelos finales.
    - *Rol:* Muestran visualmente el sesgo de regresión a la media por tramos de estancia.
    - *Edición:* Se re-generan automáticamente al re-correr los entrenamientos correspondientes.
*   **Tuning intermedio y validación (en `/RF/` y `/XGB/`)**:
    - *Función:* CSVs con MAE por tramo en validación y predicciones intermedias de escenarios A, B y C.
    - *Entrada/Salida:* Resultados del tuning y validaciones cruzadas preliminares de escenarios.
    - *Rol:* Control numérico intermedio para justificar metodológicamente la elección del escenario B.
    - *Edición:* Se actualizan automáticamente durante las etapas de optimización hiperparamétrica.
*   **[reporte_xgboost_final.md](ml/modelos/XGB/final/reporte_xgboost_final.md)**:
    - *Función:* Reporte analítico detallado del modelo ganador (XGBoost Final V2).
    - *Entrada/Salida:* Análisis de importancia de variables, errores continuos y matrices.
    - *Rol:* Bitácora principal de los resultados definitivos del boosting de la tesis.
    - *Edición:* Modificar si cambian las interpretaciones del modelo ganador o se recalcula test.
*   **Versión 1 preliminar (en subcarpetas `/v1/` de RF y XGB)**:
    - *Función:* Scripts de entrenamiento, matrices, predicciones y reportes no regularizados obsoletos.
    - *Entrada/Salida:* CSVs e informes markdown de rendimiento inicial (sobreajustado).
    - *Rol:* Evidencia del gap masivo de error entre train y test previo a la regularización.
    - *Edición:* Históricos; no modificar.

#### Subcarpeta `ml/graficos_comparativos/` (Métricas Cruzadas de Rendimiento)
*   **Scripts comparativos ([generar_graficos.py](ml/graficos_comparativos/generar_graficos.py), [generar_graficos_3_modelos.py](ml/graficos_comparativos/generar_graficos_3_modelos.py), [generar_graficos_xgb_rf.py](ml/graficos_comparativos/generar_graficos_xgb_rf.py))**:
    - *Función:* Leen los reportes CSV de holdout de los modelos y dibujan curvas comparadas.
    - *Entrada/Salida:* Consumen métricas e imágenes PNG en `/comparativa_3_modelos` y `/xgb_vs_rf`.
    - *Rol:* Generan las figuras oficiales que comparan el desempeño continuo e inferencial de los modelos.
    - *Edición:* Modificar si se desea alterar el diseño de ejes o la paleta cromática de las curvas.
*   **Gráficos consolidados (en la raíz de `/graficos_comparativos/`, `/comparativa_3_modelos/` y `/xgb_vs_rf/`)**:
    - *Función:* PNGs comparativas del 01 al 08 (dispersión, residuos, MAE en tramos, PLOS, subestimación, confusion y tabla).
    - *Entrada/Salida:* Imágenes de salida resultantes de ejecutar los scripts comparativos de la carpeta.
    - *Rol:* Insumos visuales oficiales para justificar la elección de XGBoost como el mejor estimador.
    - *Edición:* Se actualizan al ejecutar el script de comparación correspondiente.

### 6. Directorio `ml2/` (Modelado V1 - Obsoleto)

*   **Scripts y modelos v1 ([entrenar_gradient_boosting_v1.py](ml2/entrenar_gradient_boosting_v1.py) a [entrenar_xgboost_v1.py](ml2/entrenar_xgboost_v1.py))**:
    - *Función:* Entrenamientos iniciales obsoletos en días reales sin técnicas regularizadoras.
    - *Entrada/Salida:* Guardaban pkl en `ml2/models/` y reportes en `ml2/reports_modelos/`.
    - *Rol:* Registro histórico de la primera aproximación de algoritmos de bosque y boosting.
    - *Edición:* Obsoletos; no modificar.
*   **Reportes e insumos v1 (en `ml2/reports_modelos/` y pkl en `ml2/models/`)**:
    - *Función:* Predicciones y métricas descriptivas asociadas al modelamiento inicial sobreajustado.
    - *Entrada/Salida:* CSVs y archivos serializados resultantes de la ejecución de v1.
    - *Rol:* Insumos comparativos históricos del primer baseline lineal y no lineal del proyecto.
    - *Edición:* Obsoletos; no modificar.

---

## 🚀 Guía de Ejecución Paso a Paso (End-to-End)

Para ejecutar la tubería completa de datos y modelos del proyecto en el orden metodológico correcto, ejecuta la siguiente secuencia de comandos desde la raíz del repositorio (usando rutas relativas):

### Paso 1: Limpieza e Integración de Datos
Valida los códigos clínicos de entrada, depura fechas incorrectas y cruza registros.
```bash
python3 data/limpieza_datos.py
```
*   **Entrada:** `data/datos_diagnostico.csv` y `data/procedimiento_pacientes.csv`
*   **Salida Esperada:** Genera `dataset_maestro.csv` y tablas granulares en `data/processed/`, además de `reporte_limpieza.csv` en `data/reports/`.

### Paso 2: Análisis Estadístico Descriptivo (Opcional)
Calcula cuantiles de estancia, distribuciones generales y correlaciones de variables.
```bash
python3 data/analisis.py
```
*   **Entrada:** Datasets en `data/processed/`
*   **Salida Esperada:** Genera los reportes estadísticos estructurados en `data/reports/` (`reporte_estadistico_maestro.csv`, etc.).

### Paso 3: Ingeniería de Variables y Escenarios
Crea las matrices predictoras One-Hot, calcula Charlson/Elixhauser y aplica la transformación logarítmica.
```bash
python3 ml/feature_engineering/procesamiento_features_v3.py
```
*   **Entrada:** Datasets en `data/processed/`
*   **Salida Esperada:** Genera los datasets por escenarios listos para modelar en `ml/feature_engineering/processed_v3/` (`model_data_v3_escenario_B_charlson.csv` y `model_data_v3_escenario_C_elixhauser.csv`), además de `reporte_analisis_comorbilidades_v3.md`.

### Paso 4: Búsqueda de Hiperparámetros (Tuning Regularizado - Opcional)
Ejecuta la búsqueda aleatoria cross-validada con fuertes restricciones de regularización.
```bash
python3 ml/modelos/XGB/tuning_xgboost_regularizado.py
```
*   **Entrada:** `ml/feature_engineering/processed_v3/model_data_v3_escenario_B_charlson.csv`
*   **Salida Esperada:** Actualiza `mejores_hiperparametros_xgboost_regularizado.json` y guarda resultados de búsqueda en `resumen_tuning_xgboost_regularizado.csv`.

### Paso 5: Entrenamiento Final y Holdout
Entrena el modelo ganador XGBoost en train (80%) y evalúa en el holdout set de test (20%).
```bash
python3 ml/modelos/XGB/entrenar_xgboost_final.py
```
*   **Entrada:** Dataset del escenario B y configuración JSON de hiperparámetros.
*   **Salida Esperada:** Genera el modelo serializado `xgboost_final.pkl` y toda la suite de métricas CSV y gráficos PNG diagnósticos en `ml/modelos/XGB/final/`.

*(Nota: Puedes repetir el paso 5 para Random Forest y Regresión Lineal ejecutando `entrenar_rf_final.py` y `entrenar_lr_final.py` en sus respectivas carpetas).*

### Paso 6: Comparación de Rendimiento y Graficación
Genera las visualizaciones consolidadas comparativas para los tres modelos finales.
```bash
python3 ml/graficos_comparativos/generar_graficos_3_modelos.py
```
*   **Entrada:** Archivos de métricas CSV en las carpetas `/final` de LR, RF y XGB.
*   **Salida Esperada:** Genera los gráficos oficiales en `ml/graficos_comparativos/comparativa_3_modelos/` y el informe comparativo consolidado.

---

## 📈 Resumen de Resultados y Modelamiento V2 (Holdout Set)

Al evaluar los modelos definitivos regularizados sobre el **Holdout Test Set (20% de datos nuevos)**, se obtuvieron las siguientes métricas continuas:

| Modelo | Escenario | MAE (días) | RMSE (días) | MedAE (días) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Final V2** | B (Charlson) | **3.057** | **8.290** | **0.902** |
| **Random Forest Final V2** | B (Charlson) | 3.268 | 8.959 | 0.972 |
| **Regresión Lineal Baseline** | B (Charlson) | 6.311 | 53.070 | 0.866 |

### Identificación de Pacientes Críticos (PLOS ≥ 27 días)

| Métrica | Reg. Lineal | RF Final | XGBoost Final |
| :--- | :---: | :---: | :---: |
| **Precision PLOS** | 65.98% | **80.85%** | 78.67% |
| **Recall PLOS** | **52.46%** | 31.15% | 48.36% |
| **F1 PLOS** | 58.45% | 44.97% | **59.90%** |

### Conclusiones de Modelado

1.  **Modelo Ganador:** El mejor modelo es **XGBoost Regularizado sobre el Escenario B**. Controla efectivamente el sobreajuste (Train MAE de 2.60 vs Test MAE de 3.05, reduciendo el gap de la v1 de 2.27 a solo 0.45 días).
2.  **Índice de Charlson:** Su incorporación aporta información clínica valiosa que estabiliza el modelo de boosting, superando al escenario de Elixhauser.
3.  **Comportamiento de Subestimación:** Todos los modelos sufren de regresión a la media en la cola larga de estancias. XGBoost subestima el 81.1% de los casos de más de 27 días, una limitación documentada en la literatura clínica que se propone abordar en trabajo futuro mediante Regresión por Cuantiles.
4.  **Mezcla de Estancias:** El análisis de verosimilitud determinó que los datos reales se explican mediante una mezcla óptima de **42% Log-Normal** (estancias típicas cortas) y **58% Weibull** (casos complejos de cola larga).
