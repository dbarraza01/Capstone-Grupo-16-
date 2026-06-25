# Prediccion de Estancia Hospitalaria (LOS)

Proyecto Capstone para predecir la duracion de estancia hospitalaria (`LOS`, Length of Stay) a partir de datos clinicos estructurados: diagnosticos ICD-10-CM, procedimientos ICD-10-PCS, comorbilidades Charlson/Elixhauser y variables administrativas de ingreso.

El estado vigente del repositorio esta dividido en dos entregas principales:

- `ml_entrega2/`: modelamiento predictivo regularizado historico. Contiene la ingenieria de variables final y los modelos globales LR/RF/XGB usados como base metodologica.
- `ml_operacional_entrega3/`: pipeline operacional actual. Separa pacientes urgentes y programados, entrena modelos por subgrupo y evalua PLOS con umbral actualizado.

En la entrega operacional, `PLOS` se define como:

```text
PLOS = 1 si LOS >= 14 dias
PLOS = 0 si LOS < 14 dias
```

Los tramos vigentes de evaluacion son `0-2`, `3-6`, `7-13` y `14+ (PLOS)`.

## Objetivo Actual

El flujo operacional busca demostrar que los pacientes que ingresan por urgencia y los pacientes programados tienen comportamientos distintos de estancia hospitalaria. Por eso el pipeline actual:

1. Segmenta los datos en `urgente` y `programado`.
2. Entrena modelos separados por segmento.
3. Usa una arquitectura de dos etapas para XGB y RF:
   - Clasificador: estima riesgo `LOS >= 14`.
   - Regresor: predice dias exactos usando variables clinicas mas `prob_los_14`.
4. Evalua resultados en train y holdout para detectar sobreajuste.
5. Reporta metricas globales, por segmento, por tramo y comparativas train vs holdout.

## Estructura Vigente

```text
Capstone-Grupo-16-/
├── README.md
├── implementation_plan.md
├── sensitivity_analysis_plan.md
├── urgencias.md
├── resumen_modelamiento_los_para_equipo_entrega2.md
│
├── data/
│   ├── Datos proyecto LOS.xlsx
│   ├── datos_diagnostico.csv
│   ├── procedimiento_pacientes.csv
│   ├── limpieza_datos.py
│   ├── analisis.py
│   ├── processed/
│   └── reports/
│
├── LOS_0/
│   ├── analisis_los_0.py
│   ├── analisis_correlaciones_los_0.py
│   └── reportes, CSVs y graficos de pacientes con LOS = 0
│
├── graficos/
│   ├── analisis_distribucion_los.py
│   ├── visualizacion_los.py
│   ├── visualizar_weight_verosimilitud.py
│   └── diag_proc/
│
├── ml_entrega2/
│   ├── informe_final_comparativo_modelos.md
│   ├── plan_implementacion_modelos.md
│   ├── feature_engineering/
│   │   ├── procesamiento_features_v2.py
│   │   ├── procesamiento_features_v3.py
│   │   ├── processed_v2/
│   │   ├── processed_v3/
│   │   └── reports_features/
│   ├── modelos/
│   │   ├── LR/
│   │   ├── RF/
│   │   └── XGB/
│   └── graficos_comparativos/
│
├── ml_operacional_entrega3/
│   ├── preparar_datos.py
│   ├── comparar_modelos.py
│   ├── LR/
│   │   ├── entrenar_lr.py
│   │   └── evaluar_lr.py
│   ├── RF/
│   │   ├── tuning_rf.py
│   │   ├── entrenar_rf.py
│   │   ├── evaluar_rf.py
│   │   └── best_params_*.json
│   ├── XGB/
│   │   ├── tuning_xgb.py
│   │   ├── entrenar_xgb.py
│   │   ├── evaluar_xgb.py
│   │   └── best_params_*.json
│   ├── utils/
│   │   ├── pipeline_operacional.py
│   │   ├── model_workflows.py
│   │   └── metricas_operacionales.py
│   ├── data_splits/
│   ├── modelos_guardados/
│   └── reports/
│
├── Web/
│   ├── app.py
│   ├── preprocessing_helper.py
│   ├── requirements.txt
│   └── templates/
│
├── ml2/
│   └── modelamiento v1 historico/obsoleto
│
└── Modelo_Base_Ultima entrega/
    └── material historico de referencia
```

## Carpetas Clave

| Carpeta | Estado | Uso principal |
| --- | --- | --- |
| `data/` | Vigente | Limpieza, integracion y auditoria inicial de datos crudos. |
| `ml_entrega2/` | Vigente como base historica | Feature engineering final y modelos globales regularizados de la entrega 2. |
| `ml_operacional_entrega3/` | Vigente y principal | Modelos operacionales por segmento urgente/programado, reportes train/holdout y comparacion final. |
| `Web/` | Vigente para demo | Aplicacion Flask para prediccion individual, masiva y visualizacion. |
| `ml2/` | Obsoleto | Primera version de modelamiento. Mantener solo como evidencia historica. |
| `Modelo_Base_Ultima entrega/` | Historico | Documentos y analisis previos de referencia. |

## Flujo de Ejecucion Recomendado

Ejecutar desde la raiz del repositorio.

### 1. Limpieza de Datos

```bash
python3 data/limpieza_datos.py
python3 data/analisis.py
```

Salidas principales:

- `data/processed/dataset_maestro.csv`
- `data/processed/caso_diagnostico.csv`
- `data/processed/caso_procedimiento.csv`
- `data/reports/reporte_limpieza.csv`

### 2. Feature Engineering de Entrega 2

Este paso regenera la matriz de variables que consume el pipeline operacional.

```bash
python3 ml_entrega2/feature_engineering/procesamiento_features_v3.py
```

Entrada principal:

- `data/processed/dataset_maestro.csv`

Salida principal:

- `ml_entrega2/feature_engineering/processed_v3/model_data_v3_escenario_B_charlson.csv`

### 3. Preparacion de Splits Operacionales

```bash
python3 ml_operacional_entrega3/preparar_datos.py
```

Genera splits estratificados 80/20 por urgencia y tramo LOS:

- `ml_operacional_entrega3/data_splits/datos_train_urgente.csv`
- `ml_operacional_entrega3/data_splits/datos_train_programado.csv`
- `ml_operacional_entrega3/data_splits/datos_holdout_urgente.csv`
- `ml_operacional_entrega3/data_splits/datos_holdout_programado.csv`

### 4. Tuning Operacional

Solo se debe repetir si cambian los datos, la grilla o la definicion metodologica. Los scripts actuales tienen `FAST_RUN = False`, por lo que ejecutan busqueda completa con `n_iter=50`.

```bash
python3 ml_operacional_entrega3/XGB/tuning_xgb.py
python3 ml_operacional_entrega3/RF/tuning_rf.py
```

Salidas principales:

- `ml_operacional_entrega3/XGB/best_params_clf_urgente.json`
- `ml_operacional_entrega3/XGB/best_params_clf_programado.json`
- `ml_operacional_entrega3/XGB/best_params_reg_urgente.json`
- `ml_operacional_entrega3/XGB/best_params_reg_programado.json`
- `ml_operacional_entrega3/RF/best_params_clf_urgente.json`
- `ml_operacional_entrega3/RF/best_params_clf_programado.json`
- `ml_operacional_entrega3/RF/best_params_reg_urgente.json`
- `ml_operacional_entrega3/RF/best_params_reg_programado.json`

### 5. Entrenamiento Operacional

```bash
python3 ml_operacional_entrega3/XGB/entrenar_xgb.py
python3 ml_operacional_entrega3/RF/entrenar_rf.py
python3 ml_operacional_entrega3/LR/entrenar_lr.py
```

El baseline `LR` corresponde a una Regresion Lineal basica segmentada:

- Se entrena un modelo independiente para `urgente` y otro para `programado`.
- No usa penalizacion ni hiperparametro `alpha`, y no incorpora interacciones manuales.
- Usa los mismos splits operacionales y el mismo conjunto base de features para que la comparacion contra XGB/RF sea directa.

Salidas principales:

- `ml_operacional_entrega3/modelos_guardados/clf_xgb_*.joblib`
- `ml_operacional_entrega3/modelos_guardados/reg_xgb_*.joblib`
- `ml_operacional_entrega3/modelos_guardados/clf_rf_*.joblib`
- `ml_operacional_entrega3/modelos_guardados/reg_rf_*.joblib`
- `ml_operacional_entrega3/modelos_guardados/reg_lr_*.joblib`

### 6. Evaluacion Train y Holdout

```bash
python3 ml_operacional_entrega3/XGB/evaluar_xgb.py
python3 ml_operacional_entrega3/RF/evaluar_rf.py
python3 ml_operacional_entrega3/LR/evaluar_lr.py
```

Cada evaluacion genera:

- Metricas globales train y holdout.
- Metricas por segmento `urgente` y `programado`.
- Metricas por tramo `0-2`, `3-6`, `7-13`, `14+ (PLOS)`.
- Comparacion train vs holdout.
- Predicciones con `plos_real_14` y `plos_pred_14`.

### 7. Comparacion Final

```bash
python3 ml_operacional_entrega3/comparar_modelos.py
```

Salidas principales:

- `ml_operacional_entrega3/reports/comparacion_final_modelos.md`
- `ml_operacional_entrega3/reports/comparacion_final_modelos.csv`
- `ml_operacional_entrega3/reports/comparacion_final_por_segmento.csv`
- `ml_operacional_entrega3/reports/comparacion_final_train_vs_holdout.csv`

## Reportes Operacionales Importantes

| Archivo | Contenido |
| --- | --- |
| `ml_operacional_entrega3/reports/reporte_evaluacion_xgb_holdout.md` | Evaluacion holdout detallada de XGB. |
| `ml_operacional_entrega3/reports/reporte_evaluacion_rf_holdout.md` | Evaluacion holdout detallada de RF. |
| `ml_operacional_entrega3/reports/reporte_evaluacion_lr_holdout.md` | Evaluacion holdout detallada de LR basica. |
| `ml_operacional_entrega3/reports/reporte_evaluacion_xgb_train.md` | Evaluacion train de XGB para revisar sobreajuste. |
| `ml_operacional_entrega3/reports/reporte_evaluacion_rf_train.md` | Evaluacion train de RF para revisar sobreajuste. |
| `ml_operacional_entrega3/reports/reporte_evaluacion_lr_train.md` | Evaluacion train de LR basica para revisar sobreajuste. |
| `ml_operacional_entrega3/reports/comparacion_final_modelos.md` | Comparacion global XGB vs RF vs LR. |
| `ml_operacional_entrega3/reports/comparacion_final_train_vs_holdout.csv` | Gaps train-holdout por modelo y segmento. |

## Resultados Operacionales Actuales

Holdout global con `PLOS = LOS >= 14`:

| Modelo | n casos | MAE | RMSE | MedAE | Precision PLOS 14 | Recall PLOS 14 | F1 PLOS 14 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB | 2391 | 2.8600 | 7.1668 | 0.8359 | 0.8000 | 0.5878 | 0.6777 |
| RF | 2391 | 3.0966 | 8.2899 | 0.9118 | 0.8197 | 0.5376 | 0.6494 |
| LR basica | 2391 | 13.0718 | 181.1970 | 0.8983 | 0.6971 | 0.5197 | 0.5955 |

Holdout por segmento:

| Modelo | Segmento | n casos | MAE | RMSE | Precision PLOS 14 | Recall PLOS 14 | F1 PLOS 14 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB | programado | 1693 | 1.8363 | 5.5007 | 0.8052 | 0.6263 | 0.7045 |
| XGB | urgente | 698 | 5.3429 | 10.1270 | 0.7969 | 0.5667 | 0.6623 |
| RF | programado | 1693 | 2.0470 | 6.1960 | 0.8082 | 0.5960 | 0.6860 |
| RF | urgente | 698 | 5.6423 | 11.9287 | 0.8273 | 0.5056 | 0.6276 |
| LR basica | programado | 1693 | 3.3365 | 15.2619 | 0.7606 | 0.5455 | 0.6353 |
| LR basica | urgente | 698 | 36.6848 | 334.5181 | 0.6642 | 0.5056 | 0.5741 |

Lectura principal:

- XGB es el mejor modelo global por MAE y F1 PLOS.
- RF tiene la mayor precision PLOS global, pero menor recall que XGB.
- LR basica queda como baseline lineal simple e interpretable, pero muestra sobreajuste fuerte al no tener regularizacion: su MAE global holdout sube a 13.0718 dias.
- El segmento urgente presenta mayor error que el programado, lo que respalda tratar ambos grupos por separado.

## Web Demo

La carpeta `Web/` contiene una aplicacion Flask para prediccion individual, prediccion masiva y panel de analitica.

```bash
cd Web
pip install -r requirements.txt
python3 app.py
```

La aplicacion corre por defecto en:

```text
http://localhost:5000
```

## Reglas de Mantenimiento

- No usar rutas `ml/`: esa carpeta fue renombrada a `ml_entrega2/`.
- No usar rutas `ml_operacional/`: esa carpeta fue renombrada a `ml_operacional_entrega3/`.
- Para resultados actuales, priorizar siempre `ml_operacional_entrega3/reports/`.
- Para regenerar features, usar `ml_entrega2/feature_engineering/`.
- `PLOS >= 27` pertenece a reportes/metodologia anterior; en la entrega operacional vigente PLOS es `LOS >= 14`.
- Repetir tuning solo si cambian datos, grillas, umbral PLOS o definicion del pipeline.
- Despues de entrenar modelos, correr siempre los scripts `evaluar_*.py` antes de `comparar_modelos.py`.

## Dependencias Principales

El proyecto usa principalmente:

- Python 3
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- flask, para la aplicacion web

Las dependencias especificas de la demo web estan en `Web/requirements.txt`.
