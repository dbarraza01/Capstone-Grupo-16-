# Reporte Consolidado de Analisis de Sensibilidad

## 1. Resumen Ejecutivo

Este informe consolida cuatro escenarios de sensibilidad del pipeline XGBoost operacional de dos etapas. Todas las metricas se reportan para tres cohortes: `global`, `urgente` y `programado`, porque el origen de admision cambia la dificultad clinica y el costo operacional de los errores.

### Impacto sobre MAE por cohorte

#### Segmento: Global

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | delta_mae_pct | veredicto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | global | LOS >= 7 | 2.8474 | 2.5919 | 3.1058 | -0.4392 | robusto |
| 1 - Umbral PLOS | global | LOS >= 14 | 2.8600 | 2.6108 | 3.1436 | 0.0000 | robusto |
| 1 - Umbral PLOS | global | LOS >= 21 | 2.8276 | 2.5903 | 3.0967 | -1.1329 | robusto |
| 1 - Umbral PLOS | global | LOS >= 27 | 2.8640 | 2.6183 | 3.1287 | 0.1421 | robusto |
| 2 - Ablation | global | Full (linea base) | 2.8600 | 2.6108 | 3.1436 | 0.0000 | robusto |
| 2 - Ablation | global | Sin Charlson | 2.9109 | 2.6595 | 3.1995 | 1.7823 | robusto |
| 2 - Ablation | global | Sin capitulos ICD-10 | 2.8660 | 2.6140 | 3.1473 | 0.2117 | robusto |
| 2 - Ablation | global | Sin codigos clinicos | 3.4529 | 3.1669 | 3.7816 | 20.7306 | sensibilidad alta |
| 2 - Ablation | global | Sin Clasificador (1 Etapa) | 2.8441 | 2.5985 | 3.1158 | -0.5547 | robusto |
| 4 - Hiperparametros | global | Full (linea base) | 2.8600 | 2.6108 | 3.1436 | 0.0000 | robusto |
| 4 - Hiperparametros | global | Conservadora (mas regularizada) | 2.8752 | 2.6228 | 3.1582 | 0.5308 | robusto |
| 4 - Hiperparametros | global | Compleja (menos regularizada) | 2.8863 | 2.6373 | 3.1714 | 0.9202 | robusto |
| 4 - Hiperparametros | global | Perturbacion estocastica de muestreo | 2.8912 | 2.6514 | 3.1741 | 1.0910 | robusto |

#### Segmento: Urgente

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | delta_mae_pct | veredicto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | urgente | LOS >= 7 | 5.3119 | 4.6986 | 5.9529 | -0.5803 | robusto |
| 1 - Umbral PLOS | urgente | LOS >= 14 | 5.3429 | 4.7226 | 6.0001 | 0.0000 | robusto |
| 1 - Umbral PLOS | urgente | LOS >= 21 | 5.2490 | 4.6698 | 5.9184 | -1.7561 | robusto |
| 1 - Umbral PLOS | urgente | LOS >= 27 | 5.3455 | 4.7672 | 5.9498 | 0.0502 | robusto |
| 2 - Ablation | urgente | Full (linea base) | 5.3429 | 4.7226 | 6.0001 | 0.0000 | robusto |
| 2 - Ablation | urgente | Sin Charlson | 5.4959 | 4.8797 | 6.1699 | 2.8647 | robusto |
| 2 - Ablation | urgente | Sin capitulos ICD-10 | 5.4281 | 4.8314 | 6.0826 | 1.5950 | robusto |
| 2 - Ablation | urgente | Sin codigos clinicos | 5.9351 | 5.2212 | 6.6935 | 11.0849 | sensibilidad moderada |
| 2 - Ablation | urgente | Sin Clasificador (1 Etapa) | 5.2054 | 4.6536 | 5.7748 | -2.5726 | robusto |
| 4 - Hiperparametros | urgente | Full (linea base) | 5.3429 | 4.7226 | 6.0001 | 0.0000 | robusto |
| 4 - Hiperparametros | urgente | Conservadora (mas regularizada) | 5.3313 | 4.7472 | 5.9441 | -0.2172 | robusto |
| 4 - Hiperparametros | urgente | Compleja (menos regularizada) | 5.4781 | 4.7983 | 6.2275 | 2.5313 | robusto |
| 4 - Hiperparametros | urgente | Perturbacion estocastica de muestreo | 5.4117 | 4.8097 | 6.0576 | 1.2887 | robusto |

#### Segmento: Programado

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | delta_mae_pct | veredicto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | programado | LOS >= 7 | 1.8314 | 1.6078 | 2.1089 | -0.2699 | robusto |
| 1 - Umbral PLOS | programado | LOS >= 14 | 1.8363 | 1.6160 | 2.1157 | 0.0000 | robusto |
| 1 - Umbral PLOS | programado | LOS >= 21 | 1.8292 | 1.5948 | 2.0835 | -0.3853 | robusto |
| 1 - Umbral PLOS | programado | LOS >= 27 | 1.8410 | 1.6154 | 2.1011 | 0.2523 | robusto |
| 2 - Ablation | programado | Full (linea base) | 1.8363 | 1.6160 | 2.1157 | 0.0000 | robusto |
| 2 - Ablation | programado | Sin Charlson | 1.8452 | 1.6153 | 2.1158 | 0.4838 | robusto |
| 2 - Ablation | programado | Sin capitulos ICD-10 | 1.8097 | 1.5838 | 2.0803 | -1.4477 | robusto |
| 2 - Ablation | programado | Sin codigos clinicos | 2.4295 | 2.1756 | 2.7167 | 32.3011 | sensibilidad alta |
| 2 - Ablation | programado | Sin Clasificador (1 Etapa) | 1.8706 | 1.6384 | 2.1577 | 1.8659 | robusto |
| 4 - Hiperparametros | programado | Full (linea base) | 1.8363 | 1.6160 | 2.1157 | 0.0000 | robusto |
| 4 - Hiperparametros | programado | Conservadora (mas regularizada) | 1.8625 | 1.6213 | 2.1378 | 1.4281 | robusto |
| 4 - Hiperparametros | programado | Compleja (menos regularizada) | 1.8177 | 1.6030 | 2.0865 | -1.0124 | robusto |
| 4 - Hiperparametros | programado | Perturbacion estocastica de muestreo | 1.8520 | 1.6207 | 2.1325 | 0.8540 | robusto |

### Veredicto cuantitativo 5%

La tolerancia de robustez se aplica de forma independiente para `global` y `urgente`. El segmento urgente se reporta aparte porque concentra mayor riesgo operacional en gestion de camas.

| segmento | max_abs_delta_mae_pct | escenario_mas_sensible | variante_mas_sensible | veredicto_5pct |
| --- | --- | --- | --- | --- |
| global | 20.7306 | 2 - Ablation | Sin codigos clinicos | no robusto |
| urgente | 11.0849 | 2 - Ablation | Sin codigos clinicos | no robusto |

### Analisis de significancia estadistica con bootstrapping

El IC 95% del MAE se calculo por bootstrapping percentil con 1000 remuestreos del holdout. La interpretacion usada es directa: si el intervalo de una variante se solapa con el intervalo de la linea base del mismo escenario y segmento, la diferencia observada puede explicarse por variabilidad muestral; si no se solapa, se reporta como evidencia de cambio estadisticamente relevante en MAE.

Resultado: 2 comparaciones no solapan con la linea base. Esas diferencias deben reportarse como cambios potencialmente significativos en MAE, especialmente si coinciden con deltas superiores a la tolerancia operacional de 5%.

Comparaciones sin solapamiento:

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | mae_base | mae_base_ci_lower | mae_base_ci_upper | solapa_ic_base | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 - Ablation | global | Sin codigos clinicos | 3.4529 | 3.1669 | 3.7816 | 2.8600 | 2.6108 | 3.1436 | False | 20.7306 |
| 2 - Ablation | programado | Sin codigos clinicos | 2.4295 | 2.1756 | 2.7167 | 1.8363 | 1.6160 | 2.1157 | False | 32.3011 |

Tabla de solapamiento de IC 95% contra la linea base:

#### Segmento: Global

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | mae_base | mae_base_ci_lower | mae_base_ci_upper | solapa_ic_base | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | global | LOS >= 7 | 2.8474 | 2.5919 | 3.1058 | 2.8600 | 2.6108 | 3.1436 | True | -0.4392 |
| 1 - Umbral PLOS | global | LOS >= 14 | 2.8600 | 2.6108 | 3.1436 | 2.8600 | 2.6108 | 3.1436 | True | 0.0000 |
| 1 - Umbral PLOS | global | LOS >= 21 | 2.8276 | 2.5903 | 3.0967 | 2.8600 | 2.6108 | 3.1436 | True | -1.1329 |
| 1 - Umbral PLOS | global | LOS >= 27 | 2.8640 | 2.6183 | 3.1287 | 2.8600 | 2.6108 | 3.1436 | True | 0.1421 |
| 2 - Ablation | global | Full (linea base) | 2.8600 | 2.6108 | 3.1436 | 2.8600 | 2.6108 | 3.1436 | True | 0.0000 |
| 2 - Ablation | global | Sin Charlson | 2.9109 | 2.6595 | 3.1995 | 2.8600 | 2.6108 | 3.1436 | True | 1.7823 |
| 2 - Ablation | global | Sin capitulos ICD-10 | 2.8660 | 2.6140 | 3.1473 | 2.8600 | 2.6108 | 3.1436 | True | 0.2117 |
| 2 - Ablation | global | Sin codigos clinicos | 3.4529 | 3.1669 | 3.7816 | 2.8600 | 2.6108 | 3.1436 | False | 20.7306 |
| 2 - Ablation | global | Sin Clasificador (1 Etapa) | 2.8441 | 2.5985 | 3.1158 | 2.8600 | 2.6108 | 3.1436 | True | -0.5547 |
| 4 - Hiperparametros | global | Full (linea base) | 2.8600 | 2.6108 | 3.1436 | 2.8600 | 2.6108 | 3.1436 | True | 0.0000 |
| 4 - Hiperparametros | global | Conservadora (mas regularizada) | 2.8752 | 2.6228 | 3.1582 | 2.8600 | 2.6108 | 3.1436 | True | 0.5308 |
| 4 - Hiperparametros | global | Compleja (menos regularizada) | 2.8863 | 2.6373 | 3.1714 | 2.8600 | 2.6108 | 3.1436 | True | 0.9202 |
| 4 - Hiperparametros | global | Perturbacion estocastica de muestreo | 2.8912 | 2.6514 | 3.1741 | 2.8600 | 2.6108 | 3.1436 | True | 1.0910 |

#### Segmento: Urgente

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | mae_base | mae_base_ci_lower | mae_base_ci_upper | solapa_ic_base | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | urgente | LOS >= 7 | 5.3119 | 4.6986 | 5.9529 | 5.3429 | 4.7226 | 6.0001 | True | -0.5803 |
| 1 - Umbral PLOS | urgente | LOS >= 14 | 5.3429 | 4.7226 | 6.0001 | 5.3429 | 4.7226 | 6.0001 | True | 0.0000 |
| 1 - Umbral PLOS | urgente | LOS >= 21 | 5.2490 | 4.6698 | 5.9184 | 5.3429 | 4.7226 | 6.0001 | True | -1.7561 |
| 1 - Umbral PLOS | urgente | LOS >= 27 | 5.3455 | 4.7672 | 5.9498 | 5.3429 | 4.7226 | 6.0001 | True | 0.0502 |
| 2 - Ablation | urgente | Full (linea base) | 5.3429 | 4.7226 | 6.0001 | 5.3429 | 4.7226 | 6.0001 | True | 0.0000 |
| 2 - Ablation | urgente | Sin Charlson | 5.4959 | 4.8797 | 6.1699 | 5.3429 | 4.7226 | 6.0001 | True | 2.8647 |
| 2 - Ablation | urgente | Sin capitulos ICD-10 | 5.4281 | 4.8314 | 6.0826 | 5.3429 | 4.7226 | 6.0001 | True | 1.5950 |
| 2 - Ablation | urgente | Sin codigos clinicos | 5.9351 | 5.2212 | 6.6935 | 5.3429 | 4.7226 | 6.0001 | True | 11.0849 |
| 2 - Ablation | urgente | Sin Clasificador (1 Etapa) | 5.2054 | 4.6536 | 5.7748 | 5.3429 | 4.7226 | 6.0001 | True | -2.5726 |
| 4 - Hiperparametros | urgente | Full (linea base) | 5.3429 | 4.7226 | 6.0001 | 5.3429 | 4.7226 | 6.0001 | True | 0.0000 |
| 4 - Hiperparametros | urgente | Conservadora (mas regularizada) | 5.3313 | 4.7472 | 5.9441 | 5.3429 | 4.7226 | 6.0001 | True | -0.2172 |
| 4 - Hiperparametros | urgente | Compleja (menos regularizada) | 5.4781 | 4.7983 | 6.2275 | 5.3429 | 4.7226 | 6.0001 | True | 2.5313 |
| 4 - Hiperparametros | urgente | Perturbacion estocastica de muestreo | 5.4117 | 4.8097 | 6.0576 | 5.3429 | 4.7226 | 6.0001 | True | 1.2887 |

#### Segmento: Programado

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | mae_base | mae_base_ci_lower | mae_base_ci_upper | solapa_ic_base | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | programado | LOS >= 7 | 1.8314 | 1.6078 | 2.1089 | 1.8363 | 1.6160 | 2.1157 | True | -0.2699 |
| 1 - Umbral PLOS | programado | LOS >= 14 | 1.8363 | 1.6160 | 2.1157 | 1.8363 | 1.6160 | 2.1157 | True | 0.0000 |
| 1 - Umbral PLOS | programado | LOS >= 21 | 1.8292 | 1.5948 | 2.0835 | 1.8363 | 1.6160 | 2.1157 | True | -0.3853 |
| 1 - Umbral PLOS | programado | LOS >= 27 | 1.8410 | 1.6154 | 2.1011 | 1.8363 | 1.6160 | 2.1157 | True | 0.2523 |
| 2 - Ablation | programado | Full (linea base) | 1.8363 | 1.6160 | 2.1157 | 1.8363 | 1.6160 | 2.1157 | True | 0.0000 |
| 2 - Ablation | programado | Sin Charlson | 1.8452 | 1.6153 | 2.1158 | 1.8363 | 1.6160 | 2.1157 | True | 0.4838 |
| 2 - Ablation | programado | Sin capitulos ICD-10 | 1.8097 | 1.5838 | 2.0803 | 1.8363 | 1.6160 | 2.1157 | True | -1.4477 |
| 2 - Ablation | programado | Sin codigos clinicos | 2.4295 | 2.1756 | 2.7167 | 1.8363 | 1.6160 | 2.1157 | False | 32.3011 |
| 2 - Ablation | programado | Sin Clasificador (1 Etapa) | 1.8706 | 1.6384 | 2.1577 | 1.8363 | 1.6160 | 2.1157 | True | 1.8659 |
| 4 - Hiperparametros | programado | Full (linea base) | 1.8363 | 1.6160 | 2.1157 | 1.8363 | 1.6160 | 2.1157 | True | 0.0000 |
| 4 - Hiperparametros | programado | Conservadora (mas regularizada) | 1.8625 | 1.6213 | 2.1378 | 1.8363 | 1.6160 | 2.1157 | True | 1.4281 |
| 4 - Hiperparametros | programado | Compleja (menos regularizada) | 1.8177 | 1.6030 | 2.0865 | 1.8363 | 1.6160 | 2.1157 | True | -1.0124 |
| 4 - Hiperparametros | programado | Perturbacion estocastica de muestreo | 1.8520 | 1.6207 | 2.1325 | 1.8363 | 1.6160 | 2.1157 | True | 0.8540 |

## 2. Justificacion Metodologica

- La definicion de estancia prolongada no es universal; por eso el Escenario 1 prueba umbrales 7, 14, 21 y 27.
- Los tramos del Escenario 1 son adaptativos: el ultimo tramo siempre corresponde al PLOS definido por el umbral evaluado.
- Los modelos de dos etapas separan una decision binaria de riesgo PLOS de una estimacion continua de dias.
- En datos clinicos desbalanceados, precision, recall y F1 son mas informativos que mirar solo accuracy o MAE.
- La estabilidad frente a hiperparametros vecinos valida si el tuning cae en una region estable o demasiado fragil.

Referencias base: Bergstra y Bengio (2012), Chrusciel et al. (2022), Goldstein et al. (2022), Lee et al. (2024), Mahajan et al. (2023), Probst et al. (2019), Saito y Rehmsmeier (2015).

## 3. Analisis por Escenario

### Escenario 1 - Variacion del umbral PLOS

Pregunta: si cambia la definicion clinica de estancia prolongada, se mantiene estable el desempeno del modelo?

#### Segmento: Global

| umbral_plos | segmento | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | me | pup | mae_asimetrico | precision_plos | recall_plos | f1_plos | proporcion_plos | n_plos_real | n_plos_pred | delta_mae_pct_vs_umbral_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | global | 2391 | 2.8474 | 2.5919 | 3.1058 | 7.0502 | -1.0002 | 0.4316 | 4.7712 | 0.8400 | 0.6364 | 0.7241 | 0.2208 | 528 | 400 | -0.4392 |
| 14 | global | 2391 | 2.8600 | 2.6108 | 3.1436 | 7.1668 | -0.9624 | 0.4471 | 4.7712 | 0.8000 | 0.5878 | 0.6777 | 0.1167 | 279 | 205 | 0.0000 |
| 21 | global | 2391 | 2.8276 | 2.5903 | 3.0967 | 7.0460 | -0.9516 | 0.4517 | 4.7171 | 0.7890 | 0.4914 | 0.6056 | 0.0732 | 175 | 109 | -1.1329 |
| 27 | global | 2391 | 2.8640 | 2.6183 | 3.1287 | 7.0406 | -1.0043 | 0.4626 | 4.7982 | 0.7468 | 0.4836 | 0.5871 | 0.0510 | 122 | 79 | 0.1421 |

#### Segmento: Urgente

| umbral_plos | segmento | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | me | pup | mae_asimetrico | precision_plos | recall_plos | f1_plos | proporcion_plos | n_plos_real | n_plos_pred | delta_mae_pct_vs_umbral_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | urgente | 698 | 5.3119 | 4.6986 | 5.9529 | 9.8842 | -1.9178 | 0.5158 | 8.9267 | 0.8452 | 0.6677 | 0.7461 | 0.4570 | 319 | 252 | -0.5803 |
| 14 | urgente | 698 | 5.3429 | 4.7226 | 6.0001 | 10.1270 | -1.8225 | 0.5172 | 8.9255 | 0.7969 | 0.5667 | 0.6623 | 0.2579 | 180 | 128 | 0.0000 |
| 21 | urgente | 698 | 5.2490 | 4.6698 | 5.9184 | 9.8174 | -1.7822 | 0.5072 | 8.7646 | 0.8281 | 0.4690 | 0.5989 | 0.1619 | 113 | 64 | -1.7561 |
| 27 | urgente | 698 | 5.3455 | 4.7672 | 5.9498 | 9.9875 | -1.8294 | 0.5186 | 8.9330 | 0.7826 | 0.4737 | 0.5902 | 0.1089 | 76 | 46 | 0.0502 |

#### Segmento: Programado

| umbral_plos | segmento | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | me | pup | mae_asimetrico | precision_plos | recall_plos | f1_plos | proporcion_plos | n_plos_real | n_plos_pred | delta_mae_pct_vs_umbral_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | programado | 1693 | 1.8314 | 1.6078 | 2.1089 | 5.4698 | -0.6219 | 0.3969 | 3.0580 | 0.8311 | 0.5885 | 0.6891 | 0.1234 | 209 | 148 | -0.2699 |
| 14 | programado | 1693 | 1.8363 | 1.6160 | 2.1157 | 5.5007 | -0.6078 | 0.4182 | 3.0584 | 0.8052 | 0.6263 | 0.7045 | 0.0585 | 99 | 77 | 0.0000 |
| 21 | programado | 1693 | 1.8292 | 1.5948 | 2.0835 | 5.5116 | -0.6091 | 0.4288 | 3.0484 | 0.7333 | 0.5323 | 0.6168 | 0.0366 | 62 | 45 | -0.3853 |
| 27 | programado | 1693 | 1.8410 | 1.6154 | 2.1011 | 5.3741 | -0.6642 | 0.4395 | 3.0935 | 0.6970 | 0.5000 | 0.5823 | 0.0272 | 46 | 33 | 0.2523 |

Desglose por tramos adaptativos de LOS:

#### Segmento: Global

| umbral_plos_analizado | segmento | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | proporcion_plos | n_plos_real | n_plos_pred | precision_plos | recall_plos | f1_plos | accuracy_plos | tp_plos | fp_plos | fn_plos | tn_plos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | global | 0-2 | 1183 | 0.7850 | 1.5713 | 0.4549 | 0.6874 | 0.1395 | 0.8338 | 0.7937 | 0.9577 | 0.9941 | 1.2299 | 1.9174 | 7 | 0.0000 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9899 | 0 | 12 | 0 | 1171 |
| 7 | global | 3-6 | 680 | 1.6100 | 2.6564 | 1.0556 | -0.2221 | 0.6912 | 2.5261 | 0.4779 | 0.8838 | 0.9706 | 4.0324 | 3.8103 | 7 | 0.0000 | 0 | 52 | 0.0000 | 0.0000 | 0.0000 | 0.9235 | 0 | 52 | 0 | 628 |
| 7 | global | 7+ (PLOS) | 528 | 9.0619 | 14.5074 | 5.7871 | -5.7836 | 0.7519 | 16.4846 | 0.0777 | 0.2576 | 0.5928 | 20.6288 | 14.8452 | 7 | 1.0000 | 528 | 336 | 1.0000 | 0.6364 | 0.7778 | 0.6364 | 336 | 0 | 192 | 0 |
| 14 | global | 0-2 | 1183 | 0.7935 | 1.6208 | 0.4754 | 0.7041 | 0.1716 | 0.8382 | 0.7946 | 0.9637 | 0.9924 | 1.2299 | 1.9341 | 14 | 0.0000 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9966 | 0 | 4 | 0 | 1179 |
| 14 | global | 3-6 | 680 | 1.5665 | 2.6294 | 0.9859 | -0.2235 | 0.6868 | 2.4616 | 0.5044 | 0.8912 | 0.9721 | 4.0324 | 3.8089 | 14 | 0.0000 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9824 | 0 | 12 | 0 | 668 |
| 14 | global | 7-13 | 249 | 4.0996 | 4.9236 | 3.6032 | -1.7626 | 0.7189 | 7.0307 | 0.1004 | 0.4016 | 0.8755 | 9.0281 | 7.2655 | 14 | 0.0000 | 0 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.8996 | 0 | 25 | 0 | 224 |
| 14 | global | 14+ (PLOS) | 279 | 13.6682 | 19.7624 | 10.4781 | -9.1154 | 0.7885 | 25.0600 | 0.0573 | 0.1649 | 0.3226 | 30.9821 | 21.8667 | 14 | 1.0000 | 279 | 164 | 1.0000 | 0.5878 | 0.7404 | 0.5878 | 164 | 0 | 115 | 0 |
| 21 | global | 0-5 | 1772 | 0.9452 | 1.7175 | 0.5816 | 0.3955 | 0.3505 | 1.2200 | 0.7060 | 0.9554 | 0.9893 | 2.0604 | 2.4559 | 21 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 0 | 1772 |
| 21 | global | 6-12 | 326 | 3.8573 | 4.8173 | 3.1705 | -1.3832 | 0.7423 | 6.4776 | 0.1166 | 0.4509 | 0.8957 | 8.0123 | 6.6291 | 21 | 0.0000 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9847 | 0 | 5 | 0 | 321 |
| 21 | global | 13-20 | 118 | 8.3159 | 10.4087 | 8.0259 | -2.7261 | 0.6949 | 13.8370 | 0.0678 | 0.1864 | 0.4661 | 16.0593 | 13.3332 | 21 | 0.0000 | 0 | 18 | 0.0000 | 0.0000 | 0.0000 | 0.8475 | 0 | 18 | 0 | 100 |
| 21 | global | 21+ (PLOS) | 175 | 16.2691 | 23.0685 | 12.2739 | -12.5913 | 0.7714 | 30.6993 | 0.0400 | 0.1314 | 0.2800 | 39.6057 | 27.0144 | 21 | 1.0000 | 175 | 86 | 1.0000 | 0.4914 | 0.6590 | 0.4914 | 86 | 0 | 89 | 0 |
| 27 | global | 0-6 | 1863 | 1.0657 | 1.9401 | 0.6134 | 0.3445 | 0.3752 | 1.4263 | 0.6694 | 0.9367 | 0.9882 | 2.2528 | 2.5973 | 27 | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9995 | 0 | 1 | 0 | 1862 |
| 27 | global | 7-15 | 293 | 4.8454 | 6.2917 | 4.1428 | -1.7677 | 0.7406 | 8.1519 | 0.0922 | 0.3379 | 0.7986 | 9.8464 | 8.0788 | 27 | 0.0000 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.9761 | 0 | 7 | 0 | 286 |
| 27 | global | 16-26 | 113 | 9.7264 | 11.2849 | 9.8412 | -5.7354 | 0.7876 | 17.4573 | 0.0619 | 0.1327 | 0.3540 | 20.3097 | 14.5744 | 27 | 0.0000 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.8938 | 0 | 12 | 0 | 101 |
| 27 | global | 27+ (PLOS) | 122 | 19.2105 | 26.4760 | 14.9487 | -15.3865 | 0.8279 | 36.5090 | 0.0410 | 0.1066 | 0.2869 | 46.8197 | 31.4331 | 27 | 1.0000 | 122 | 59 | 1.0000 | 0.4836 | 0.6519 | 0.4836 | 59 | 0 | 63 | 0 |

#### Segmento: Urgente

| umbral_plos_analizado | segmento | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | proporcion_plos | n_plos_real | n_plos_pred | precision_plos | recall_plos | f1_plos | accuracy_plos | tp_plos | fp_plos | fn_plos | tn_plos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | urgente | 0-2 | 200 | 1.9092 | 3.2646 | 1.1410 | 1.8060 | 0.1050 | 1.9608 | 0.4650 | 0.8300 | 0.9700 | 1.1150 | 2.9210 | 7 | 0.0000 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.9550 | 0 | 9 | 0 | 191 |
| 7 | urgente | 3-6 | 179 | 2.3881 | 3.8696 | 1.4837 | 0.6363 | 0.5642 | 3.2640 | 0.3296 | 0.7933 | 0.9330 | 4.2961 | 4.9324 | 7 | 0.0000 | 0 | 30 | 0.0000 | 0.0000 | 0.0000 | 0.8324 | 0 | 30 | 0 | 149 |
| 7 | urgente | 7+ (PLOS) | 319 | 9.0858 | 14.0956 | 5.9001 | -5.6857 | 0.7461 | 16.4715 | 0.0721 | 0.2508 | 0.5831 | 21.0439 | 15.3582 | 7 | 1.0000 | 319 | 213 | 1.0000 | 0.6677 | 0.8008 | 0.6677 | 213 | 0 | 106 | 0 |
| 14 | urgente | 0-2 | 200 | 1.8946 | 3.2613 | 1.0597 | 1.7941 | 0.1050 | 1.9449 | 0.4850 | 0.8400 | 0.9650 | 1.1150 | 2.9091 | 14 | 0.0000 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9850 | 0 | 3 | 0 | 197 |
| 14 | urgente | 3-6 | 179 | 2.3943 | 3.9707 | 1.4590 | 0.6311 | 0.5810 | 3.2759 | 0.3855 | 0.7765 | 0.9330 | 4.2961 | 4.9272 | 14 | 0.0000 | 0 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.9553 | 0 | 8 | 0 | 171 |
| 14 | urgente | 7-13 | 139 | 3.8934 | 4.6539 | 3.4850 | -1.2728 | 0.6691 | 6.4764 | 0.1007 | 0.4245 | 0.8993 | 9.0647 | 7.7919 | 14 | 0.0000 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.8921 | 0 | 15 | 0 | 124 |
| 14 | urgente | 14+ (PLOS) | 180 | 13.2257 | 18.8007 | 10.4777 | -8.7054 | 0.7944 | 24.1913 | 0.0611 | 0.1500 | 0.3167 | 30.2944 | 21.5891 | 14 | 1.0000 | 180 | 102 | 1.0000 | 0.5667 | 0.7234 | 0.5667 | 102 | 0 | 78 | 0 |
| 21 | urgente | 0-5 | 345 | 1.8217 | 2.9990 | 1.1600 | 1.1246 | 0.2986 | 2.1703 | 0.4551 | 0.8522 | 0.9594 | 2.2841 | 3.4086 | 21 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 0 | 345 |
| 21 | urgente | 6-12 | 163 | 4.0339 | 5.0029 | 3.3935 | -0.5575 | 0.6687 | 6.3296 | 0.1043 | 0.4110 | 0.8957 | 8.1840 | 7.6265 | 21 | 0.0000 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9877 | 0 | 2 | 0 | 161 |
| 21 | urgente | 13-20 | 77 | 7.7917 | 9.6357 | 6.8786 | -3.3290 | 0.7403 | 13.3520 | 0.0649 | 0.1948 | 0.5065 | 15.9610 | 12.6321 | 21 | 0.0000 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.8831 | 0 | 9 | 0 | 68 |
| 21 | urgente | 21+ (PLOS) | 113 | 15.7331 | 21.6453 | 12.4937 | -11.3693 | 0.7522 | 29.2843 | 0.0265 | 0.1239 | 0.2743 | 38.5310 | 27.1617 | 21 | 1.0000 | 113 | 53 | 1.0000 | 0.4690 | 0.6386 | 0.4690 | 53 | 0 | 60 | 0 |
| 27 | urgente | 0-6 | 379 | 2.1032 | 3.4952 | 1.2671 | 1.2213 | 0.3272 | 2.5442 | 0.4090 | 0.8153 | 0.9525 | 2.6174 | 3.8387 | 27 | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9974 | 0 | 1 | 0 | 378 |
| 27 | urgente | 7-15 | 169 | 4.8885 | 6.3583 | 4.2557 | -1.3954 | 0.6864 | 8.0305 | 0.0769 | 0.3373 | 0.7988 | 10.0118 | 8.6164 | 27 | 0.0000 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9763 | 0 | 4 | 0 | 165 |
| 27 | urgente | 16-26 | 74 | 8.9187 | 10.6602 | 8.7607 | -6.3198 | 0.8378 | 16.5380 | 0.0811 | 0.1757 | 0.4324 | 20.5541 | 14.2342 | 27 | 0.0000 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9324 | 0 | 5 | 0 | 69 |
| 27 | urgente | 27+ (PLOS) | 76 | 19.0517 | 25.5861 | 15.0487 | -13.6353 | 0.7895 | 35.3952 | 0.0395 | 0.1184 | 0.2500 | 46.0526 | 32.4173 | 27 | 1.0000 | 76 | 36 | 1.0000 | 0.4737 | 0.6429 | 0.4737 | 36 | 0 | 40 | 0 |

#### Segmento: Programado

| umbral_plos_analizado | segmento | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | proporcion_plos | n_plos_real | n_plos_pred | precision_plos | recall_plos | f1_plos | accuracy_plos | tp_plos | fp_plos | fn_plos | tn_plos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | programado | 0-2 | 983 | 0.5563 | 0.8961 | 0.4002 | 0.4599 | 0.1465 | 0.6045 | 0.8606 | 0.9837 | 0.9990 | 1.2533 | 1.7132 | 7 | 0.0000 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9969 | 0 | 3 | 0 | 980 |
| 7 | programado | 3-6 | 501 | 1.3320 | 2.0562 | 0.8981 | -0.5288 | 0.7365 | 2.2624 | 0.5309 | 0.9162 | 0.9840 | 3.9381 | 3.4094 | 7 | 0.0000 | 0 | 22 | 0.0000 | 0.0000 | 0.0000 | 0.9561 | 0 | 22 | 0 | 479 |
| 7 | programado | 7+ (PLOS) | 209 | 9.0254 | 15.1143 | 5.5047 | -5.9331 | 0.7608 | 16.5047 | 0.0861 | 0.2679 | 0.6077 | 19.9952 | 14.0621 | 7 | 1.0000 | 209 | 123 | 1.0000 | 0.5885 | 0.7410 | 0.5885 | 123 | 0 | 86 | 0 |
| 14 | programado | 0-2 | 983 | 0.5695 | 0.9987 | 0.4323 | 0.4824 | 0.1851 | 0.6130 | 0.8576 | 0.9888 | 0.9980 | 1.2533 | 1.7357 | 14 | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9990 | 0 | 1 | 0 | 982 |
| 14 | programado | 3-6 | 501 | 1.2708 | 1.9368 | 0.9066 | -0.5288 | 0.7246 | 2.1706 | 0.5469 | 0.9321 | 0.9860 | 3.9381 | 3.4093 | 14 | 0.0000 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9920 | 0 | 4 | 0 | 497 |
| 14 | programado | 7-13 | 110 | 4.3602 | 5.2447 | 3.9220 | -2.3814 | 0.7818 | 7.7311 | 0.1000 | 0.3727 | 0.8455 | 8.9818 | 6.6004 | 14 | 0.0000 | 0 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.9091 | 0 | 10 | 0 | 100 |
| 14 | programado | 14+ (PLOS) | 99 | 14.4727 | 21.4005 | 10.4859 | -9.8608 | 0.7778 | 26.6395 | 0.0505 | 0.1919 | 0.3333 | 32.2323 | 22.3715 | 14 | 1.0000 | 99 | 62 | 1.0000 | 0.6263 | 0.7702 | 0.6263 | 62 | 0 | 37 | 0 |
| 21 | programado | 0-5 | 1427 | 0.7333 | 1.2201 | 0.4954 | 0.2193 | 0.3630 | 0.9903 | 0.7666 | 0.9804 | 0.9965 | 2.0063 | 2.2256 | 21 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 0 | 1427 |
| 21 | programado | 6-12 | 163 | 3.6807 | 4.6243 | 3.0269 | -2.2089 | 0.8160 | 6.6256 | 0.1288 | 0.4908 | 0.8957 | 7.8405 | 5.6316 | 21 | 0.0000 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9816 | 0 | 3 | 0 | 160 |
| 21 | programado | 13-20 | 41 | 9.3005 | 11.7237 | 9.3266 | -1.5940 | 0.6098 | 14.7478 | 0.0732 | 0.1707 | 0.3902 | 16.2439 | 14.6499 | 21 | 0.0000 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.7805 | 0 | 9 | 0 | 32 |
| 21 | programado | 21+ (PLOS) | 62 | 17.2460 | 25.4584 | 11.6140 | -14.8187 | 0.8065 | 33.2784 | 0.0645 | 0.1452 | 0.2903 | 41.5645 | 26.7459 | 21 | 1.0000 | 62 | 33 | 1.0000 | 0.5323 | 0.6947 | 0.5323 | 33 | 0 | 29 | 0 |
| 27 | programado | 0-6 | 1484 | 0.8008 | 1.2670 | 0.5127 | 0.1206 | 0.3875 | 1.1409 | 0.7358 | 0.9677 | 0.9973 | 2.1597 | 2.2803 | 27 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 0 | 1484 |
| 27 | programado | 7-15 | 124 | 4.7866 | 6.1998 | 4.1254 | -2.2750 | 0.8145 | 8.3174 | 0.1129 | 0.3387 | 0.7984 | 9.6210 | 7.3460 | 27 | 0.0000 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9758 | 0 | 3 | 0 | 121 |
| 27 | programado | 16-26 | 39 | 11.2588 | 12.3839 | 10.3845 | -4.6264 | 0.6923 | 19.2014 | 0.0256 | 0.0513 | 0.2051 | 19.8462 | 15.2198 | 27 | 0.0000 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.8205 | 0 | 7 | 0 | 32 |
| 27 | programado | 27+ (PLOS) | 46 | 19.4728 | 27.8841 | 13.9904 | -18.2799 | 0.8913 | 38.3492 | 0.0435 | 0.0870 | 0.3478 | 48.0870 | 29.8070 | 27 | 1.0000 | 46 | 23 | 1.0000 | 0.5000 | 0.6667 | 0.5000 | 23 | 0 | 23 | 0 |

Interpretacion:

- Global: mejor MAE con LOS >= 21 (MAE=2.828); mayor desviacion vs umbral 14 con LOS >= 21 (delta=-1.1%). 4/4 umbrales quedan dentro de +/-5%.
- Urgente: mejor MAE con LOS >= 21 (MAE=5.249); mayor desviacion vs umbral 14 con LOS >= 21 (delta=-1.8%). 4/4 umbrales quedan dentro de +/-5%.
- Programado: mejor MAE con LOS >= 21 (MAE=1.829); mayor desviacion vs umbral 14 con LOS >= 21 (delta=-0.4%). 4/4 umbrales quedan dentro de +/-5%.

#### Glosario de metricas - Escenario 1

- `umbral_plos`: definicion usada para estancia prolongada; por ejemplo, 14 significa LOS real o predicho >= 14 dias.
- `segmento`: cohorte evaluada; `global` concatena todos los pacientes, `urgente` usa admisiones urgentes y `programado` usa admisiones no urgentes.
- `n_casos`: numero de pacientes evaluados en la fila.
- `mae`: error absoluto medio en dias; menor es mejor.
- `mae_ci_lower` / `mae_ci_upper`: limites inferior y superior del IC 95% del MAE calculado por bootstrapping.
- `rmse`: raiz del error cuadratico medio; penaliza mas los errores grandes.
- `medae`: mediana del error absoluto; muestra el error tipico robusto a outliers.
- `me`: error medio firmado, calculado como prediccion menos valor real; negativo indica subestimacion promedio.
- `pup`: proporcion de pacientes subestimados, es decir, casos donde el LOS predicho fue menor que el LOS real.
- `mae_asimetrico`: MAE que penaliza mas la subestimacion, usando alpha=2.
- `precision_plos`: de los pacientes marcados como PLOS por el modelo, proporcion que realmente era PLOS.
- `recall_plos`: de los pacientes realmente PLOS, proporcion detectada por el modelo.
- `f1_plos`: balance entre precision y recall para deteccion PLOS.
- `accuracy_plos`: proporcion total de clasificaciones PLOS/no-PLOS correctas.
- `proporcion_plos`: fraccion de pacientes reales PLOS dentro de la fila.
- `n_plos_real`: numero de pacientes cuyo LOS real cumple el umbral PLOS.
- `n_plos_pred`: numero de pacientes cuyo LOS predicho cumple el umbral PLOS.
- `tp_plos`, `fp_plos`, `fn_plos`, `tn_plos`: verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos para clasificacion PLOS.
- `pct_error_abs_le_1d`, `pct_error_abs_le_3d`, `pct_error_abs_le_7d`: proporcion de pacientes con error absoluto menor o igual a 1, 3 o 7 dias.
- `los_real_promedio` / `los_pred_promedio`: promedio del LOS observado y del LOS predicho.
- `delta_mae_pct_vs_umbral_14`: cambio porcentual del MAE respecto al umbral base LOS >= 14 dentro del mismo segmento.
- `tramo`: intervalo adaptativo de LOS real usado para analizar si el desempeno cambia segun estancias cortas, intermedias o PLOS.

#### Interpretacion academica - Escenario 1

Este escenario responde una pregunta central: si cambiamos la definicion de estancia prolongada, el modelo se vuelve inestable o mantiene un comportamiento parecido? La respuesta general es que el modelo se mantiene estable. Al probar umbrales de 7, 14, 21 y 27 dias, el MAE cambia muy poco en los tres segmentos. En terminos practicos, esto significa que el desempeno del pipeline no depende fragilmente de haber elegido exactamente 14 dias como punto de corte. La decision de usar LOS >= 14 sigue siendo clinicamente razonable, pero el modelo no se derrumba si se evalua con otras definiciones de PLOS.

El resultado tambien muestra que el problema no es igual para todos los pacientes. En programados, el MAE se mantiene alrededor de 1.8 dias, mientras que en urgentes esta sobre 5 dias. Esto demuestra que los pacientes urgentes son considerablemente mas dificiles de predecir. No es un fallo aislado del algoritmo, sino una senal esperable: la admision urgente suele involucrar mayor incertidumbre clinica, trayectorias menos planificadas y mayor variabilidad en la evolucion hospitalaria.

El desglose por tramos confirma una idea importante para la gestion hospitalaria: el modelo funciona mucho mejor en estancias cortas que en estancias prolongadas. Para los tramos bajos, gran parte de los pacientes queda con errores pequenos; en cambio, en los tramos PLOS el error aumenta bastante y aparece subestimacion. Esto quiere decir que el modelo es util para la mayoria de los pacientes, que suelen tener estancias menores, pero debe usarse con mayor cautela cuando se trata de pacientes de larga estancia. En esos casos, no basta con mirar solo la prediccion puntual de dias; conviene mirar tambien la alerta PLOS y las politicas de riesgo.

La principal conclusion de sensibilidad es que el umbral PLOS cambia la lectura clinica de la alerta, pero no altera drasticamente la capacidad del modelo para estimar LOS. Por eso, el umbral de 14 dias puede defenderse como una definicion operacional y clinica, no como una condicion artificial que fuerza el buen resultado del modelo. El analisis demuestra robustez frente a distintas definiciones de estancia prolongada.

### Escenario 2 - Ablation study

Pregunta: que componentes son indispensables y que se pierde al removerlos?

#### Segmento: Global

| segmento | variante | pregunta | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | Full (linea base) | Referencia completa del pipeline en dos etapas. | 2391 | 2.8600 | 2.6108 | 3.1436 | 7.1668 | 0.5878 | 0.6777 | 0.8000 | 4.7712 | 0.0000 |
| global | Sin Charlson | Mide la dependencia del indice de comorbilidad Charlson. | 2391 | 2.9109 | 2.6595 | 3.1995 | 7.3094 | 0.5806 | 0.6750 | 0.8060 | 4.8493 | 1.7823 |
| global | Sin capitulos ICD-10 | Mide el aporte marginal de agrupaciones raras por capitulo. | 2391 | 2.8660 | 2.6140 | 3.1473 | 7.1804 | 0.6057 | 0.6969 | 0.8204 | 4.7881 | 0.2117 |
| global | Sin codigos clinicos | Mide cuanto se pierde al remover codigos clinicos detallados. | 2391 | 3.4529 | 3.1669 | 3.7816 | 8.1420 | 0.4659 | 0.5791 | 0.7647 | 5.8839 | 20.7306 |
| global | Sin Clasificador (1 Etapa) | Mide si la probabilidad PLOS de la etapa 1 aporta valor neto. | 2391 | 2.8441 | 2.5985 | 3.1158 | 6.9792 | 0.5986 | 0.7046 | 0.8564 | 4.7981 | -0.5547 |

#### Segmento: Urgente

| segmento | variante | pregunta | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urgente | Full (linea base) | Referencia completa del pipeline en dos etapas. | 698 | 5.3429 | 4.7226 | 6.0001 | 10.1270 | 0.5667 | 0.6623 | 0.7969 | 8.9255 | 0.0000 |
| urgente | Sin Charlson | Mide la dependencia del indice de comorbilidad Charlson. | 698 | 5.4959 | 4.8797 | 6.1699 | 10.3714 | 0.5667 | 0.6602 | 0.7907 | 9.1301 | 2.8647 |
| urgente | Sin capitulos ICD-10 | Mide el aporte marginal de agrupaciones raras por capitulo. | 698 | 5.4281 | 4.8314 | 6.0826 | 10.1982 | 0.5833 | 0.6840 | 0.8268 | 9.0340 | 1.5950 |
| urgente | Sin codigos clinicos | Mide cuanto se pierde al remover codigos clinicos detallados. | 698 | 5.9351 | 5.2212 | 6.6935 | 11.5315 | 0.4889 | 0.5946 | 0.7586 | 10.0841 | 11.0849 |
| urgente | Sin Clasificador (1 Etapa) | Mide si la probabilidad PLOS de la etapa 1 aporta valor neto. | 698 | 5.2054 | 4.6536 | 5.7748 | 9.4009 | 0.5889 | 0.6928 | 0.8413 | 8.7958 | -2.5726 |

#### Segmento: Programado

| segmento | variante | pregunta | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| programado | Full (linea base) | Referencia completa del pipeline en dos etapas. | 1693 | 1.8363 | 1.6160 | 2.1157 | 5.5007 | 0.6263 | 0.7045 | 0.8052 | 3.0584 | 0.0000 |
| programado | Sin Charlson | Mide la dependencia del indice de comorbilidad Charlson. | 1693 | 1.8452 | 1.6153 | 2.1158 | 5.5773 | 0.6061 | 0.7018 | 0.8333 | 3.0844 | 0.4838 |
| programado | Sin capitulos ICD-10 | Mide el aporte marginal de agrupaciones raras por capitulo. | 1693 | 1.8097 | 1.5838 | 2.0803 | 5.4714 | 0.6465 | 0.7191 | 0.8101 | 3.0377 | -1.4477 |
| programado | Sin codigos clinicos | Mide cuanto se pierde al remover codigos clinicos detallados. | 1693 | 2.4295 | 2.1756 | 2.7167 | 6.2289 | 0.4242 | 0.5490 | 0.7778 | 4.1523 | 32.3011 |
| programado | Sin Clasificador (1 Etapa) | Mide si la probabilidad PLOS de la etapa 1 aporta valor neto. | 1693 | 1.8706 | 1.6384 | 2.1577 | 5.6882 | 0.6162 | 0.7262 | 0.8841 | 3.1499 | 1.8659 |

Interpretacion:

- Global: la ablacion mas sensible es `Sin codigos clinicos` (MAE=3.453, delta=20.7%). La variante sin clasificador obtiene MAE=2.844 (delta=-0.6%).
- Urgente: la ablacion mas sensible es `Sin codigos clinicos` (MAE=5.935, delta=11.1%). La variante sin clasificador obtiene MAE=5.205 (delta=-2.6%).
- Programado: la ablacion mas sensible es `Sin codigos clinicos` (MAE=2.429, delta=32.3%). La variante sin clasificador obtiene MAE=1.871 (delta=1.9%).

#### Discusion de la Paradoja de Stacking (2 Etapas vs. 1 Etapa)

La variante `Sin Clasificador (1 Etapa)` obtiene en el holdout global un MAE ligeramente menor que la `Full (linea base)` de dos etapas. En los resultados actuales, el MAE global pasa de 2.860 dias en dos etapas a 2.844 dias en una etapa. Esta diferencia es pequena en magnitud, pero metodologicamente relevante porque evidencia una paradoja de stacking: agregar una prediccion intermedia no garantiza mejorar una metrica continua como MAE.

En el pipeline completo, el clasificador estima una probabilidad de PLOS y esa probabilidad entra al regresor. Como la etapa 1 fue optimizada para una tarea binaria y no mediante joint-tuning con la etapa 2, cualquier ruido de calibracion, error de ranking o sesgo de probabilidad puede propagarse en cascada al estimador de dias. En otras palabras, la etapa 1 entrega una senal clinicamente interpretable, pero esa senal tambien puede introducir error propagation cuando el objetivo final evaluado es una prediccion continua.

La defensa de la arquitectura de dos etapas no debe basarse solo en MAE. El enfoque de dos etapas provee una probabilidad explicita de riesgo PLOS, lo que permite ajustar politicas clinicas de alerta como en el Escenario 3. Un regresor puro de una etapa entrega una estimacion puntual de dias, pero no entrega de forma natural una perilla operacional de sensibilidad/especificidad para decidir cuantas alertas aceptar, cuantos falsos negativos tolerar o cuantas camas bloquear preventivamente.

Al comparar metricas adicionales, la superioridad no es uniforme. La una etapa mejora el MAE global y urgente, pero la dos etapas mantiene mejor MAE y MAE asimetrico en programados. Por eso el resultado correcto no es `una etapa siempre gana`, sino que existe un trade-off: la una etapa puede ser mas limpia para error continuo, mientras la dos etapas entrega una salida probabilistica accionable para gestion hospitalaria.

Resumen cuantitativo por segmento:

| segmento | mae_2_etapas | mae_1_etapa | ganador_mae | rmse_2_etapas | rmse_1_etapa | ganador_rmse | mae_asimetrico_2_etapas | mae_asimetrico_1_etapa | ganador_mae_asimetrico | precision_plos_2_etapas | precision_plos_1_etapa | ganador_precision_plos | recall_plos_2_etapas | recall_plos_1_etapa | ganador_recall_plos | f1_plos_2_etapas | f1_plos_1_etapa | ganador_f1_plos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | 2.8600 | 2.8441 | 1 etapa | 7.1668 | 6.9792 | 1 etapa | 4.7712 | 4.7981 | 2 etapas | 0.8000 | 0.8564 | 1 etapa | 0.5878 | 0.5986 | 1 etapa | 0.6777 | 0.7046 | 1 etapa |
| urgente | 5.3429 | 5.2054 | 1 etapa | 10.1270 | 9.4009 | 1 etapa | 8.9255 | 8.7958 | 1 etapa | 0.7969 | 0.8413 | 1 etapa | 0.5667 | 0.5889 | 1 etapa | 0.6623 | 0.6928 | 1 etapa |
| programado | 1.8363 | 1.8706 | 2 etapas | 5.5007 | 5.6882 | 2 etapas | 3.0584 | 3.1499 | 2 etapas | 0.8052 | 0.8841 | 1 etapa | 0.6263 | 0.6162 | 2 etapas | 0.7045 | 0.7262 | 1 etapa |

#### Significancia estadistica de las diferencias de MAE

El IC 95% del MAE se calculo por bootstrapping percentil con 1000 remuestreos del holdout. La interpretacion usada es directa: si el intervalo de una variante se solapa con el intervalo de la linea base del mismo escenario y segmento, la diferencia observada puede explicarse por variabilidad muestral; si no se solapa, se reporta como evidencia de cambio estadisticamente relevante en MAE.

Resultado: 2 comparaciones no solapan con la linea base. Esas diferencias deben reportarse como cambios potencialmente significativos en MAE, especialmente si coinciden con deltas superiores a la tolerancia operacional de 5%.

Comparaciones sin solapamiento:

| escenario | segmento | variante | mae | mae_ci_lower | mae_ci_upper | mae_base | mae_base_ci_lower | mae_base_ci_upper | solapa_ic_base | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 - Ablation | global | Sin codigos clinicos | 3.4529 | 3.1669 | 3.7816 | 2.8600 | 2.6108 | 3.1436 | False | 20.7306 |
| 2 - Ablation | programado | Sin codigos clinicos | 2.4295 | 2.1756 | 2.7167 | 1.8363 | 1.6160 | 2.1157 | False | 32.3011 |

#### Glosario de metricas - Escenario 2

- `variante`: configuracion evaluada en la ablacion; cada variante reentrena modelos temporales bajo esa condicion.
- `pregunta`: objetivo metodologico de la variante.
- `Full (linea base)`: pipeline completo de dos etapas usado como referencia.
- `Sin Charlson`: reentrenamiento sin la variable `charlson_index`.
- `Sin capitulos ICD-10`: reentrenamiento sin agrupaciones raras por capitulos ICD-10.
- `Sin codigos clinicos`: reentrenamiento removiendo codigos clinicos detallados y dejando solo variables resumidas.
- `Sin Clasificador (1 Etapa)`: reentrenamiento de un regresor directo sin usar la probabilidad PLOS de la etapa 1.
- `mae`: error absoluto medio en dias; menor es mejor.
- `mae_ci_lower` / `mae_ci_upper`: limites inferior y superior del IC 95% del MAE calculado por bootstrapping.
- `rmse`: raiz del error cuadratico medio; aumenta cuando hay errores grandes.
- `recall_plos`: capacidad para detectar pacientes realmente PLOS.
- `precision_plos`: confiabilidad de las alertas PLOS emitidas por el modelo.
- `f1_plos`: balance entre precision y recall PLOS.
- `mae_asimetrico`: error absoluto con penalizacion mayor para subestimacion.
- `delta_mae_pct`: cambio porcentual del MAE respecto a `Full (linea base)` dentro del mismo segmento.
- `mae_base`: MAE de la linea base contra la que se compara la variante.
- `mae_base_ci_lower` / `mae_base_ci_upper`: IC 95% del MAE de la linea base.
- `solapa_ic_base`: indica si el IC 95% de la variante se solapa con el IC 95% de la linea base; `False` sugiere cambio estadisticamente relevante en MAE.
- `ganador_*`: indica si una metrica favorece al modelo de dos etapas o al modelo de una etapa.

#### Interpretacion academica - Escenario 2

Este escenario busca entender que partes del modelo son realmente importantes. La conclusion mas fuerte es que los codigos clinicos detallados son el componente mas critico del sistema. Cuando se remueven, el MAE aumenta de forma importante, especialmente en el segmento programado. Esto demuestra que el modelo no esta prediciendo LOS solo por patrones generales como urgencia, numero de diagnosticos o numero de procedimientos. Esta aprendiendo informacion clinica especifica contenida en diagnosticos y procedimientos codificados.

La variante sin Charlson empeora poco. Esto no significa que la comorbilidad no importe clinicamente, sino que en este dataset el efecto del indice Charlson probablemente queda parcialmente capturado por otras variables clinicas. Es decir, si el paciente tiene diagnosticos complejos, esos diagnosticos ya entregan parte de la informacion que Charlson resume. Por eso, remover Charlson no destruye el desempeno.

La variante sin capitulos ICD-10 tambien cambia poco el resultado. Esto sugiere que esas agrupaciones raras agregan informacion marginal, pero no son el soporte principal del modelo. El modelo parece depender mas de la informacion clinica detallada completa que de estas agrupaciones especificas.

El resultado mas delicado es la comparacion entre dos etapas y una etapa. La variante de una etapa tiene un MAE global levemente menor que la linea base de dos etapas. Sin embargo, los intervalos de confianza se solapan, por lo que no se puede afirmar que sea una mejora estadisticamente clara. Academicamente, esto es importante: no debemos concluir que el modelo de una etapa es superior solo porque su MAE puntual sea menor. La diferencia puede estar dentro de la variabilidad natural del holdout.

La arquitectura de dos etapas sigue teniendo valor porque entrega algo que la regresion directa no entrega de manera natural: una probabilidad explicita de riesgo PLOS. Esa probabilidad permite definir politicas hospitalarias flexibles. Por ejemplo, una clinica puede elegir capturar mas pacientes de riesgo aunque aumenten las falsas alertas, o puede elegir alertas mas precisas para no bloquear camas innecesariamente. Por lo tanto, la decision no es puramente estadistica; es operacional. El modelo de una etapa puede ser competitivo para estimar dias, pero el modelo de dos etapas es mas util para transformar la prediccion en decisiones de gestion de camas.

El analisis de significancia refuerza esta lectura. Las diferencias relevantes y estadisticamente mas claras aparecen cuando se eliminan los codigos clinicos, no cuando se elimina el clasificador. Por lo tanto, la conclusion principal es que la informacion clinica detallada sostiene el rendimiento predictivo, mientras que el clasificador aporta principalmente una capa de decision operacional.

### Escenario 3 - Punto de operacion del clasificador

Pregunta: que politica de alerta conviene para gestion de camas en cada origen de admision?

#### Segmento: Global

| segmento | politica_clinica | umbral_probabilidad | tp | fp | fn | tn | precision | recall | f1 | accuracy | camas_bloqueadas_fp | pacientes_plos_perdidos_fn | fp_por_fn | promedio_dias_subestimados_fn | promedio_dias_sobrestimados_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | Politica B - Alta Seguridad / Alto Recall | 0.3500 | 245 | 229 | 34 | 1883 | 0.5169 | 0.8781 | 0.6507 | 0.8900 | 229 | 34 | 6.7353 | 15.0153 | 4.3280 |
| global | Politica A - Base / Equilibrio | 0.5000 | 229 | 148 | 50 | 1964 | 0.6074 | 0.8208 | 0.6982 | 0.9172 | 148 | 50 | 2.9600 | 15.5198 | 5.3252 |
| global | Politica C - Eficiencia / Alertas Confiables | 0.6500 | 204 | 94 | 75 | 2018 | 0.6846 | 0.7312 | 0.7071 | 0.9293 | 94 | 75 | 1.2533 | 14.1642 | 6.5762 |

#### Segmento: Urgente

| segmento | politica_clinica | umbral_probabilidad | tp | fp | fn | tn | precision | recall | f1 | accuracy | camas_bloqueadas_fp | pacientes_plos_perdidos_fn | fp_por_fn | promedio_dias_subestimados_fn | promedio_dias_sobrestimados_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urgente | Politica B - Alta Seguridad / Alto Recall | 0.3500 | 153 | 129 | 27 | 389 | 0.5426 | 0.8500 | 0.6623 | 0.7765 | 129 | 27 | 4.7778 | 15.0232 | 4.8658 |
| urgente | Politica A - Base / Equilibrio | 0.5000 | 145 | 83 | 35 | 435 | 0.6360 | 0.8056 | 0.7108 | 0.8309 | 83 | 35 | 2.3714 | 15.1042 | 6.0029 |
| urgente | Politica C - Eficiencia / Alertas Confiables | 0.6500 | 124 | 46 | 56 | 472 | 0.7294 | 0.6889 | 0.7086 | 0.8539 | 46 | 56 | 0.8214 | 13.4380 | 7.8433 |

#### Segmento: Programado

| segmento | politica_clinica | umbral_probabilidad | tp | fp | fn | tn | precision | recall | f1 | accuracy | camas_bloqueadas_fp | pacientes_plos_perdidos_fn | fp_por_fn | promedio_dias_subestimados_fn | promedio_dias_sobrestimados_fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| programado | Politica B - Alta Seguridad / Alto Recall | 0.3500 | 92 | 100 | 7 | 1494 | 0.4792 | 0.9293 | 0.6323 | 0.9368 | 100 | 7 | 14.2857 | 14.9848 | 3.6343 |
| programado | Politica A - Base / Equilibrio | 0.5000 | 84 | 65 | 15 | 1529 | 0.5638 | 0.8485 | 0.6774 | 0.9527 | 65 | 15 | 4.3333 | 16.4894 | 4.4600 |
| programado | Politica C - Eficiencia / Alertas Confiables | 0.6500 | 80 | 48 | 19 | 1546 | 0.6250 | 0.8081 | 0.7048 | 0.9604 | 48 | 19 | 2.5263 | 16.3044 | 5.3619 |

Interpretacion del balance FP/FN:

- Global: mejor F1 con `Politica C - Eficiencia / Alertas Confiables` (umbral=0.65, F1=0.707, FP=94, FN=75, dias_FN=14.164, dias_FP=6.576). Mayor recall con `Politica B - Alta Seguridad / Alto Recall` (recall=0.878, FN=34); mayor precision con `Politica C - Eficiencia / Alertas Confiables` (precision=0.685, FP=94).
- Urgente: mejor F1 con `Politica A - Base / Equilibrio` (umbral=0.50, F1=0.711, FP=83, FN=35, dias_FN=15.104, dias_FP=6.003). Mayor recall con `Politica B - Alta Seguridad / Alto Recall` (recall=0.850, FN=27); mayor precision con `Politica C - Eficiencia / Alertas Confiables` (precision=0.729, FP=46).
- Programado: mejor F1 con `Politica C - Eficiencia / Alertas Confiables` (umbral=0.65, F1=0.705, FP=48, FN=19, dias_FN=16.304, dias_FP=5.362). Mayor recall con `Politica B - Alta Seguridad / Alto Recall` (recall=0.929, FN=7); mayor precision con `Politica C - Eficiencia / Alertas Confiables` (precision=0.625, FP=48).

Recomendacion operacional segmentada:
- Urgente: usar como punto inicial `Politica A - Base / Equilibrio` (umbral=0.50) porque equilibra FP=83 y FN=35; esos FN tienen un descalce promedio de 15.104 dias. Si la clinica prioriza no perder PLOS urgentes, `Politica B - Alta Seguridad / Alto Recall` reduce FN a 27, aceptando FP=129.
- Programado: usar `Politica C - Eficiencia / Alertas Confiables` (umbral=0.65) porque reduce camas bloqueadas por falsos positivos a FP=48, con FN=19; las falsas alertas tienen un descalce promedio de 5.362 dias. En cirugia o admisiones planificadas, esta politica evita sobrerreservar camas cuando la agenda puede ajustarse con mas anticipacion.

#### Glosario de metricas - Escenario 3

- `politica_clinica`: regla operacional evaluada para transformar la probabilidad PLOS en alerta.
- `umbral_probabilidad`: probabilidad minima requerida para emitir alerta PLOS; valores bajos aumentan recall y valores altos aumentan precision.
- `tp`: verdaderos positivos; pacientes reales PLOS correctamente alertados.
- `fp`: falsos positivos; pacientes no-PLOS alertados como PLOS.
- `fn`: falsos negativos; pacientes reales PLOS no alertados.
- `tn`: verdaderos negativos; pacientes no-PLOS correctamente no alertados.
- `precision`: proporcion de alertas emitidas que fueron correctas.
- `recall`: proporcion de pacientes PLOS reales detectados por la politica.
- `f1`: balance entre precision y recall.
- `accuracy`: proporcion total de clasificaciones correctas.
- `camas_bloqueadas_fp`: interpretacion operacional de `fp`; posibles camas reservadas innecesariamente por falsas alertas.
- `pacientes_plos_perdidos_fn`: interpretacion operacional de `fn`; pacientes PLOS que no recibieron alerta temprana.
- `fp_por_fn`: razon entre falsas alertas y pacientes PLOS perdidos; ayuda a cuantificar el costo de aumentar seguridad.
- `promedio_dias_subestimados_fn`: error absoluto medio en dias solo sobre falsos negativos; estima el descalce promedio de planificacion por paciente PLOS no detectado.
- `promedio_dias_sobrestimados_fp`: error absoluto medio en dias solo sobre falsos positivos; estima los dias promedio asociados a falsas alertas.

#### Interpretacion academica - Escenario 3

Este escenario muestra por que la probabilidad del clasificador es operacionalmente valiosa. El modelo no solo entrega una prediccion de dias; tambien permite decidir que tan sensible o estricta debe ser la alerta de estancia prolongada. En un hospital, esa decision no es trivial: perder un paciente PLOS puede generar falta de cama, retrasos o mala planificacion; pero alertar demasiados pacientes puede bloquear recursos innecesariamente.

Las tres politicas representan tres estilos de gestion. La politica de alta seguridad usa un umbral bajo y detecta mas pacientes PLOS, pero genera mas falsos positivos. La politica eficiente usa un umbral alto y reduce falsas alertas, pero deja pasar mas pacientes PLOS. La politica intermedia busca equilibrio. Esto demuestra que no existe un unico umbral universalmente mejor; el mejor punto de operacion depende del costo que la clinica asigna a equivocarse en cada direccion.

El analisis segmentado es especialmente importante. En urgentes, la politica intermedia aparece como un punto razonable porque mantiene un buen balance entre pacientes detectados y falsas alertas. Si la clinica quiere ser mas conservadora en urgencias, puede bajar el umbral y aceptar mas falsas alertas para perder menos pacientes PLOS. Esto tiene sentido clinico: los urgentes son mas inciertos y mas dificiles de planificar, por lo que puede ser preferible tolerar mas alertas para evitar sorpresas en camas.

En programados, la politica eficiente resulta mas atractiva. Los pacientes programados suelen tener mayor anticipacion y mejor planificacion inicial, por lo que bloquear demasiadas camas por falsas alertas puede ser costoso. En este segmento, usar un umbral mas alto reduce falsos positivos y mantiene un nivel razonable de deteccion. Esto demuestra que urgente y programado no deberian tratarse con la misma politica de alerta.

Las métricas expresadas en días permiten cuantificar el impacto clínico de los errores de clasificación. Los falsos negativos presentan descalces promedio cercanos a 15 días, por lo que la ausencia de una alerta PLOS puede traducirse en una diferencia de varias jornadas de hospitalización no previstas. Los falsos positivos presentan errores promedio menores, pero también son relevantes porque pueden inducir una sobrerreserva de camas.

La conclusion de este escenario es que el clasificador convierte el modelo en una herramienta de decision. No obliga a usar un umbral fijo; permite adaptar la politica al contexto hospitalario. Esto es precisamente la ventaja practica de la arquitectura de dos etapas: aunque la regresion de una etapa sea competitiva en MAE, no ofrece con la misma claridad una salida probabilistica para ajustar la tolerancia al riesgo.

### Escenario 4 - Hiperparametros vecinos

Pregunta: el tuning encontrado esta en una region estable o fragil?

#### Segmento: Global

| segmento | variante_hiperparametros | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | Full (linea base) | 2391 | 2.8600 | 2.6108 | 3.1436 | 7.1668 | 0.5878 | 0.6777 | 0.8000 | 4.7712 | 0.0000 |
| global | Conservadora (mas regularizada) | 2391 | 2.8752 | 2.6228 | 3.1582 | 7.1739 | 0.5914 | 0.6832 | 0.8088 | 4.8205 | 0.5308 |
| global | Compleja (menos regularizada) | 2391 | 2.8863 | 2.6373 | 3.1714 | 7.3926 | 0.6201 | 0.7047 | 0.8160 | 4.8192 | 0.9202 |
| global | Perturbacion estocastica de muestreo | 2391 | 2.8912 | 2.6514 | 3.1741 | 7.1004 | 0.5771 | 0.6736 | 0.8090 | 4.8313 | 1.0910 |

#### Segmento: Urgente

| segmento | variante_hiperparametros | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urgente | Full (linea base) | 698 | 5.3429 | 4.7226 | 6.0001 | 10.1270 | 0.5667 | 0.6623 | 0.7969 | 8.9255 | 0.0000 |
| urgente | Conservadora (mas regularizada) | 698 | 5.3313 | 4.7472 | 5.9441 | 9.9600 | 0.5778 | 0.6731 | 0.8062 | 8.9127 | -0.2172 |
| urgente | Compleja (menos regularizada) | 698 | 5.4781 | 4.7983 | 6.2275 | 10.8418 | 0.6111 | 0.6962 | 0.8088 | 9.1691 | 2.5313 |
| urgente | Perturbacion estocastica de muestreo | 698 | 5.4117 | 4.8097 | 6.0576 | 9.9212 | 0.5611 | 0.6601 | 0.8016 | 9.0293 | 1.2887 |

#### Segmento: Programado

| segmento | variante_hiperparametros | n_casos | mae | mae_ci_lower | mae_ci_upper | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| programado | Full (linea base) | 1693 | 1.8363 | 1.6160 | 2.1157 | 5.5007 | 0.6263 | 0.7045 | 0.8052 | 3.0584 | 0.0000 |
| programado | Conservadora (mas regularizada) | 1693 | 1.8625 | 1.6213 | 2.1378 | 5.6376 | 0.6162 | 0.7011 | 0.8133 | 3.1333 | 1.4281 |
| programado | Compleja (menos regularizada) | 1693 | 1.8177 | 1.6030 | 2.0865 | 5.3590 | 0.6364 | 0.7200 | 0.8289 | 3.0257 | -1.0124 |
| programado | Perturbacion estocastica de muestreo | 1693 | 1.8520 | 1.6207 | 2.1325 | 5.5335 | 0.6061 | 0.6977 | 0.8219 | 3.1006 | 0.8540 |

Interpretacion:

- Global: mayor variacion con `Perturbacion estocastica de muestreo` (MAE=2.891, delta=1.1%). 4/4 variantes quedan dentro de +/-5%.
- Urgente: mayor variacion con `Compleja (menos regularizada)` (MAE=5.478, delta=2.5%). 4/4 variantes quedan dentro de +/-5%.
- Programado: mayor variacion con `Conservadora (mas regularizada)` (MAE=1.863, delta=1.4%). 4/4 variantes quedan dentro de +/-5%.

#### Glosario de metricas - Escenario 4

- `variante_hiperparametros`: configuracion vecina del tuning evaluada para medir estabilidad.
- `Full (linea base)`: configuracion original usada como referencia dentro del escenario.
- `Conservadora (mas regularizada)`: variante con menor complejidad o mayor regularizacion para reducir sobreajuste.
- `Compleja (menos regularizada)`: variante con mayor complejidad o menor regularizacion para probar sensibilidad a sobreajuste.
- `Perturbacion estocastica de muestreo`: variante que modifica parametros de muestreo como `subsample` o `colsample_bytree`.
- `mae`: error absoluto medio en dias; menor es mejor.
- `mae_ci_lower` / `mae_ci_upper`: limites inferior y superior del IC 95% del MAE calculado por bootstrapping.
- `rmse`: raiz del error cuadratico medio; penaliza errores grandes.
- `recall_plos`: proporcion de pacientes PLOS reales detectados.
- `precision_plos`: proporcion de alertas PLOS correctas.
- `f1_plos`: balance entre precision y recall PLOS.
- `mae_asimetrico`: MAE con penalizacion mayor de subestimacion.
- `delta_mae_pct`: cambio porcentual del MAE respecto a la configuracion base del mismo segmento.

#### Interpretacion academica - Escenario 4

Este escenario evalua si el buen resultado del modelo depende de una combinacion exacta y fragil de hiperparametros. La conclusion es favorable: al mover el modelo hacia configuraciones mas conservadoras, mas complejas o con cambios en muestreo, el MAE se mantiene dentro de variaciones pequenas. En todos los segmentos, las diferencias quedan bajo el margen de 5%, y los intervalos de confianza se solapan con la linea base.

Esto significa que el tuning no parece haber encontrado un punto accidental. Si un pequeno cambio en hiperparametros produjera una gran caida de desempeno, el modelo seria fragil y dificil de defender academicamente. En cambio, los resultados sugieren que el modelo se encuentra en una zona relativamente estable: distintas configuraciones cercanas producen desempenos parecidos.

La estabilidad es relevante porque aumenta la confianza en la generalizacion dentro del holdout. No prueba por si sola que el modelo funcionara igual en otra clinica o en otro periodo temporal, pero si reduce la preocupacion de que el resultado sea producto de una configuracion demasiado especifica. En terminos academicos, esto fortalece la robustez interna del pipeline.

Tambien se observa que los cambios de hiperparametros no modifican sustancialmente el comportamiento entre segmentos. Urgentes sigue siendo el grupo mas dificil, programados sigue siendo mas estable y el global queda en un punto intermedio. Esto indica que las diferencias entre cohortes provienen mas de la naturaleza clinica de los pacientes que de una sensibilidad tecnica del algoritmo.

La conclusion practica es que el modelo puede defenderse como tecnicamente estable frente a perturbaciones razonables de hiperparametros. Las mayores debilidades del sistema no parecen estar en el tuning, sino en la disponibilidad y calidad de informacion clinica detallada, y en la dificultad inherente de predecir pacientes urgentes o de larga estancia.

## 4. Limitaciones y Mejoras Propuestas

- Los escenarios no re-tunean hiperparametros; aislan el efecto de cada perturbacion, pero no miden el mejor rendimiento posible bajo cada nuevo supuesto.
- La sensibilidad debe leerse por cohorte. Un resultado global puede ocultar degradacion en urgentes, que es el segmento operacionalmente mas critico.
- El Escenario 2 remueve bloques completos de informacion; una sensibilidad alta al modo sin codigos clinicos no es una falla del modelo, sino evidencia de que las variables clinicas detalladas sostienen rendimiento.
- En urgencias, el veredicto 5% queda como `no robusto` porque la ablacion sin codigos clinicos supera la tolerancia; la mejora principal es no simplificar el bloque clinico, y en segundo lugar calibrar umbrales del clasificador por segmento.
- El punto de operacion del clasificador debe elegirse con gestion de camas: falsos positivos bloquean capacidad; falsos negativos dejan pacientes PLOS sin alerta temprana.
- Una validacion externa en otra clinica o periodo temporal sigue siendo necesaria para probar generalizacion fuera del holdout actual.

## 5. Veredicto de Robustez y Uso Clinico

El modelo debe defenderse como una solucion operacional con dos salidas complementarias: dias esperados de LOS para planificacion cuantitativa y probabilidad PLOS para alerta temprana. La evaluacion segmentada evita concluir desde el promedio global cuando urgentes y programados tienen riesgos, prevalencias y costos de error distintos.

| segmento | max_abs_delta_mae_pct | escenario_mas_sensible | variante_mas_sensible | veredicto_5pct |
| --- | --- | --- | --- | --- |
| global | 20.7306 | 2 - Ablation | Sin codigos clinicos | no robusto |
| urgente | 11.0849 | 2 - Ablation | Sin codigos clinicos | no robusto |

Recomendacion operacional segmentada:
- Urgente: usar como punto inicial `Politica A - Base / Equilibrio` (umbral=0.50) porque equilibra FP=83 y FN=35; esos FN tienen un descalce promedio de 15.104 dias. Si la clinica prioriza no perder PLOS urgentes, `Politica B - Alta Seguridad / Alto Recall` reduce FN a 27, aceptando FP=129.
- Programado: usar `Politica C - Eficiencia / Alertas Confiables` (umbral=0.65) porque reduce camas bloqueadas por falsos positivos a FP=48, con FN=19; las falsas alertas tienen un descalce promedio de 5.362 dias. En cirugia o admisiones planificadas, esta politica evita sobrerreservar camas cuando la agenda puede ajustarse con mas anticipacion.
