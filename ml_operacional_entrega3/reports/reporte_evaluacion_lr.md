# Reporte Train vs Holdout LR

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Lectura

Un gap alto entre holdout y train indica riesgo de sobreajuste. Las filas por segmento permiten verificar si urgentes y programados se comportan distinto.

## Metricas Train por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | train | programado | 0 | 6772 | 1.9726 | 8.4226 | 0.6168 | -0.6755 | 0.4668 | 3.2966 | 0.6620 | 0.8925 | 0.9505 | 4.4219 | 3.7464 | 14 | 396 | 235 | 0.9064 | 0.5379 | 0.6751 | 0.9697 | 6354 | 22 | 183 | 213 |
| LR | train | urgente | 1 | 2788 | 5.4992 | 12.1035 | 2.2337 | -2.4251 | 0.5014 | 9.4614 | 0.2590 | 0.5793 | 0.7948 | 11.5897 | 9.1646 | 14 | 721 | 469 | 0.8443 | 0.5492 | 0.6655 | 0.8572 | 1994 | 73 | 325 | 396 |

## Metricas Holdout por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | holdout | programado | 0 | 1693 | 2.0873 | 7.2608 | 0.6044 | -0.6517 | 0.4536 | 3.4568 | 0.6568 | 0.8813 | 0.9480 | 4.3615 | 3.7098 | 14 | 99 | 50 | 0.9400 | 0.4747 | 0.6309 | 0.9675 | 1591 | 3 | 52 | 47 |
| LR | holdout | urgente | 1 | 698 | 5.7846 | 11.8637 | 2.4858 | -1.5074 | 0.5072 | 9.4306 | 0.2493 | 0.5573 | 0.7794 | 11.0387 | 9.5312 | 14 | 180 | 110 | 0.8273 | 0.5056 | 0.6276 | 0.8453 | 499 | 19 | 89 | 91 |

## Gap Train vs Holdout

| modelo | alcance | segmento | metrica | train | holdout | gap_holdout_minus_train | ratio_holdout_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR | global | todos | mae | 3.0010 | 3.1666 | 0.1656 | 1.0552 |
| LR | global | todos | rmse | 9.6423 | 8.8553 | -0.7869 | 0.9184 |
| LR | global | todos | medae | 0.8681 | 0.8637 | -0.0043 | 0.9950 |
| LR | global | todos | me | -1.1858 | -0.9015 | 0.2842 | 0.7603 |
| LR | global | todos | pup | 0.4769 | 0.4693 | -0.0076 | 0.9840 |
| LR | global | todos | mae_asimetrico_alpha_2 | 5.0944 | 5.2007 | 0.1062 | 1.0209 |
| LR | global | todos | precision_plos_14 | 0.8651 | 0.8625 | -0.0026 | 0.9970 |
| LR | global | todos | recall_plos_14 | 0.5452 | 0.4946 | -0.0506 | 0.9072 |
| LR | global | todos | f1_plos_14 | 0.6689 | 0.6287 | -0.0402 | 0.9400 |
| LR | segmento | programado | mae | 1.9726 | 2.0873 | 0.1147 | 1.0581 |
| LR | segmento | programado | rmse | 8.4226 | 7.2608 | -1.1618 | 0.8621 |
| LR | segmento | programado | medae | 0.6168 | 0.6044 | -0.0123 | 0.9800 |
| LR | segmento | programado | me | -0.6755 | -0.6517 | 0.0238 | 0.9648 |
| LR | segmento | programado | pup | 0.4668 | 0.4536 | -0.0131 | 0.9718 |
| LR | segmento | programado | mae_asimetrico_alpha_2 | 3.2966 | 3.4568 | 0.1602 | 1.0486 |
| LR | segmento | programado | precision_plos_14 | 0.9064 | 0.9400 | 0.0336 | 1.0371 |
| LR | segmento | programado | recall_plos_14 | 0.5379 | 0.4747 | -0.0631 | 0.8826 |
| LR | segmento | programado | f1_plos_14 | 0.6751 | 0.6309 | -0.0442 | 0.9345 |
| LR | segmento | urgente | mae | 5.4992 | 5.7846 | 0.2854 | 1.0519 |
| LR | segmento | urgente | rmse | 12.1035 | 11.8637 | -0.2398 | 0.9802 |
| LR | segmento | urgente | medae | 2.2337 | 2.4858 | 0.2521 | 1.1128 |
| LR | segmento | urgente | me | -2.4251 | -1.5074 | 0.9177 | 0.6216 |
| LR | segmento | urgente | pup | 0.5014 | 0.5072 | 0.0057 | 1.0114 |
| LR | segmento | urgente | mae_asimetrico_alpha_2 | 9.4614 | 9.4306 | -0.0308 | 0.9967 |
| LR | segmento | urgente | precision_plos_14 | 0.8443 | 0.8273 | -0.0171 | 0.9798 |
| LR | segmento | urgente | recall_plos_14 | 0.5492 | 0.5056 | -0.0437 | 0.9205 |
| LR | segmento | urgente | f1_plos_14 | 0.6655 | 0.6276 | -0.0380 | 0.9430 |

## Holdout por Segmento y Tramo

| split | segmento | es_urgencia | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | programado | 0 | 0-2 | 983 | 0.5903 | 0.9339 | 0.3989 | 0.4843 | 0.2431 | 0.6433 | 0.8332 | 0.9878 | 0.9980 | 1.2533 | 1.7376 | 14 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 983 | 0 | 0 | 0 |
| holdout | programado | 0 | 3-6 | 501 | 1.1865 | 1.6346 | 0.8560 | -0.5396 | 0.7006 | 2.0495 | 0.5509 | 0.9321 | 0.9960 | 3.9381 | 3.3985 | 14 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 501 | 0 | 0 | 0 |
| holdout | programado | 0 | 7-13 | 110 | 3.8392 | 4.5093 | 3.7837 | -3.3718 | 0.8909 | 7.4447 | 0.1273 | 0.4000 | 0.9000 | 8.9818 | 5.6100 | 14 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9727 | 107 | 3 | 0 | 0 |
| holdout | programado | 0 | 14+ (PLOS) | 99 | 19.5629 | 29.2707 | 12.3362 | -9.4770 | 0.8081 | 34.0828 | 0.0303 | 0.1010 | 0.2626 | 32.2323 | 22.7553 | 14 | 99 | 47 | 1.0000 | 0.4747 | 0.6438 | 0.4747 | 0 | 0 | 52 | 47 |
| holdout | urgente | 1 | 0-2 | 200 | 2.0442 | 3.5416 | 1.3745 | 1.9865 | 0.0800 | 2.0731 | 0.4200 | 0.8100 | 0.9700 | 1.1150 | 3.1015 | 14 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9900 | 198 | 2 | 0 | 0 |
| holdout | urgente | 1 | 3-6 | 179 | 2.0416 | 3.4200 | 1.4967 | 0.5154 | 0.5363 | 2.8047 | 0.3687 | 0.8324 | 0.9609 | 4.2961 | 4.8115 | 14 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9721 | 174 | 5 | 0 | 0 |
| holdout | urgente | 1 | 7-13 | 139 | 3.7518 | 4.6307 | 3.4570 | -1.4666 | 0.6906 | 6.3609 | 0.1439 | 0.4317 | 0.9209 | 9.0647 | 7.5981 | 14 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9137 | 127 | 12 | 0 | 0 |
| holdout | urgente | 1 | 14+ (PLOS) | 180 | 15.2325 | 22.4423 | 11.0993 | -7.4327 | 0.8111 | 26.5651 | 0.0222 | 0.1000 | 0.2778 | 30.2944 | 22.8617 | 14 | 180 | 91 | 1.0000 | 0.5056 | 0.6716 | 0.5056 | 0 | 0 | 89 | 91 |
