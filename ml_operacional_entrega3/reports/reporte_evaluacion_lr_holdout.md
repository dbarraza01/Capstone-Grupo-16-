# Reporte de Evaluacion LR - holdout

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Metricas Globales

| Metrica | Valor |
|---|---:|
| n_casos | 2391 |
| mae | 3.1666 |
| rmse | 8.8553 |
| medae | 0.8637 |
| me | -0.9015 |
| pup | 46.93% |
| mae_asimetrico_alpha_2 | 5.2007 |
| precision_plos_14 | 86.25% |
| recall_plos_14 | 49.46% |
| f1_plos_14 | 62.87% |
| accuracy_plos_14 | 93.18% |
| tp_plos_14 | 138.0000 |
| fp_plos_14 | 22.0000 |
| fn_plos_14 | 141.0000 |
| tn_plos_14 | 2090.0000 |
| pct_error_abs_le_1d | 53.79% |
| pct_error_abs_le_3d | 78.67% |
| pct_error_abs_le_7d | 89.88% |

## Metricas por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | holdout | programado | 0 | 1693 | 2.0873 | 7.2608 | 0.6044 | -0.6517 | 0.4536 | 3.4568 | 0.6568 | 0.8813 | 0.9480 | 4.3615 | 3.7098 | 14 | 99 | 50 | 0.9400 | 0.4747 | 0.6309 | 0.9675 | 1591 | 3 | 52 | 47 |
| LR | holdout | urgente | 1 | 698 | 5.7846 | 11.8637 | 2.4858 | -1.5074 | 0.5072 | 9.4306 | 0.2493 | 0.5573 | 0.7794 | 11.0387 | 9.5312 | 14 | 180 | 110 | 0.8273 | 0.5056 | 0.6276 | 0.8453 | 499 | 19 | 89 | 91 |

## Metricas por Tramo

| split | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 0-2 | 1183 | 0.8361 | 1.6868 | 0.4739 | 0.7383 | 0.2156 | 0.8850 | 0.7633 | 0.9577 | 0.9932 | 1.2299 | 1.9682 | 14 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9983 | 1181 | 2 | 0 | 0 |
| holdout | 3-6 | 680 | 1.4116 | 2.2467 | 0.9937 | -0.2619 | 0.6574 | 2.2483 | 0.5029 | 0.9059 | 0.9868 | 4.0324 | 3.7704 | 14 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9926 | 675 | 5 | 0 | 0 |
| holdout | 7-13 | 249 | 3.7904 | 4.5775 | 3.5733 | -2.3083 | 0.7791 | 6.8397 | 0.1365 | 0.4177 | 0.9116 | 9.0281 | 6.7198 | 14 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.9398 | 234 | 15 | 0 | 0 |
| holdout | 14+ (PLOS) | 279 | 16.7691 | 25.0790 | 11.4384 | -8.1581 | 0.8100 | 29.2327 | 0.0251 | 0.1004 | 0.2724 | 30.9821 | 22.8240 | 14 | 279 | 138 | 1.0000 | 0.4946 | 0.6619 | 0.4946 | 0 | 0 | 141 | 138 |

## Metricas por Segmento y Tramo

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
