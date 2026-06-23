# Reporte Train vs Holdout LR

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Lectura

Un gap alto entre holdout y train indica riesgo de sobreajuste. Las filas por segmento permiten verificar si urgentes y programados se comportan distinto.

## Metricas Train por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | train | programado | 0 | 6772 | 1.3650 | 4.3695 | 0.4814 | -0.3378 | 0.4913 | 2.2164 | 0.7185 | 0.9183 | 0.9678 | 4.4219 | 4.0841 | 14 | 396 | 293 | 0.9317 | 0.6894 | 0.7925 | 0.9789 | 6356 | 20 | 123 | 273 |
| LR | train | urgente | 1 | 2788 | 3.8781 | 9.2485 | 1.6614 | -0.8395 | 0.4982 | 6.2369 | 0.3544 | 0.6718 | 0.8626 | 11.5897 | 10.7502 | 14 | 721 | 568 | 0.9032 | 0.7115 | 0.7960 | 0.9057 | 2012 | 55 | 208 | 513 |

## Metricas Holdout por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | holdout | programado | 0 | 1693 | 2.6008 | 11.0081 | 0.5608 | 0.0396 | 0.4885 | 3.8814 | 0.6639 | 0.8630 | 0.9362 | 4.3615 | 4.4011 | 14 | 99 | 65 | 0.8000 | 0.5253 | 0.6341 | 0.9646 | 1581 | 13 | 47 | 52 |
| LR | holdout | urgente | 1 | 698 | 16.5618 | 127.6121 | 2.8208 | 9.3614 | 0.5172 | 20.1619 | 0.2479 | 0.5186 | 0.7550 | 11.0387 | 20.4001 | 14 | 180 | 122 | 0.7623 | 0.5167 | 0.6159 | 0.8338 | 489 | 29 | 87 | 93 |

## Gap Train vs Holdout

| modelo | alcance | segmento | metrica | train | holdout | gap_holdout_minus_train | ratio_holdout_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR | global | todos | mae | 2.0979 | 6.6764 | 4.5785 | 3.1824 |
| LR | global | todos | rmse | 6.2024 | 69.5687 | 63.3663 | 11.2165 |
| LR | global | todos | medae | 0.6707 | 0.8486 | 0.1779 | 1.2653 |
| LR | global | todos | me | -0.4841 | 2.7609 | 3.2450 | -5.7029 |
| LR | global | todos | pup | 0.4933 | 0.4969 | 0.0036 | 1.0072 |
| LR | global | todos | mae_asimetrico_alpha_2 | 3.3889 | 8.6341 | 5.2452 | 2.5477 |
| LR | global | todos | precision_plos_14 | 0.9129 | 0.7754 | -0.1375 | 0.8494 |
| LR | global | todos | recall_plos_14 | 0.7037 | 0.5197 | -0.1840 | 0.7386 |
| LR | global | todos | f1_plos_14 | 0.7947 | 0.6223 | -0.1724 | 0.7830 |
| LR | segmento | programado | mae | 1.3650 | 2.6008 | 1.2358 | 1.9053 |
| LR | segmento | programado | rmse | 4.3695 | 11.0081 | 6.6385 | 2.5193 |
| LR | segmento | programado | medae | 0.4814 | 0.5608 | 0.0794 | 1.1649 |
| LR | segmento | programado | me | -0.3378 | 0.0396 | 0.3774 | -0.1173 |
| LR | segmento | programado | pup | 0.4913 | 0.4885 | -0.0028 | 0.9943 |
| LR | segmento | programado | mae_asimetrico_alpha_2 | 2.2164 | 3.8814 | 1.6649 | 1.7512 |
| LR | segmento | programado | precision_plos_14 | 0.9317 | 0.8000 | -0.1317 | 0.8586 |
| LR | segmento | programado | recall_plos_14 | 0.6894 | 0.5253 | -0.1641 | 0.7619 |
| LR | segmento | programado | f1_plos_14 | 0.7925 | 0.6341 | -0.1583 | 0.8002 |
| LR | segmento | urgente | mae | 3.8781 | 16.5618 | 12.6837 | 4.2706 |
| LR | segmento | urgente | rmse | 9.2485 | 127.6121 | 118.3636 | 13.7982 |
| LR | segmento | urgente | medae | 1.6614 | 2.8208 | 1.1594 | 1.6978 |
| LR | segmento | urgente | me | -0.8395 | 9.3614 | 10.2009 | -11.1513 |
| LR | segmento | urgente | pup | 0.4982 | 0.5172 | 0.0190 | 1.0381 |
| LR | segmento | urgente | mae_asimetrico_alpha_2 | 6.2369 | 20.1619 | 13.9250 | 3.2327 |
| LR | segmento | urgente | precision_plos_14 | 0.9032 | 0.7623 | -0.1409 | 0.8440 |
| LR | segmento | urgente | recall_plos_14 | 0.7115 | 0.5167 | -0.1948 | 0.7262 |
| LR | segmento | urgente | f1_plos_14 | 0.7960 | 0.6159 | -0.1801 | 0.7738 |

## Holdout por Segmento y Tramo

| split | segmento | es_urgencia | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | programado | 0 | 0-2 | 983 | 0.6009 | 1.1967 | 0.3014 | 0.4072 | 0.3530 | 0.6978 | 0.8413 | 0.9746 | 0.9959 | 1.2533 | 1.6606 | 14 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9990 | 982 | 1 | 0 | 0 |
| holdout | programado | 0 | 3-6 | 501 | 1.4273 | 2.5837 | 0.8588 | -0.0785 | 0.6427 | 2.1803 | 0.5509 | 0.8962 | 0.9800 | 3.9381 | 3.8596 | 14 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.9860 | 494 | 7 | 0 | 0 |
| holdout | programado | 0 | 7-13 | 110 | 4.1647 | 5.5845 | 3.4698 | -2.5502 | 0.8273 | 7.5221 | 0.1727 | 0.4091 | 0.8545 | 8.9818 | 6.4317 | 14 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9545 | 105 | 5 | 0 | 0 |
| holdout | programado | 0 | 14+ (PLOS) | 99 | 26.6587 | 44.6050 | 15.1145 | -0.1351 | 0.6768 | 40.0556 | 0.0202 | 0.0909 | 0.2121 | 32.2323 | 32.0972 | 14 | 99 | 52 | 1.0000 | 0.5253 | 0.6887 | 0.5253 | 0 | 0 | 47 | 52 |
| holdout | urgente | 1 | 0-2 | 200 | 2.1855 | 5.4216 | 0.9574 | 1.9729 | 0.2200 | 2.2917 | 0.5200 | 0.8400 | 0.9550 | 1.1150 | 3.0879 | 14 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9800 | 196 | 4 | 0 | 0 |
| holdout | urgente | 1 | 3-6 | 179 | 2.4842 | 3.6177 | 1.7447 | 0.4990 | 0.5307 | 3.4768 | 0.2905 | 0.7374 | 0.9330 | 4.2961 | 4.7951 | 14 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9721 | 174 | 5 | 0 | 0 |
| holdout | urgente | 1 | 7-13 | 139 | 4.8571 | 6.6903 | 4.1111 | -0.9254 | 0.7122 | 7.7484 | 0.0863 | 0.3381 | 0.8345 | 9.0647 | 8.1394 | 14 | 0 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.8561 | 119 | 20 | 0 | 0 |
| holdout | urgente | 1 | 14+ (PLOS) | 180 | 55.5734 | 251.1349 | 12.2287 | 34.3278 | 0.6833 | 66.1961 | 0.0278 | 0.0833 | 0.2944 | 30.2944 | 64.6223 | 14 | 180 | 93 | 1.0000 | 0.5167 | 0.6813 | 0.5167 | 0 | 0 | 87 | 93 |
