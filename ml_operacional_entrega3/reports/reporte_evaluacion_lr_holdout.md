# Reporte de Evaluacion LR - holdout

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Metricas Globales

| Metrica | Valor |
|---|---:|
| n_casos | 2391 |
| mae | 6.6764 |
| rmse | 69.5687 |
| medae | 0.8486 |
| me | 2.7609 |
| pup | 49.69% |
| mae_asimetrico_alpha_2 | 8.6341 |
| precision_plos_14 | 77.54% |
| recall_plos_14 | 51.97% |
| f1_plos_14 | 62.23% |
| accuracy_plos_14 | 92.64% |
| tp_plos_14 | 145.0000 |
| fp_plos_14 | 42.0000 |
| fn_plos_14 | 134.0000 |
| tn_plos_14 | 2070.0000 |
| pct_error_abs_le_1d | 54.25% |
| pct_error_abs_le_3d | 76.24% |
| pct_error_abs_le_7d | 88.33% |

## Metricas por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | holdout | programado | 0 | 1693 | 2.6008 | 11.0081 | 0.5608 | 0.0396 | 0.4885 | 3.8814 | 0.6639 | 0.8630 | 0.9362 | 4.3615 | 4.4011 | 14 | 99 | 65 | 0.8000 | 0.5253 | 0.6341 | 0.9646 | 1581 | 13 | 47 | 52 |
| LR | holdout | urgente | 1 | 698 | 16.5618 | 127.6121 | 2.8208 | 9.3614 | 0.5172 | 20.1619 | 0.2479 | 0.5186 | 0.7550 | 11.0387 | 20.4001 | 14 | 180 | 122 | 0.7623 | 0.5167 | 0.6159 | 0.8338 | 489 | 29 | 87 | 93 |

## Metricas por Tramo

| split | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 0-2 | 1183 | 0.8688 | 2.4818 | 0.3938 | 0.6719 | 0.3305 | 0.9672 | 0.7870 | 0.9518 | 0.9890 | 1.2299 | 1.9019 | 14 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9958 | 1178 | 5 | 0 | 0 |
| holdout | 3-6 | 680 | 1.7055 | 2.8920 | 1.0759 | 0.0735 | 0.6132 | 2.5216 | 0.4824 | 0.8544 | 0.9676 | 4.0324 | 4.1058 | 14 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9824 | 668 | 12 | 0 | 0 |
| holdout | 7-13 | 249 | 4.5512 | 6.2261 | 3.9059 | -1.6431 | 0.7631 | 7.6484 | 0.1245 | 0.3695 | 0.8434 | 9.0281 | 7.3850 | 14 | 0 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.8996 | 224 | 25 | 0 | 0 |
| holdout | 14+ (PLOS) | 279 | 45.3133 | 203.4588 | 13.1960 | 22.0990 | 0.6810 | 56.9204 | 0.0251 | 0.0860 | 0.2652 | 30.9821 | 53.0811 | 14 | 279 | 145 | 1.0000 | 0.5197 | 0.6840 | 0.5197 | 0 | 0 | 134 | 145 |

## Metricas por Segmento y Tramo

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
