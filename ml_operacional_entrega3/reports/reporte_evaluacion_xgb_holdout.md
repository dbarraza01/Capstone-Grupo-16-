# Reporte de Evaluacion XGB - holdout

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Metricas Globales

| Metrica | Valor |
|---|---:|
| n_casos | 2391 |
| mae | 2.8600 |
| rmse | 7.1668 |
| medae | 0.8359 |
| me | -0.9624 |
| pup | 44.71% |
| mae_asimetrico_alpha_2 | 4.7712 |
| precision_plos_14 | 80.00% |
| recall_plos_14 | 58.78% |
| f1_plos_14 | 67.77% |
| accuracy_plos_14 | 93.48% |
| tp_plos_14 | 164.0000 |
| fp_plos_14 | 41.0000 |
| fn_plos_14 | 115.0000 |
| tn_plos_14 | 2071.0000 |
| pct_error_abs_le_1d | 55.37% |
| pct_error_abs_le_3d | 79.13% |
| pct_error_abs_le_7d | 89.63% |

## Metricas por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | holdout | programado | 0 | 1693 | 1.8363 | 5.5007 | 0.5848 | -0.6078 | 0.4182 | 3.0584 | 0.6692 | 0.8854 | 0.9457 | 4.3615 | 3.7537 | 14 | 99 | 77 | 0.8052 | 0.6263 | 0.7045 | 0.9693 | 1579 | 15 | 37 | 62 |
| XGB | holdout | urgente | 1 | 698 | 5.3429 | 10.1270 | 2.4921 | -1.8225 | 0.5172 | 8.9255 | 0.2736 | 0.5630 | 0.7765 | 11.0387 | 9.2162 | 14 | 180 | 128 | 0.7969 | 0.5667 | 0.6623 | 0.8510 | 492 | 26 | 78 | 102 |

## Metricas por Tramo

| split | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 0-2 | 1183 | 0.7935 | 1.6208 | 0.4754 | 0.7041 | 0.1716 | 0.8382 | 0.7946 | 0.9637 | 0.9924 | 1.2299 | 1.9341 | 14 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9966 | 1179 | 4 | 0 | 0 |
| holdout | 3-6 | 680 | 1.5665 | 2.6294 | 0.9859 | -0.2235 | 0.6868 | 2.4616 | 0.5044 | 0.8912 | 0.9721 | 4.0324 | 3.8089 | 14 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9824 | 668 | 12 | 0 | 0 |
| holdout | 7-13 | 249 | 4.0996 | 4.9236 | 3.6032 | -1.7626 | 0.7189 | 7.0307 | 0.1004 | 0.4016 | 0.8755 | 9.0281 | 7.2655 | 14 | 0 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.8996 | 224 | 25 | 0 | 0 |
| holdout | 14+ (PLOS) | 279 | 13.6682 | 19.7624 | 10.4781 | -9.1154 | 0.7885 | 25.0600 | 0.0573 | 0.1649 | 0.3226 | 30.9821 | 21.8667 | 14 | 279 | 164 | 1.0000 | 0.5878 | 0.7404 | 0.5878 | 0 | 0 | 115 | 164 |

## Metricas por Segmento y Tramo

| split | segmento | es_urgencia | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | programado | 0 | 0-2 | 983 | 0.5695 | 0.9987 | 0.4323 | 0.4824 | 0.1851 | 0.6130 | 0.8576 | 0.9888 | 0.9980 | 1.2533 | 1.7357 | 14 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9990 | 982 | 1 | 0 | 0 |
| holdout | programado | 0 | 3-6 | 501 | 1.2708 | 1.9368 | 0.9066 | -0.5288 | 0.7246 | 2.1706 | 0.5469 | 0.9321 | 0.9860 | 3.9381 | 3.4093 | 14 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9920 | 497 | 4 | 0 | 0 |
| holdout | programado | 0 | 7-13 | 110 | 4.3602 | 5.2447 | 3.9220 | -2.3814 | 0.7818 | 7.7311 | 0.1000 | 0.3727 | 0.8455 | 8.9818 | 6.6004 | 14 | 0 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.9091 | 100 | 10 | 0 | 0 |
| holdout | programado | 0 | 14+ (PLOS) | 99 | 14.4727 | 21.4005 | 10.4859 | -9.8608 | 0.7778 | 26.6395 | 0.0505 | 0.1919 | 0.3333 | 32.2323 | 22.3715 | 14 | 99 | 62 | 1.0000 | 0.6263 | 0.7702 | 0.6263 | 0 | 0 | 37 | 62 |
| holdout | urgente | 1 | 0-2 | 200 | 1.8946 | 3.2613 | 1.0597 | 1.7941 | 0.1050 | 1.9449 | 0.4850 | 0.8400 | 0.9650 | 1.1150 | 2.9091 | 14 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.9850 | 197 | 3 | 0 | 0 |
| holdout | urgente | 1 | 3-6 | 179 | 2.3943 | 3.9707 | 1.4590 | 0.6311 | 0.5810 | 3.2759 | 0.3855 | 0.7765 | 0.9330 | 4.2961 | 4.9272 | 14 | 0 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.9553 | 171 | 8 | 0 | 0 |
| holdout | urgente | 1 | 7-13 | 139 | 3.8934 | 4.6539 | 3.4850 | -1.2728 | 0.6691 | 6.4764 | 0.1007 | 0.4245 | 0.8993 | 9.0647 | 7.7919 | 14 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.8921 | 124 | 15 | 0 | 0 |
| holdout | urgente | 1 | 14+ (PLOS) | 180 | 13.2257 | 18.8007 | 10.4777 | -8.7054 | 0.7944 | 24.1913 | 0.0611 | 0.1500 | 0.3167 | 30.2944 | 21.5891 | 14 | 180 | 102 | 1.0000 | 0.5667 | 0.7234 | 0.5667 | 0 | 0 | 78 | 102 |
