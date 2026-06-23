# Reporte Train vs Holdout XGB

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Lectura

Un gap alto entre holdout y train indica riesgo de sobreajuste. Las filas por segmento permiten verificar si urgentes y programados se comportan distinto.

## Metricas Train por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | train | programado | 0 | 6772 | 1.5669 | 4.5873 | 0.5985 | -0.4617 | 0.4340 | 2.5812 | 0.6735 | 0.8906 | 0.9600 | 4.4219 | 3.9602 | 14 | 396 | 326 | 0.9018 | 0.7424 | 0.8144 | 0.9802 | 6344 | 32 | 102 | 294 |
| XGB | train | urgente | 1 | 2788 | 4.0887 | 8.1597 | 1.8864 | -1.6399 | 0.5126 | 6.9531 | 0.3016 | 0.6424 | 0.8472 | 11.5897 | 9.9497 | 14 | 721 | 543 | 0.9153 | 0.6893 | 0.7864 | 0.9032 | 2021 | 46 | 224 | 497 |

## Metricas Holdout por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | holdout | programado | 0 | 1693 | 1.8363 | 5.5007 | 0.5848 | -0.6078 | 0.4182 | 3.0584 | 0.6692 | 0.8854 | 0.9457 | 4.3615 | 3.7537 | 14 | 99 | 77 | 0.8052 | 0.6263 | 0.7045 | 0.9693 | 1579 | 15 | 37 | 62 |
| XGB | holdout | urgente | 1 | 698 | 5.3429 | 10.1270 | 2.4921 | -1.8225 | 0.5172 | 8.9255 | 0.2736 | 0.5630 | 0.7765 | 11.0387 | 9.2162 | 14 | 180 | 128 | 0.7969 | 0.5667 | 0.6623 | 0.8510 | 492 | 26 | 78 | 102 |

## Gap Train vs Holdout

| modelo | alcance | segmento | metrica | train | holdout | gap_holdout_minus_train | ratio_holdout_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | global | todos | mae | 2.3023 | 2.8600 | 0.5576 | 1.2422 |
| XGB | global | todos | rmse | 5.8586 | 7.1668 | 1.3082 | 1.2233 |
| XGB | global | todos | medae | 0.8234 | 0.8359 | 0.0125 | 1.0152 |
| XGB | global | todos | me | -0.8053 | -0.9624 | -0.1570 | 1.1950 |
| XGB | global | todos | pup | 0.4569 | 0.4471 | -0.0098 | 0.9785 |
| XGB | global | todos | mae_asimetrico_alpha_2 | 3.8562 | 4.7712 | 0.9150 | 1.2373 |
| XGB | global | todos | precision_plos_14 | 0.9102 | 0.8000 | -0.1102 | 0.8789 |
| XGB | global | todos | recall_plos_14 | 0.7081 | 0.5878 | -0.1203 | 0.8301 |
| XGB | global | todos | f1_plos_14 | 0.7966 | 0.6777 | -0.1189 | 0.8507 |
| XGB | segmento | programado | mae | 1.5669 | 1.8363 | 0.2694 | 1.1720 |
| XGB | segmento | programado | rmse | 4.5873 | 5.5007 | 0.9134 | 1.1991 |
| XGB | segmento | programado | medae | 0.5985 | 0.5848 | -0.0136 | 0.9772 |
| XGB | segmento | programado | me | -0.4617 | -0.6078 | -0.1460 | 1.3163 |
| XGB | segmento | programado | pup | 0.4340 | 0.4182 | -0.0158 | 0.9636 |
| XGB | segmento | programado | mae_asimetrico_alpha_2 | 2.5812 | 3.0584 | 0.4772 | 1.1849 |
| XGB | segmento | programado | precision_plos_14 | 0.9018 | 0.8052 | -0.0966 | 0.8928 |
| XGB | segmento | programado | recall_plos_14 | 0.7424 | 0.6263 | -0.1162 | 0.8435 |
| XGB | segmento | programado | f1_plos_14 | 0.8144 | 0.7045 | -0.1099 | 0.8651 |
| XGB | segmento | urgente | mae | 4.0887 | 5.3429 | 1.2541 | 1.3067 |
| XGB | segmento | urgente | rmse | 8.1597 | 10.1270 | 1.9673 | 1.2411 |
| XGB | segmento | urgente | medae | 1.8864 | 2.4921 | 0.6057 | 1.3211 |
| XGB | segmento | urgente | me | -1.6399 | -1.8225 | -0.1825 | 1.1113 |
| XGB | segmento | urgente | pup | 0.5126 | 0.5172 | 0.0046 | 1.0090 |
| XGB | segmento | urgente | mae_asimetrico_alpha_2 | 6.9531 | 8.9255 | 1.9724 | 1.2837 |
| XGB | segmento | urgente | precision_plos_14 | 0.9153 | 0.7969 | -0.1184 | 0.8706 |
| XGB | segmento | urgente | recall_plos_14 | 0.6893 | 0.5667 | -0.1227 | 0.8221 |
| XGB | segmento | urgente | f1_plos_14 | 0.7864 | 0.6623 | -0.1241 | 0.8422 |

## Holdout por Segmento y Tramo

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
