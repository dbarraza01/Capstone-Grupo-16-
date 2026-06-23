# Reporte Train vs Holdout RF

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Lectura

Un gap alto entre holdout y train indica riesgo de sobreajuste. Las filas por segmento permiten verificar si urgentes y programados se comportan distinto.

## Metricas Train por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | train | programado | 0 | 6772 | 1.7676 | 6.1638 | 0.6689 | -0.7493 | 0.4504 | 3.0260 | 0.6664 | 0.8881 | 0.9587 | 4.4219 | 3.6726 | 14 | 396 | 326 | 0.9018 | 0.7424 | 0.8144 | 0.9802 | 6344 | 32 | 102 | 294 |
| RF | train | urgente | 1 | 2788 | 5.0163 | 11.7231 | 2.1285 | -2.6109 | 0.5072 | 8.8298 | 0.2866 | 0.5961 | 0.8246 | 11.5897 | 8.9788 | 14 | 721 | 506 | 0.8992 | 0.6311 | 0.7416 | 0.8863 | 2016 | 51 | 266 | 455 |

## Metricas Holdout por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | holdout | programado | 0 | 1693 | 2.0470 | 6.1960 | 0.7103 | -0.8401 | 0.4347 | 3.4905 | 0.6556 | 0.8671 | 0.9409 | 4.3615 | 3.5214 | 14 | 99 | 73 | 0.8082 | 0.5960 | 0.6860 | 0.9681 | 1580 | 14 | 40 | 59 |
| RF | holdout | urgente | 1 | 698 | 5.6423 | 11.9287 | 2.5490 | -2.5336 | 0.5043 | 9.7303 | 0.2335 | 0.5430 | 0.7808 | 11.0387 | 8.5051 | 14 | 180 | 110 | 0.8273 | 0.5056 | 0.6276 | 0.8453 | 499 | 19 | 89 | 91 |

## Gap Train vs Holdout

| modelo | alcance | segmento | metrica | train | holdout | gap_holdout_minus_train | ratio_holdout_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF | global | todos | mae | 2.7150 | 3.0966 | 0.3816 | 1.1405 |
| RF | global | todos | rmse | 8.1849 | 8.2899 | 0.1051 | 1.0128 |
| RF | global | todos | medae | 0.8740 | 0.9118 | 0.0378 | 1.0432 |
| RF | global | todos | me | -1.2922 | -1.3345 | -0.0423 | 1.0328 |
| RF | global | todos | pup | 0.4669 | 0.4550 | -0.0119 | 0.9745 |
| RF | global | todos | mae_asimetrico_alpha_2 | 4.7186 | 5.3121 | 0.5935 | 1.1258 |
| RF | global | todos | precision_plos_14 | 0.9002 | 0.8197 | -0.0806 | 0.9105 |
| RF | global | todos | recall_plos_14 | 0.6705 | 0.5376 | -0.1329 | 0.8018 |
| RF | global | todos | f1_plos_14 | 0.7686 | 0.6494 | -0.1192 | 0.8448 |
| RF | segmento | programado | mae | 1.7676 | 2.0470 | 0.2794 | 1.1581 |
| RF | segmento | programado | rmse | 6.1638 | 6.1960 | 0.0322 | 1.0052 |
| RF | segmento | programado | medae | 0.6689 | 0.7103 | 0.0414 | 1.0619 |
| RF | segmento | programado | me | -0.7493 | -0.8401 | -0.0909 | 1.1213 |
| RF | segmento | programado | pup | 0.4504 | 0.4347 | -0.0157 | 0.9652 |
| RF | segmento | programado | mae_asimetrico_alpha_2 | 3.0260 | 3.4905 | 0.4645 | 1.1535 |
| RF | segmento | programado | precision_plos_14 | 0.9018 | 0.8082 | -0.0936 | 0.8962 |
| RF | segmento | programado | recall_plos_14 | 0.7424 | 0.5960 | -0.1465 | 0.8027 |
| RF | segmento | programado | f1_plos_14 | 0.8144 | 0.6860 | -0.1284 | 0.8424 |
| RF | segmento | urgente | mae | 5.0163 | 5.6423 | 0.6261 | 1.1248 |
| RF | segmento | urgente | rmse | 11.7231 | 11.9287 | 0.2056 | 1.0175 |
| RF | segmento | urgente | medae | 2.1285 | 2.5490 | 0.4205 | 1.1976 |
| RF | segmento | urgente | me | -2.6109 | -2.5336 | 0.0772 | 0.9704 |
| RF | segmento | urgente | pup | 0.5072 | 0.5043 | -0.0029 | 0.9943 |
| RF | segmento | urgente | mae_asimetrico_alpha_2 | 8.8298 | 9.7303 | 0.9005 | 1.1020 |
| RF | segmento | urgente | precision_plos_14 | 0.8992 | 0.8273 | -0.0719 | 0.9200 |
| RF | segmento | urgente | recall_plos_14 | 0.6311 | 0.5056 | -0.1255 | 0.8011 |
| RF | segmento | urgente | f1_plos_14 | 0.7416 | 0.6276 | -0.1141 | 0.8462 |

## Holdout por Segmento y Tramo

| split | segmento | es_urgencia | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | programado | 0 | 0-2 | 983 | 0.6031 | 1.1415 | 0.4079 | 0.5238 | 0.2187 | 0.6427 | 0.8525 | 0.9817 | 0.9980 | 1.2533 | 1.7771 | 14 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9990 | 982 | 1 | 0 | 0 |
| holdout | programado | 0 | 3-6 | 501 | 1.3781 | 2.0464 | 0.9552 | -0.5004 | 0.6986 | 2.3173 | 0.5190 | 0.9022 | 0.9880 | 3.9381 | 3.4377 | 14 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9920 | 497 | 4 | 0 | 0 |
| holdout | programado | 0 | 7-13 | 110 | 4.4867 | 5.4382 | 4.1619 | -2.6080 | 0.8182 | 8.0341 | 0.0909 | 0.3818 | 0.8364 | 8.9818 | 6.3738 | 14 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.9182 | 101 | 9 | 0 | 0 |
| holdout | programado | 0 | 14+ (PLOS) | 99 | 17.0583 | 24.2800 | 11.5957 | -14.1373 | 0.8182 | 32.6562 | 0.0202 | 0.0909 | 0.2525 | 32.2323 | 18.0950 | 14 | 99 | 59 | 1.0000 | 0.5960 | 0.7468 | 0.5960 | 0 | 0 | 40 | 59 |
| holdout | urgente | 1 | 0-2 | 200 | 2.0720 | 3.0622 | 1.3677 | 1.9809 | 0.0900 | 2.1175 | 0.4100 | 0.7700 | 0.9550 | 1.1150 | 3.0959 | 14 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 200 | 0 | 0 | 0 |
| holdout | urgente | 1 | 3-6 | 179 | 2.3983 | 4.0781 | 1.4106 | 0.7962 | 0.5251 | 3.1994 | 0.3296 | 0.7821 | 0.9385 | 4.2961 | 5.0923 | 14 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9721 | 174 | 5 | 0 | 0 |
| holdout | urgente | 1 | 7-13 | 139 | 4.0916 | 5.2220 | 3.6342 | -0.8817 | 0.6331 | 6.5783 | 0.1079 | 0.4317 | 0.8705 | 9.0647 | 8.1831 | 14 | 0 | 14 | 0.0000 | 0.0000 | 0.0000 | 0.8993 | 125 | 14 | 0 | 0 |
| holdout | urgente | 1 | 14+ (PLOS) | 180 | 14.0329 | 22.4448 | 9.9080 | -12.1367 | 0.8444 | 27.1177 | 0.0389 | 0.1389 | 0.3611 | 30.2944 | 18.1577 | 14 | 180 | 91 | 1.0000 | 0.5056 | 0.6716 | 0.5056 | 0 | 0 | 89 | 91 |
