# Reporte de Evaluacion RF - holdout

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Metricas Globales

| Metrica | Valor |
|---|---:|
| n_casos | 2391 |
| mae | 3.0966 |
| rmse | 8.2899 |
| medae | 0.9118 |
| me | -1.3345 |
| pup | 45.50% |
| mae_asimetrico_alpha_2 | 5.3121 |
| precision_plos_14 | 81.97% |
| recall_plos_14 | 53.76% |
| f1_plos_14 | 64.94% |
| accuracy_plos_14 | 93.22% |
| tp_plos_14 | 150.0000 |
| fp_plos_14 | 33.0000 |
| fn_plos_14 | 129.0000 |
| tn_plos_14 | 2079.0000 |
| pct_error_abs_le_1d | 53.24% |
| pct_error_abs_le_3d | 77.25% |
| pct_error_abs_le_7d | 89.42% |

## Metricas por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | holdout | programado | 0 | 1693 | 2.0470 | 6.1960 | 0.7103 | -0.8401 | 0.4347 | 3.4905 | 0.6556 | 0.8671 | 0.9409 | 4.3615 | 3.5214 | 14 | 99 | 73 | 0.8082 | 0.5960 | 0.6860 | 0.9681 | 1580 | 14 | 40 | 59 |
| RF | holdout | urgente | 1 | 698 | 5.6423 | 11.9287 | 2.5490 | -2.5336 | 0.5043 | 9.7303 | 0.2335 | 0.5430 | 0.7808 | 11.0387 | 8.5051 | 14 | 180 | 110 | 0.8273 | 0.5056 | 0.6276 | 0.8453 | 499 | 19 | 89 | 91 |

## Metricas por Tramo

| split | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 0-2 | 1183 | 0.8514 | 1.6334 | 0.4739 | 0.7701 | 0.1970 | 0.8921 | 0.7777 | 0.9459 | 0.9907 | 1.2299 | 2.0000 | 14 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9992 | 1182 | 1 | 0 | 0 |
| holdout | 3-6 | 680 | 1.6466 | 2.7319 | 1.0873 | -0.1591 | 0.6529 | 2.5495 | 0.4691 | 0.8706 | 0.9750 | 4.0324 | 3.8733 | 14 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.9868 | 671 | 9 | 0 | 0 |
| holdout | 7-13 | 249 | 4.2662 | 5.3186 | 3.9321 | -1.6443 | 0.7149 | 7.2214 | 0.1004 | 0.4096 | 0.8554 | 9.0281 | 7.3838 | 14 | 0 | 23 | 0.0000 | 0.0000 | 0.0000 | 0.9076 | 226 | 23 | 0 | 0 |
| holdout | 14+ (PLOS) | 279 | 15.1064 | 23.1127 | 10.4378 | -12.8466 | 0.8351 | 29.0830 | 0.0323 | 0.1219 | 0.3226 | 30.9821 | 18.1355 | 14 | 279 | 150 | 1.0000 | 0.5376 | 0.6993 | 0.5376 | 0 | 0 | 129 | 150 |

## Metricas por Segmento y Tramo

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
