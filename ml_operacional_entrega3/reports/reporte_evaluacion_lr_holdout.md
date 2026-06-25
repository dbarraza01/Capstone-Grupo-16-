# Reporte de Evaluacion LR - holdout

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Metricas Globales

| Metrica | Valor |
|---|---:|
| n_casos | 2391 |
| mae | 13.0718 |
| rmse | 181.1970 |
| medae | 0.8983 |
| me | 8.9717 |
| pup | 50.10% |
| mae_asimetrico_alpha_2 | 15.1219 |
| precision_plos_14 | 69.71% |
| recall_plos_14 | 51.97% |
| f1_plos_14 | 59.55% |
| accuracy_plos_14 | 91.76% |
| tp_plos_14 | 145.0000 |
| fp_plos_14 | 63.0000 |
| fn_plos_14 | 134.0000 |
| tn_plos_14 | 2049.0000 |
| pct_error_abs_le_1d | 53.12% |
| pct_error_abs_le_3d | 74.61% |
| pct_error_abs_le_7d | 85.99% |

## Metricas por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | holdout | programado | 0 | 1693 | 3.3365 | 15.2619 | 0.5772 | 0.7439 | 0.4897 | 4.6328 | 0.6551 | 0.8512 | 0.9303 | 4.3615 | 5.1054 | 14 | 99 | 71 | 0.7606 | 0.5455 | 0.6353 | 0.9634 | 1577 | 17 | 45 | 54 |
| LR | holdout | urgente | 1 | 698 | 36.6848 | 334.5181 | 3.0954 | 28.9281 | 0.5287 | 40.5631 | 0.2307 | 0.4914 | 0.6891 | 11.0387 | 39.9668 | 14 | 180 | 137 | 0.6642 | 0.5056 | 0.5741 | 0.8066 | 472 | 46 | 89 | 91 |

## Metricas por Tramo

| split | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 0-2 | 1183 | 1.0259 | 3.6510 | 0.4108 | 0.7708 | 0.3576 | 1.1534 | 0.7861 | 0.9434 | 0.9839 | 1.2299 | 2.0008 | 14 | 0 | 11 | 0.0000 | 0.0000 | 0.0000 | 0.9907 | 1172 | 11 | 0 | 0 |
| holdout | 3-6 | 680 | 2.4020 | 6.5532 | 1.1641 | 0.6365 | 0.6132 | 3.2848 | 0.4485 | 0.8162 | 0.9544 | 4.0324 | 4.6688 | 14 | 0 | 23 | 0.0000 | 0.0000 | 0.0000 | 0.9662 | 657 | 23 | 0 | 0 |
| holdout | 7-13 | 249 | 5.6301 | 10.0437 | 4.3965 | -0.8014 | 0.7108 | 8.8459 | 0.1165 | 0.3574 | 0.7751 | 9.0281 | 8.2267 | 14 | 0 | 29 | 0.0000 | 0.0000 | 0.0000 | 0.8835 | 220 | 29 | 0 | 0 |
| holdout | 14+ (PLOS) | 279 | 96.7950 | 530.2062 | 16.1565 | 72.7821 | 0.6487 | 108.8015 | 0.0215 | 0.0860 | 0.1792 | 30.9821 | 103.7641 | 14 | 279 | 145 | 1.0000 | 0.5197 | 0.6840 | 0.5197 | 0 | 0 | 134 | 145 |

## Metricas por Segmento y Tramo

| split | segmento | es_urgencia | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | programado | 0 | 0-2 | 983 | 0.6773 | 1.4548 | 0.3272 | 0.4491 | 0.3744 | 0.7915 | 0.8372 | 0.9654 | 0.9949 | 1.2533 | 1.7024 | 14 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9980 | 981 | 2 | 0 | 0 |
| holdout | programado | 0 | 3-6 | 501 | 1.8445 | 5.1595 | 0.9090 | 0.3346 | 0.6267 | 2.5994 | 0.5369 | 0.8842 | 0.9780 | 3.9381 | 4.2727 | 14 | 0 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.9840 | 493 | 8 | 0 | 0 |
| holdout | programado | 0 | 7-13 | 110 | 5.0669 | 9.1513 | 4.2625 | -1.7527 | 0.7545 | 8.4767 | 0.1273 | 0.3545 | 0.8273 | 8.9818 | 7.2291 | 14 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.9364 | 103 | 7 | 0 | 0 |
| holdout | programado | 0 | 14+ (PLOS) | 99 | 35.3684 | 61.1103 | 16.6644 | 8.5174 | 0.6465 | 48.7939 | 0.0303 | 0.1010 | 0.1616 | 32.2323 | 40.7498 | 14 | 99 | 54 | 1.0000 | 0.5455 | 0.7059 | 0.5455 | 0 | 0 | 45 | 54 |
| holdout | urgente | 1 | 0-2 | 200 | 2.7391 | 8.2731 | 0.9854 | 2.3522 | 0.2750 | 2.9325 | 0.5350 | 0.8350 | 0.9300 | 1.1150 | 3.4672 | 14 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.9550 | 191 | 9 | 0 | 0 |
| holdout | urgente | 1 | 3-6 | 179 | 3.9624 | 9.4144 | 2.4115 | 1.4815 | 0.5754 | 5.2029 | 0.2011 | 0.6257 | 0.8883 | 4.2961 | 5.7776 | 14 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.9162 | 164 | 15 | 0 | 0 |
| holdout | urgente | 1 | 7-13 | 139 | 6.0759 | 10.6973 | 4.5998 | -0.0485 | 0.6763 | 9.1381 | 0.1079 | 0.3597 | 0.7338 | 9.0647 | 9.0162 | 14 | 0 | 22 | 0.0000 | 0.0000 | 0.0000 | 0.8417 | 117 | 22 | 0 | 0 |
| holdout | urgente | 1 | 14+ (PLOS) | 180 | 130.5796 | 658.5437 | 15.7326 | 108.1276 | 0.6500 | 141.8057 | 0.0167 | 0.0778 | 0.1889 | 30.2944 | 138.4221 | 14 | 180 | 91 | 1.0000 | 0.5056 | 0.6716 | 0.5056 | 0 | 0 | 89 | 91 |
