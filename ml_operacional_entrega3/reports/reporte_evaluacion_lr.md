# Reporte Train vs Holdout LR

## Definicion Operacional

PLOS se define como `LOS >= 14` dias. Los tramos de evaluacion son: 0-2, 3-6, 7-13, 14+ (PLOS).

## Lectura

Un gap alto entre holdout y train indica riesgo de sobreajuste. Las filas por segmento permiten verificar si urgentes y programados se comportan distinto.

## Metricas Train por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | train | programado | 0 | 6772 | 1.2556 | 3.7797 | 0.4649 | -0.2555 | 0.4935 | 2.0111 | 0.7312 | 0.9211 | 0.9711 | 4.4219 | 4.1664 | 14 | 396 | 314 | 0.9299 | 0.7374 | 0.8225 | 0.9814 | 6354 | 22 | 104 | 292 |
| LR | train | urgente | 1 | 2788 | 3.5934 | 7.8000 | 1.4719 | -0.5099 | 0.5036 | 5.6450 | 0.3917 | 0.6919 | 0.8763 | 11.5897 | 11.0798 | 14 | 721 | 609 | 0.8998 | 0.7601 | 0.8241 | 0.9161 | 2006 | 61 | 173 | 548 |

## Metricas Holdout por Segmento

| modelo | split | segmento | es_urgencia | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | n_plos_real | n_plos_pred | precision_plos_14 | recall_plos_14 | f1_plos_14 | accuracy_plos_14 | tn_plos_14 | fp_plos_14 | fn_plos_14 | tp_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | holdout | programado | 0 | 1693 | 3.3365 | 15.2619 | 0.5772 | 0.7439 | 0.4897 | 4.6328 | 0.6551 | 0.8512 | 0.9303 | 4.3615 | 5.1054 | 14 | 99 | 71 | 0.7606 | 0.5455 | 0.6353 | 0.9634 | 1577 | 17 | 45 | 54 |
| LR | holdout | urgente | 1 | 698 | 36.6848 | 334.5181 | 3.0954 | 28.9281 | 0.5287 | 40.5631 | 0.2307 | 0.4914 | 0.6891 | 11.0387 | 39.9668 | 14 | 180 | 137 | 0.6642 | 0.5056 | 0.5741 | 0.8066 | 472 | 46 | 89 | 91 |

## Gap Train vs Holdout

| modelo | alcance | segmento | metrica | train | holdout | gap_holdout_minus_train | ratio_holdout_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR | global | todos | mae | 1.9374 | 13.0718 | 11.1344 | 6.7472 |
| LR | global | todos | rmse | 5.2785 | 181.1970 | 175.9185 | 34.3275 |
| LR | global | todos | medae | 0.6324 | 0.8983 | 0.2659 | 1.4204 |
| LR | global | todos | me | -0.3297 | 8.9717 | 9.3014 | -27.2138 |
| LR | global | todos | pup | 0.4964 | 0.5010 | 0.0046 | 1.0093 |
| LR | global | todos | mae_asimetrico_alpha_2 | 3.0709 | 15.1219 | 12.0510 | 4.9243 |
| LR | global | todos | precision_plos_14 | 0.9101 | 0.6971 | -0.2130 | 0.7660 |
| LR | global | todos | recall_plos_14 | 0.7520 | 0.5197 | -0.2323 | 0.6911 |
| LR | global | todos | f1_plos_14 | 0.8235 | 0.5955 | -0.2280 | 0.7231 |
| LR | segmento | programado | mae | 1.2556 | 3.3365 | 2.0809 | 2.6573 |
| LR | segmento | programado | rmse | 3.7797 | 15.2619 | 11.4822 | 4.0379 |
| LR | segmento | programado | medae | 0.4649 | 0.5772 | 0.1123 | 1.2415 |
| LR | segmento | programado | me | -0.2555 | 0.7439 | 0.9994 | -2.9118 |
| LR | segmento | programado | pup | 0.4935 | 0.4897 | -0.0038 | 0.9922 |
| LR | segmento | programado | mae_asimetrico_alpha_2 | 2.0111 | 4.6328 | 2.6217 | 2.3036 |
| LR | segmento | programado | precision_plos_14 | 0.9299 | 0.7606 | -0.1694 | 0.8179 |
| LR | segmento | programado | recall_plos_14 | 0.7374 | 0.5455 | -0.1919 | 0.7397 |
| LR | segmento | programado | f1_plos_14 | 0.8225 | 0.6353 | -0.1872 | 0.7724 |
| LR | segmento | urgente | mae | 3.5934 | 36.6848 | 33.0914 | 10.2090 |
| LR | segmento | urgente | rmse | 7.8000 | 334.5181 | 326.7181 | 42.8870 |
| LR | segmento | urgente | medae | 1.4719 | 3.0954 | 1.6235 | 2.1030 |
| LR | segmento | urgente | me | -0.5099 | 28.9281 | 29.4380 | -56.7361 |
| LR | segmento | urgente | pup | 0.5036 | 0.5287 | 0.0251 | 1.0498 |
| LR | segmento | urgente | mae_asimetrico_alpha_2 | 5.6450 | 40.5631 | 34.9181 | 7.1857 |
| LR | segmento | urgente | precision_plos_14 | 0.8998 | 0.6642 | -0.2356 | 0.7382 |
| LR | segmento | urgente | recall_plos_14 | 0.7601 | 0.5056 | -0.2545 | 0.6652 |
| LR | segmento | urgente | f1_plos_14 | 0.8241 | 0.5741 | -0.2499 | 0.6967 |

## Holdout por Segmento y Tramo

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
