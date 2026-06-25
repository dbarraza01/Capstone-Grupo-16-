# Comparacion Final de Modelos Operacionales

PLOS se define como `LOS >= 14` dias.

## Holdout Global

| modelo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | 2391 | 2.8600 | 7.1668 | 0.8359 | -0.9624 | 0.4471 | 4.7712 | 0.8000 | 0.5878 | 0.6777 |
| RF | 2391 | 3.0966 | 8.2899 | 0.9118 | -1.3345 | 0.4550 | 5.3121 | 0.8197 | 0.5376 | 0.6494 |
| LR | 2391 | 13.0718 | 181.1970 | 0.8983 | 8.9717 | 0.5010 | 15.1219 | 0.6971 | 0.5197 | 0.5955 |

## Holdout por Segmento

| modelo | segmento | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | programado | 1693 | 1.8363 | 5.5007 | 0.5848 | -0.6078 | 0.4182 | 3.0584 | 0.8052 | 0.6263 | 0.7045 |
| XGB | urgente | 698 | 5.3429 | 10.1270 | 2.4921 | -1.8225 | 0.5172 | 8.9255 | 0.7969 | 0.5667 | 0.6623 |
| RF | programado | 1693 | 2.0470 | 6.1960 | 0.7103 | -0.8401 | 0.4347 | 3.4905 | 0.8082 | 0.5960 | 0.6860 |
| RF | urgente | 698 | 5.6423 | 11.9287 | 2.5490 | -2.5336 | 0.5043 | 9.7303 | 0.8273 | 0.5056 | 0.6276 |
| LR | programado | 1693 | 3.3365 | 15.2619 | 0.5772 | 0.7439 | 0.4897 | 4.6328 | 0.7606 | 0.5455 | 0.6353 |
| LR | urgente | 698 | 36.6848 | 334.5181 | 3.0954 | 28.9281 | 0.5287 | 40.5631 | 0.6642 | 0.5056 | 0.5741 |

## Holdout por Tramo LOS

Los tramos `0-2`, `3-6` y `7-13` corresponden a LOS < 14 dias. Son clinicamente relevantes porque concentran la mayor parte de los casos.

| modelo | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | n_plos_pred | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | 0-2 | 1183 | 0.7935 | 1.6208 | 0.4754 | 0.7041 | 0.1716 | 0.8382 | 0.7946 | 0.9637 | 0.9924 | 1.2299 | 1.9341 | 4 | 0.0000 | 0.0000 |
| XGB | 3-6 | 680 | 1.5665 | 2.6294 | 0.9859 | -0.2235 | 0.6868 | 2.4616 | 0.5044 | 0.8912 | 0.9721 | 4.0324 | 3.8089 | 12 | 0.0000 | 0.0000 |
| XGB | 7-13 | 249 | 4.0996 | 4.9236 | 3.6032 | -1.7626 | 0.7189 | 7.0307 | 0.1004 | 0.4016 | 0.8755 | 9.0281 | 7.2655 | 25 | 0.0000 | 0.0000 |
| XGB | 14+ (PLOS) | 279 | 13.6682 | 19.7624 | 10.4781 | -9.1154 | 0.7885 | 25.0600 | 0.0573 | 0.1649 | 0.3226 | 30.9821 | 21.8667 | 164 | 0.5878 | 0.7404 |
| RF | 0-2 | 1183 | 0.8514 | 1.6334 | 0.4739 | 0.7701 | 0.1970 | 0.8921 | 0.7777 | 0.9459 | 0.9907 | 1.2299 | 2.0000 | 1 | 0.0000 | 0.0000 |
| RF | 3-6 | 680 | 1.6466 | 2.7319 | 1.0873 | -0.1591 | 0.6529 | 2.5495 | 0.4691 | 0.8706 | 0.9750 | 4.0324 | 3.8733 | 9 | 0.0000 | 0.0000 |
| RF | 7-13 | 249 | 4.2662 | 5.3186 | 3.9321 | -1.6443 | 0.7149 | 7.2214 | 0.1004 | 0.4096 | 0.8554 | 9.0281 | 7.3838 | 23 | 0.0000 | 0.0000 |
| RF | 14+ (PLOS) | 279 | 15.1064 | 23.1127 | 10.4378 | -12.8466 | 0.8351 | 29.0830 | 0.0323 | 0.1219 | 0.3226 | 30.9821 | 18.1355 | 150 | 0.5376 | 0.6993 |
| LR | 0-2 | 1183 | 1.0259 | 3.6510 | 0.4108 | 0.7708 | 0.3576 | 1.1534 | 0.7861 | 0.9434 | 0.9839 | 1.2299 | 2.0008 | 11 | 0.0000 | 0.0000 |
| LR | 3-6 | 680 | 2.4020 | 6.5532 | 1.1641 | 0.6365 | 0.6132 | 3.2848 | 0.4485 | 0.8162 | 0.9544 | 4.0324 | 4.6688 | 23 | 0.0000 | 0.0000 |
| LR | 7-13 | 249 | 5.6301 | 10.0437 | 4.3965 | -0.8014 | 0.7108 | 8.8459 | 0.1165 | 0.3574 | 0.7751 | 9.0281 | 8.2267 | 29 | 0.0000 | 0.0000 |
| LR | 14+ (PLOS) | 279 | 96.7950 | 530.2062 | 16.1565 | 72.7821 | 0.6487 | 108.8015 | 0.0215 | 0.0860 | 0.1792 | 30.9821 | 103.7641 | 145 | 0.5197 | 0.6840 |

## Sintesis LOS < 14

Resumen ponderado por cantidad de casos en los tramos no PLOS (`0-2`, `3-6`, `7-13`).

| modelo | n_casos_los_lt_14 | mae_los_lt_14 | rmse_los_lt_14 | me_los_lt_14 | pup_los_lt_14 | mae_asim_los_lt_14 | pct_le_1d_los_lt_14 | pct_le_3d_los_lt_14 | pct_le_7d_los_lt_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | 2112 | 1.4322 | 2.5604 | 0.1147 | 0.4020 | 2.0909 | 0.6193 | 0.8741 | 0.9721 |
| RF | 2112 | 1.5100 | 2.6893 | 0.1863 | 0.4048 | 2.1719 | 0.5985 | 0.8584 | 0.9697 |
| LR | 2112 | 2.0118 | 5.7607 | 0.5422 | 0.4815 | 2.7466 | 0.5985 | 0.8333 | 0.9498 |

## Interpretacion Clinica Final

- El mejor MAE global en holdout es XGB (2.8600).
- Para LOS < 14 dias, que concentra la mayoria de los casos, el menor MAE ponderado es XGB (1.4322).
- En el tramo 0-2 dias gana XGB (MAE 0.7935); en 3-6 gana XGB (MAE 1.5665); en 7-13 gana XGB (MAE 4.0996).
- En PLOS, el mejor recall lo obtiene XGB (0.5878), lo que reduce el riesgo de no anticipar estancias prolongadas.
- Lectura clinica: LR corresponde al baseline de regresion lineal basica. Si LR gana en algun tramo corto, eso indica que una regla lineal simple ya captura parte importante del patron local; si XGB gana globalmente o en PLOS, mantiene ventaja operacional por balancear error general y deteccion de estancias prolongadas.

## Estabilidad del Modelo (K-Fold sobre Train)

Este diagnostico reentrena la receta final de cada modelo en 5 folds del train operacional y evalua el fold restante. No vuelve a tunear hiperparametros y no usa el holdout final.

| modelo | n_folds | n_casos_promedio_fold | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | 5 | 1912 | 321.6709 +/- 560.4538 | 9127.4324 +/- 15458.0311 | 1.0061 +/- 0.0282 | 317.2411 +/- 560.6485 | 0.4896 +/- 0.0170 | 323.8858 +/- 560.3565 | 0.5658 +/- 0.0319 | 0.5461 +/- 0.0223 | 0.5554 +/- 0.0227 |
| RF | 5 | 1912 | 3.3173 +/- 0.1703 | 9.7119 +/- 0.5887 | 0.9197 +/- 0.0197 | -1.5666 +/- 0.1135 | 0.4823 +/- 0.0149 | 5.7593 +/- 0.3100 | 0.7919 +/- 0.0296 | 0.5290 +/- 0.0295 | 0.6338 +/- 0.0246 |
| XGB | 5 | 1912 | 3.1007 +/- 0.1267 | 8.3368 +/- 0.6178 | 0.8867 +/- 0.0260 | -1.1778 +/- 0.1274 | 0.4584 +/- 0.0201 | 5.2399 +/- 0.2448 | 0.7730 +/- 0.0272 | 0.6132 +/- 0.0278 | 0.6835 +/- 0.0207 |

## Diagnostico de Sobreajuste Train vs Holdout

El gap compara el rendimiento del modelo final ya entrenado contra el holdout. Gaps grandes en MAE/RMSE o caidas fuertes de recall/F1 PLOS indican mayor riesgo de sobreajuste.

| modelo | metrica | train | holdout | gap_holdout_minus_train | ratio_holdout_train |
| --- | --- | --- | --- | --- | --- |
| XGB | mae | 2.3023 | 2.8600 | 0.5576 | 1.2422 |
| XGB | rmse | 5.8586 | 7.1668 | 1.3082 | 1.2233 |
| XGB | mae_asimetrico_alpha_2 | 3.8562 | 4.7712 | 0.9150 | 1.2373 |
| XGB | recall_plos_14 | 0.7081 | 0.5878 | -0.1203 | 0.8301 |
| XGB | f1_plos_14 | 0.7966 | 0.6777 | -0.1189 | 0.8507 |
| RF | mae | 2.7150 | 3.0966 | 0.3816 | 1.1405 |
| RF | rmse | 8.1849 | 8.2899 | 0.1051 | 1.0128 |
| RF | mae_asimetrico_alpha_2 | 4.7186 | 5.3121 | 0.5935 | 1.1258 |
| RF | recall_plos_14 | 0.6705 | 0.5376 | -0.1329 | 0.8018 |
| RF | f1_plos_14 | 0.7686 | 0.6494 | -0.1192 | 0.8448 |
| LR | mae | 1.9374 | 13.0718 | 11.1344 | 6.7472 |
| LR | rmse | 5.2785 | 181.1970 | 175.9185 | 34.3275 |
| LR | mae_asimetrico_alpha_2 | 3.0709 | 15.1219 | 12.0510 | 4.9243 |
| LR | recall_plos_14 | 0.7520 | 0.5197 | -0.2323 | 0.6911 |
| LR | f1_plos_14 | 0.8235 | 0.5955 | -0.2280 | 0.7231 |

## Gap Train vs Holdout Detalle

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
