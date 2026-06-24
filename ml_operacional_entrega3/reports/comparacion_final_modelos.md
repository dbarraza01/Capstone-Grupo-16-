# Comparacion Final de Modelos Operacionales

PLOS se define como `LOS >= 14` dias.

## Holdout Global

| modelo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | 2391 | 2.8600 | 7.1668 | 0.8359 | -0.9624 | 0.4471 | 4.7712 | 0.8000 | 0.5878 | 0.6777 |
| RF | 2391 | 3.0966 | 8.2899 | 0.9118 | -1.3345 | 0.4550 | 5.3121 | 0.8197 | 0.5376 | 0.6494 |
| LR | 2391 | 3.1666 | 8.8553 | 0.8637 | -0.9015 | 0.4693 | 5.2007 | 0.8625 | 0.4946 | 0.6287 |

## Holdout por Segmento

| modelo | segmento | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | programado | 1693 | 1.8363 | 5.5007 | 0.5848 | -0.6078 | 0.4182 | 3.0584 | 0.8052 | 0.6263 | 0.7045 |
| XGB | urgente | 698 | 5.3429 | 10.1270 | 2.4921 | -1.8225 | 0.5172 | 8.9255 | 0.7969 | 0.5667 | 0.6623 |
| RF | programado | 1693 | 2.0470 | 6.1960 | 0.7103 | -0.8401 | 0.4347 | 3.4905 | 0.8082 | 0.5960 | 0.6860 |
| RF | urgente | 698 | 5.6423 | 11.9287 | 2.5490 | -2.5336 | 0.5043 | 9.7303 | 0.8273 | 0.5056 | 0.6276 |
| LR | programado | 1693 | 2.0873 | 7.2608 | 0.6044 | -0.6517 | 0.4536 | 3.4568 | 0.9400 | 0.4747 | 0.6309 |
| LR | urgente | 698 | 5.7846 | 11.8637 | 2.4858 | -1.5074 | 0.5072 | 9.4306 | 0.8273 | 0.5056 | 0.6276 |

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
| LR | 0-2 | 1183 | 0.8361 | 1.6868 | 0.4739 | 0.7383 | 0.2156 | 0.8850 | 0.7633 | 0.9577 | 0.9932 | 1.2299 | 1.9682 | 2 | 0.0000 | 0.0000 |
| LR | 3-6 | 680 | 1.4116 | 2.2467 | 0.9937 | -0.2619 | 0.6574 | 2.2483 | 0.5029 | 0.9059 | 0.9868 | 4.0324 | 3.7704 | 5 | 0.0000 | 0.0000 |
| LR | 7-13 | 249 | 3.7904 | 4.5775 | 3.5733 | -2.3083 | 0.7791 | 6.8397 | 0.1365 | 0.4177 | 0.9116 | 9.0281 | 6.7198 | 15 | 0.0000 | 0.0000 |
| LR | 14+ (PLOS) | 279 | 16.7691 | 25.0790 | 11.4384 | -8.1581 | 0.8100 | 29.2327 | 0.0251 | 0.1004 | 0.2724 | 30.9821 | 22.8240 | 138 | 0.4946 | 0.6619 |

## Sintesis LOS < 14

Resumen ponderado por cantidad de casos en los tramos no PLOS (`0-2`, `3-6`, `7-13`).

| modelo | n_casos_los_lt_14 | mae_los_lt_14 | rmse_los_lt_14 | me_los_lt_14 | pup_los_lt_14 | mae_asim_los_lt_14 | pct_le_1d_los_lt_14 | pct_le_3d_los_lt_14 | pct_le_7d_los_lt_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | 2112 | 1.3697 | 2.3852 | 0.0571 | 0.4242 | 2.0260 | 0.6056 | 0.8774 | 0.9815 |
| XGB | 2112 | 1.4322 | 2.5604 | 0.1147 | 0.4020 | 2.0909 | 0.6193 | 0.8741 | 0.9721 |
| RF | 2112 | 1.5100 | 2.6893 | 0.1863 | 0.4048 | 2.1719 | 0.5985 | 0.8584 | 0.9697 |

## Interpretacion Clinica Final

- El mejor MAE global en holdout es XGB (2.8600).
- Para LOS < 14 dias, que concentra la mayoria de los casos, el menor MAE ponderado es LR (1.3697).
- En el tramo 0-2 dias gana XGB (MAE 0.7935); en 3-6 gana LR (MAE 1.4116); en 7-13 gana LR (MAE 3.7904).
- En PLOS, el mejor recall lo obtiene XGB (0.5878), lo que reduce el riesgo de no anticipar estancias prolongadas.
- Lectura clinica: LR/Ridge es el modelo mas fuerte si se prioriza estrictamente el desempeno en LOS < 14 dias. XGB es el mejor candidato operacional si se necesita un unico modelo balanceado, porque combina el mejor MAE global, buen rendimiento en el tramo mas frecuente 0-2 y mejor deteccion de PLOS.

## Estabilidad del Modelo (K-Fold sobre Train)

Este diagnostico reentrena la receta final de cada modelo en 5 folds del train operacional y evalua el fold restante. No vuelve a tunear hiperparametros y no usa el holdout final.

| modelo | n_folds | n_casos_promedio_fold | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | 5 | 1912 | 3.5997 +/- 0.6743 | 16.8949 +/- 14.7682 | 0.9286 +/- 0.0327 | -0.9863 +/- 0.5266 | 0.4778 +/- 0.0216 | 5.8927 +/- 0.7831 | 0.8150 +/- 0.0269 | 0.4906 +/- 0.0316 | 0.6123 +/- 0.0311 |
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
| LR | mae | 3.0010 | 3.1666 | 0.1656 | 1.0552 |
| LR | rmse | 9.6423 | 8.8553 | -0.7869 | 0.9184 |
| LR | mae_asimetrico_alpha_2 | 5.0944 | 5.2007 | 0.1062 | 1.0209 |
| LR | recall_plos_14 | 0.5452 | 0.4946 | -0.0506 | 0.9072 |
| LR | f1_plos_14 | 0.6689 | 0.6287 | -0.0402 | 0.9400 |

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
| LR | global | todos | mae | 3.0010 | 3.1666 | 0.1656 | 1.0552 |
| LR | global | todos | rmse | 9.6423 | 8.8553 | -0.7869 | 0.9184 |
| LR | global | todos | medae | 0.8681 | 0.8637 | -0.0043 | 0.9950 |
| LR | global | todos | me | -1.1858 | -0.9015 | 0.2842 | 0.7603 |
| LR | global | todos | pup | 0.4769 | 0.4693 | -0.0076 | 0.9840 |
| LR | global | todos | mae_asimetrico_alpha_2 | 5.0944 | 5.2007 | 0.1062 | 1.0209 |
| LR | global | todos | precision_plos_14 | 0.8651 | 0.8625 | -0.0026 | 0.9970 |
| LR | global | todos | recall_plos_14 | 0.5452 | 0.4946 | -0.0506 | 0.9072 |
| LR | global | todos | f1_plos_14 | 0.6689 | 0.6287 | -0.0402 | 0.9400 |
| LR | segmento | programado | mae | 1.9726 | 2.0873 | 0.1147 | 1.0581 |
| LR | segmento | programado | rmse | 8.4226 | 7.2608 | -1.1618 | 0.8621 |
| LR | segmento | programado | medae | 0.6168 | 0.6044 | -0.0123 | 0.9800 |
| LR | segmento | programado | me | -0.6755 | -0.6517 | 0.0238 | 0.9648 |
| LR | segmento | programado | pup | 0.4668 | 0.4536 | -0.0131 | 0.9718 |
| LR | segmento | programado | mae_asimetrico_alpha_2 | 3.2966 | 3.4568 | 0.1602 | 1.0486 |
| LR | segmento | programado | precision_plos_14 | 0.9064 | 0.9400 | 0.0336 | 1.0371 |
| LR | segmento | programado | recall_plos_14 | 0.5379 | 0.4747 | -0.0631 | 0.8826 |
| LR | segmento | programado | f1_plos_14 | 0.6751 | 0.6309 | -0.0442 | 0.9345 |
| LR | segmento | urgente | mae | 5.4992 | 5.7846 | 0.2854 | 1.0519 |
| LR | segmento | urgente | rmse | 12.1035 | 11.8637 | -0.2398 | 0.9802 |
| LR | segmento | urgente | medae | 2.2337 | 2.4858 | 0.2521 | 1.1128 |
| LR | segmento | urgente | me | -2.4251 | -1.5074 | 0.9177 | 0.6216 |
| LR | segmento | urgente | pup | 0.5014 | 0.5072 | 0.0057 | 1.0114 |
| LR | segmento | urgente | mae_asimetrico_alpha_2 | 9.4614 | 9.4306 | -0.0308 | 0.9967 |
| LR | segmento | urgente | precision_plos_14 | 0.8443 | 0.8273 | -0.0171 | 0.9798 |
| LR | segmento | urgente | recall_plos_14 | 0.5492 | 0.5056 | -0.0437 | 0.9205 |
| LR | segmento | urgente | f1_plos_14 | 0.6655 | 0.6276 | -0.0380 | 0.9430 |
