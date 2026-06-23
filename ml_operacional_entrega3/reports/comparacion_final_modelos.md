# Comparacion Final de Modelos Operacionales

PLOS se define como `LOS >= 14` dias.

## Holdout Global

| modelo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | 2391 | 2.8600 | 7.1668 | 0.8359 | -0.9624 | 0.4471 | 4.7712 | 0.8000 | 0.5878 | 0.6777 |
| RF | 2391 | 3.0966 | 8.2899 | 0.9118 | -1.3345 | 0.4550 | 5.3121 | 0.8197 | 0.5376 | 0.6494 |
| LR | 2391 | 6.6764 | 69.5687 | 0.8486 | 2.7609 | 0.4969 | 8.6341 | 0.7754 | 0.5197 | 0.6223 |

## Holdout por Segmento

| modelo | segmento | n_casos | mae | rmse | medae | me | pup | mae_asimetrico_alpha_2 | precision_plos_14 | recall_plos_14 | f1_plos_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | programado | 1693 | 1.8363 | 5.5007 | 0.5848 | -0.6078 | 0.4182 | 3.0584 | 0.8052 | 0.6263 | 0.7045 |
| XGB | urgente | 698 | 5.3429 | 10.1270 | 2.4921 | -1.8225 | 0.5172 | 8.9255 | 0.7969 | 0.5667 | 0.6623 |
| RF | programado | 1693 | 2.0470 | 6.1960 | 0.7103 | -0.8401 | 0.4347 | 3.4905 | 0.8082 | 0.5960 | 0.6860 |
| RF | urgente | 698 | 5.6423 | 11.9287 | 2.5490 | -2.5336 | 0.5043 | 9.7303 | 0.8273 | 0.5056 | 0.6276 |
| LR | programado | 1693 | 2.6008 | 11.0081 | 0.5608 | 0.0396 | 0.4885 | 3.8814 | 0.8000 | 0.5253 | 0.6341 |
| LR | urgente | 698 | 16.5618 | 127.6121 | 2.8208 | 9.3614 | 0.5172 | 20.1619 | 0.7623 | 0.5167 | 0.6159 |

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
