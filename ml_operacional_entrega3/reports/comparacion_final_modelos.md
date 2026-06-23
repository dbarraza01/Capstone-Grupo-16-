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
