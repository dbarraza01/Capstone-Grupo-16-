# Impacto del Clasificador en el Regresor

PLOS se define como `LOS >= 14` dias. El umbral `prob_riesgo >= 0.50` solo se usa para convertir la probabilidad del clasificador en alerta binaria; el regresor recibe la probabilidad continua `prob_los_14`.

## Desempeno Global del Clasificador

| modelo | n_casos | n_plos_real | n_alertas_clasificador | roc_auc | pr_auc | brier | precision | recall | f1 | accuracy | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | 2391 | 279 | 377 | 0.9554 | 0.7763 | 0.0610 | 0.6074 | 0.8208 | 0.6982 | 0.9172 | 148 | 50 | 229 |
| RF | 2391 | 279 | 419 | 0.9529 | 0.7311 | 0.0676 | 0.5609 | 0.8423 | 0.6734 | 0.9046 | 184 | 44 | 235 |

## Desempeno del Clasificador por Segmento

| modelo | segmento | n_casos | n_plos_real | n_alertas_clasificador | roc_auc | pr_auc | brier | precision | recall | f1 | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | programado | 1693 | 99 | 149 | 0.9777 | 0.8202 | 0.0355 | 0.5638 | 0.8485 | 0.6774 | 65 | 15 | 84 |
| XGB | urgente | 698 | 180 | 228 | 0.8917 | 0.7754 | 0.1228 | 0.6360 | 0.8056 | 0.7108 | 83 | 35 | 145 |
| RF | programado | 1693 | 99 | 166 | 0.9784 | 0.7813 | 0.0371 | 0.5361 | 0.8990 | 0.6717 | 77 | 10 | 89 |
| RF | urgente | 698 | 180 | 253 | 0.8809 | 0.7473 | 0.1415 | 0.5771 | 0.8111 | 0.6744 | 107 | 34 | 146 |

## Clasificador vs Salida Final del Regresor

| modelo | salida | n_alertas | precision | recall | f1 | accuracy | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB | clasificador_prob_ge_0_50 | 377 | 0.6074 | 0.8208 | 0.6982 | 0.9172 | 148 | 50 | 229 |
| XGB | regresor_los_ge_14 | 205 | 0.8000 | 0.5878 | 0.6777 | 0.9348 | 41 | 115 | 164 |
| RF | clasificador_prob_ge_0_50 | 419 | 0.5609 | 0.8423 | 0.6734 | 0.9046 | 184 | 44 | 235 |
| RF | regresor_los_ge_14 | 183 | 0.8197 | 0.5376 | 0.6494 | 0.9322 | 33 | 129 | 150 |

## Importancia de `prob_los_14` en el Regresor XGB

| segmento | importance_type | rank | importance | importance_pct | n_features_modelo |
| --- | --- | --- | --- | --- | --- |
| programado | cover | 196 | 3097.6843 | 0.0026 | 1652 |
| programado | feature_importances_attr | 1 | 0.0822 | 0.0822 | 1652 |
| programado | gain | 1 | 62.0531 | 0.0822 | 1652 |
| programado | total_cover | 1 | 696979.0000 | 0.0626 | 1652 |
| programado | total_gain | 1 | 13961.9375 | 0.5419 | 1652 |
| programado | weight | 1 | 225.0000 | 0.0813 | 1652 |
| urgente | cover | 305 | 1073.9272 | 0.0015 | 1652 |
| urgente | feature_importances_attr | 2 | 0.0371 | 0.0371 | 1652 |
| urgente | gain | 2 | 38.1330 | 0.0371 | 1652 |
| urgente | total_cover | 1 | 118132.0000 | 0.0331 | 1652 |
| urgente | total_gain | 1 | 4194.6270 | 0.3021 | 1652 |
| urgente | weight | 1 | 110.0000 | 0.0518 | 1652 |

### Como leer `importance_type`

`importance_type` indica la forma en que XGBoost calcula la importancia de una variable dentro de los arboles del regresor:

- `gain`: mejora promedio que produce una variable cada vez que se usa para dividir un nodo. Si es alto, significa que esa variable ayuda mucho a reducir el error cuando aparece.
- `total_gain`: suma total de toda la mejora aportada por esa variable en todos los arboles. Combina que tan util es y cuantas veces aporta.
- `weight`: cantidad de veces que la variable fue usada para hacer divisiones en los arboles. Si es alto, el modelo recurre muchas veces a esa variable.
- `cover`: cantidad promedio de observaciones afectadas por las divisiones donde aparece la variable.
- `total_cover`: suma total de observaciones afectadas por todas las divisiones donde aparece la variable.
- `feature_importances_attr`: importancia normalizada que entrega directamente el objeto `XGBRegressor`. En este caso coincide con una version normalizada de `gain`.

La columna `rank` muestra el puesto de `prob_los_14` entre todas las variables del regresor. Rank 1 significa que fue la variable mas importante bajo ese criterio. `importance_pct` muestra que proporcion de la importancia total corresponde a `prob_los_14`.

## Interpretacion

El pipeline tiene dos salidas distintas que no deben interpretarse como si fueran lo mismo:

1. `prob_riesgo`: salida del clasificador. Es una probabilidad de que el paciente tenga PLOS, es decir, `LOS >= 14` dias.
2. `los_dias_predichos`: salida del regresor. Es una estimacion de cuantos dias exactos estara hospitalizado el paciente.

Cuando decimos que conviene separar dos usos, nos referimos a esto:

- Si el hospital quiere una alerta temprana de riesgo PLOS, deberia mirar `prob_riesgo`. Por ejemplo, podria definir una regla como `prob_riesgo >= 0.50` o ajustar el umbral a `0.35`, `0.40`, etc., segun si quiere capturar mas pacientes de riesgo o reducir falsas alarmas.
- Si el hospital quiere estimar cuantos dias podria durar la hospitalizacion, deberia mirar `los_dias_predichos`.

El problema aparece cuando se usa `los_dias_predichos >= 14` como si fuera la unica alerta PLOS. El regresor intenta predecir dias exactos y tiende a ser conservador con estancias largas. Por eso puede ocurrir que un paciente tenga alto `prob_riesgo`, pero el regresor prediga 12 o 13 dias. Clinicamente ese paciente sigue siendo riesgoso, aunque la regla `los_dias_predichos >= 14` no lo marque como PLOS.

En los resultados de holdout esto se ve claramente para XGB:

- El clasificador con `prob_riesgo >= 0.50` detecta 229 de 279 pacientes PLOS reales. Eso equivale a un recall de 82.08%.
- La salida final del regresor con `los_dias_predichos >= 14` detecta 164 de 279 pacientes PLOS reales. Eso equivale a un recall de 58.78%.
- Por lo tanto, el clasificador identifica 65 pacientes PLOS reales adicionales que el regresor deja bajo 14 dias.

Esto no significa que el regresor este mal. Significa que cumple una funcion distinta: estimar dias. Para emitir alertas de riesgo PLOS, la salida mas directa y sensible es la probabilidad del clasificador.

La importancia de variables confirma que el clasificador si influye en el regresor:

- En XGB programado, `prob_los_14` queda rank 1 por `gain`, `total_gain`, `weight` y `feature_importances_attr`. Esto quiere decir que, entre 1652 variables, la probabilidad PLOS generada por el clasificador es la senal mas importante para el regresor programado.
- En XGB urgente, `prob_los_14` queda rank 2 por `gain` y `feature_importances_attr`, y rank 1 por `weight` y `total_gain`. Esto quiere decir que tambien es una de las variables centrales del regresor urgente.

Que `prob_los_14` tenga ranking alto pero no concentre 100% de la importancia tambien es esperable. El regresor no solo decide si un paciente sera PLOS; intenta estimar dias exactos para todos los pacientes. Por eso necesita otras variables clinicas para distinguir 1 vs 2 dias, 3 vs 6 dias, 8 vs 12 dias, o 15 vs 30 dias.

La conclusion defendible es que `prob_los_14` funciona como una senal central para orientar al regresor hacia riesgo de estancia prolongada, pero el regresor sigue usando el resto de variables clinicas para ajustar la cantidad exacta de dias.

Conclusion: el clasificador tiene impacto real en los regresores, porque `prob_los_14` aparece entre las variables mas importantes. Sin embargo, para la decision clinica de alerta PLOS conviene usar directamente `prob_riesgo`; para la planificacion de dias esperados conviene usar `los_dias_predichos`.
