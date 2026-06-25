# Informe Metodologico: Regresion Lineal Baseline para Prediccion de LOS

## 1. Objetivo del modelo

La regresion lineal implementada en esta carpeta corresponde al modelo base de la Entrega 3 para estimar la duracion de estancia hospitalaria, medida como `los_dias`. Su funcion principal no es competir en complejidad con XGBoost o Random Forest, sino entregar una referencia estadistica simple, interpretable y comparable bajo los mismos splits operacionales de entrenamiento y holdout.

Este baseline permite responder una pregunta metodologica clave: cuanto valor agregan los modelos no lineales frente a una relacion lineal entre las variables disponibles al ingreso y los dias de hospitalizacion observados.

## 2. Segmentacion del problema

El modelo no se entrena como una unica regresion global. Se entrenan dos regresiones lineales separadas:

- `reg_lr_urgente.joblib`: entrenada solo con pacientes de urgencia.
- `reg_lr_programado.joblib`: entrenada solo con pacientes no urgentes o programados.

Esta decision es coherente con el planteamiento operacional del proyecto: los pacientes urgentes y programados tienen patrones distintos de estancia, prevalencia de PLOS y complejidad clinica. Por lo tanto, se evita forzar una unica recta comun para dos subpoblaciones que se comportan de manera diferente.

Como cada modelo se entrena dentro de su propio segmento, la variable `es_urgencia` se elimina antes del entrenamiento. Dentro del modelo urgente todos los casos tienen `es_urgencia = 1`, y dentro del modelo programado todos tienen `es_urgencia = 0`; por eso esa variable no aporta variacion interna.

## 3. Ecuacion estadistica del modelo

La regresion lineal asume que existe una relacion aditiva entre las variables explicativas y el valor esperado del target transformado. En este caso, no se ajusta directamente `los_dias`, sino una transformacion logaritmica:

```text
z_i = log(1 + LOS_i)
```

Luego, para cada paciente `i`, el modelo estima:

```text
z_i = beta_0 + beta_1 x_{i1} + beta_2 x_{i2} + ... + beta_p x_{ip} + epsilon_i
```

donde:

- `z_i` es `log(1 + LOS_i)`.
- `beta_0` es el intercepto.
- `beta_j` es el coeficiente asociado a la variable `x_j`.
- `x_{ij}` es el valor de la variable `j` para el paciente `i`.
- `epsilon_i` es el error residual no explicado por el modelo.
- `p = 1650` variables predictoras en cada segmento.

Una vez obtenida la prediccion en escala logaritmica, se vuelve a la escala original de dias mediante:

```text
LOS_predicho_i = exp(z_predicho_i) - 1
```

En el codigo, esto se implementa con `TransformedTargetRegressor`, usando:

```text
func = np.log1p
inverse_func = np.expm1
```

Finalmente, las predicciones negativas se recortan a cero:

```text
LOS_predicho_i = max(LOS_predicho_i, 0)
```

Esto es necesario porque el LOS no puede ser negativo en terminos clinicos.

## 4. Por que se transforma el target con log(1 + LOS)

El LOS hospitalario suele tener una distribucion asimetrica: muchos pacientes tienen estancias cortas y pocos pacientes tienen estancias extremadamente largas. Esta cola larga puede distorsionar una regresion lineal entrenada directamente sobre dias.

La transformacion `log(1 + LOS)` busca reducir parcialmente ese problema:

- comprime los valores extremos de estancias largas;
- estabiliza la escala del target;
- hace que el modelo no quede completamente dominado por pocos pacientes con LOS muy alto;
- permite que el ajuste sea mas razonable para la mayoria de los pacientes, que se concentran en estancias cortas o intermedias.

Sin embargo, esta transformacion tambien tiene una consecuencia importante: al volver desde la escala logaritmica a dias, el modelo tiende a suavizar predicciones extremas. Por eso la regresion lineal puede mostrar dificultades en pacientes PLOS o de cola larga.

## 5. Variables utilizadas

Cada modelo lineal utiliza 1650 variables predictoras. Estas variables provienen del dataset operacional ya preprocesado y corresponden principalmente a informacion disponible al ingreso.

Las categorias principales de variables son:

### 5.1 Variables de conteo clinico

Ejemplos:

- `n_procedimientos`
- `n_diag_primarios`
- `n_diag_secundarios`
- `n_diag_total`
- `tiene_diag_primario`

Estas variables resumen la carga diagnostica y procedimental del caso.

### 5.2 Variables temporales

Ejemplos:

- `mes_ingreso`
- `dia_semana_ingreso`

Estas variables capturan diferencias asociadas al momento de ingreso, como patrones semanales o mensuales de hospitalizacion.

### 5.3 Variables one-hot de diagnosticos

Ejemplos:

- `diag_A04`
- `diag_A40`
- `diag_A41`
- `diag_B00`
- `diag_B18`

Estas variables indican la presencia o ausencia de codigos diagnosticos especificos o agrupados generados durante el feature engineering.

Matematicamente, si una variable `diag_X` toma valor 1, significa que el paciente presenta ese diagnostico o agrupacion diagnostica. Si toma valor 0, significa que no aparece en el caso.

### 5.4 Variables one-hot de procedimientos

El modelo tambien incorpora variables derivadas de procedimientos, codificadas como indicadores binarios. Estas variables representan si ciertos procedimientos aparecen o no en el caso.

En conjunto, diagnosticos y procedimientos permiten representar informacion clinica de alta dimensionalidad mediante variables numericas aptas para regresion.

## 6. Variables excluidas

Antes del entrenamiento se eliminan variables que no deben entrar al baseline lineal:

```text
case_id
los_dias
es_urgencia
prob_los_14
int_charlson_diag
int_charlson_proc
int_proc_diag
```

La razon de cada exclusion es la siguiente:

- `case_id`: identificador administrativo, no variable predictiva clinica.
- `los_dias`: target del modelo; incluirlo como predictor seria fuga de informacion.
- `es_urgencia`: se elimina porque el modelo ya fue separado por segmento.
- `prob_los_14`: se elimina porque pertenece al clasificador de la arquitectura de dos etapas; la LR baseline debe ser una regresion directa, no un modelo apilado.
- `int_charlson_diag`, `int_charlson_proc`, `int_proc_diag`: interacciones manuales excluidas para mantener el modelo como baseline lineal simple.

## 7. Se cruzan variables?

No. En la version actual, la regresion lineal no usa cruces ni interacciones entre variables.

Esto esta registrado explicitamente en la metadata del modelo:

```text
interactions = []
regularization = none
stage = baseline_linear_regression
```

Por lo tanto, el modelo asume efectos aditivos. Es decir, el aporte de una variable se suma al aporte de otra variable, pero el modelo no aprende que la combinacion de dos variables tenga un efecto adicional especifico.

Por ejemplo, si existen las variables:

```text
diag_A = 1
proc_B = 1
```

el modelo lineal puede estimar:

```text
beta_diag_A * diag_A + beta_proc_B * proc_B
```

pero no incluye automaticamente un termino como:

```text
beta_interaccion * diag_A * proc_B
```

Esto diferencia a la regresion lineal de modelos como XGBoost, que si pueden capturar interacciones no lineales entre variables mediante particiones sucesivas de arboles.

## 8. Regularizacion

La regresion lineal actual no usa regularizacion Ridge ni Lasso. El estimador base es:

```text
LinearRegression()
```

Esto significa que los coeficientes se ajustan mediante minimos cuadrados ordinarios sobre el target transformado.

La funcion objetivo aproximada en escala logaritmica es:

```text
min_beta sum_i (z_i - z_predicho_i)^2
```

donde:

```text
z_i = log(1 + LOS_i)
```

Al no usar regularizacion, el modelo no penaliza explicitamente coeficientes grandes. Esto lo hace mas simple como baseline, pero tambien mas vulnerable a inestabilidad cuando hay muchas variables one-hot, colinealidad o categorias poco frecuentes.

## 9. Interpretacion de los coeficientes

En una regresion lineal tradicional sin transformacion del target, cada coeficiente `beta_j` se interpreta como el cambio promedio en dias asociado a aumentar una unidad en `x_j`, manteniendo las demas variables constantes.

En este caso, la interpretacion cambia porque el modelo se entrena sobre:

```text
log(1 + LOS)
```

Por eso, un coeficiente positivo indica que la variable se asocia con un aumento multiplicativo aproximado del LOS esperado, mientras que un coeficiente negativo indica una reduccion relativa.

Para variables binarias one-hot, la interpretacion conceptual es:

- si `beta_j > 0`, la presencia de esa condicion o procedimiento se asocia con mayor estancia esperada;
- si `beta_j < 0`, la presencia de esa condicion o procedimiento se asocia con menor estancia esperada;
- si `beta_j` esta cerca de 0, su aporte marginal lineal es bajo.

Esta interpretacion debe hacerse con cautela, porque la presencia de colinealidad entre codigos diagnosticos y procedimentales puede repartir el efecto entre varias variables relacionadas.

## 10. Rol dentro del proyecto

Este modelo cumple el rol de baseline academico y operacional. Su utilidad principal es mostrar cuanto se puede lograr con una aproximacion lineal simple, antes de justificar modelos mas complejos.

Los resultados actuales muestran que XGBoost supera claramente a esta regresion lineal basica en MAE, RMSE y deteccion PLOS14. Esa diferencia es metodologicamente relevante porque indica que el problema de LOS no se comporta solo como una suma lineal de efectos independientes.

En particular, la regresion lineal tiene limitaciones para:

- capturar interacciones entre diagnosticos y procedimientos;
- modelar relaciones no lineales;
- manejar categorias raras de alta dimensionalidad;
- estimar adecuadamente pacientes de estancia prolongada;
- controlar outliers extremos sin regularizacion.

Por lo tanto, la regresion lineal sirve como referencia interpretable, pero no como el modelo operacional ganador.

## 11. Resumen metodologico

| Elemento | Descripcion |
|---|---|
| Modelo | Regresion lineal basica |
| Segmentacion | Un modelo para urgentes y otro para programados |
| Target | `los_dias` |
| Transformacion del target | `log(1 + los_dias)` |
| Inversa | `exp(prediccion) - 1` |
| Regularizacion | No usa |
| Interacciones manuales | No usa |
| Variable `prob_los_14` | No usa |
| Variable `es_urgencia` | No usa, porque se entrena por segmento |
| Numero de features | 1650 por segmento |
| Archivos de modelo | `reg_lr_urgente.joblib`, `reg_lr_programado.joblib` |

