# ¿Qué muestra el gráfico de Verosimilitud vs. Weight?

## El problema de fondo

Cuando graficamos los días de estadía hospitalaria (LOS) de todos los pacientes, la curva resultante tiene una forma que **ninguna distribución estándar explica bien por sí sola**. Esto es una señal: probablemente hay dos tipos de pacientes mezclados en los datos, cada uno con una dinámica distinta.

El gráfico responde la pregunta: *¿en qué proporción se mezclan esos dos grupos para explicar mejor los datos?*

---

## Las dos distribuciones y qué representan

El modelo usa dos distribuciones como componentes de la mezcla.

La **Log-Normal** describe bien situaciones donde la duración de algo está determinada por muchos pasos secuenciales predecibles. En el contexto hospitalario, modela pacientes cuya estadía tiene un "tiempo típico" alrededor del cual se concentra la mayoría: cirugías electivas, casos médicos sin complicaciones, donde el flujo de recuperación es bastante estándar.

La **Weibull** es clásica en análisis de supervivencia porque modela situaciones donde la probabilidad de que ocurra un evento (como el alta) **cambia a medida que pasa el tiempo**. Describe mejor pacientes de alta complejidad, donde cuanto más tiempo llevan internados, más difícil es predecir cuándo serán dados de alta.

---

## ¿Qué es la verosimilitud y por qué importa?

La **verosimilitud** responde una sola pregunta: *¿qué tan bien este modelo reproduce los datos que tenemos?* Más precisamente, es la probabilidad de haber observado exactamente los datos de la muestra, asumiendo que el modelo propuesto es correcto. Un modelo bueno produce una verosimilitud alta; uno malo, una verosimilitud baja.

Como calcular esto implica multiplicar miles de probabilidades muy pequeñas, el número resultante sería tan cercano a cero que la computadora no podría manejarlo. Por eso se usa el **logaritmo**: transforma esas multiplicaciones en sumas, produciendo números manejables, sin perder el punto óptimo (el máximo sigue siendo el mismo).

---

## Cómo actúa el algoritmo

El algoritmo introduce una variable llamada **weight**, que representa qué proporción de la mezcla corresponde a Log-Normal. Si weight = 0.42, entonces el modelo dice: el 42% de los pacientes sigue una dinámica Log-Normal y el 58% restante sigue Weibull.

El algoritmo prueba sistemáticamente todos los valores posibles de weight, de 0% a 100%. Para cada valor, calcula la log-verosimilitud de ese modelo sobre todos los datos. El valor de weight que produce la log-verosimilitud más alta es el óptimo, y ese es el punto que aparece marcado con la estrella roja en el gráfico.

Lo que hace esto riguroso es que para cada valor fijo de weight, el algoritmo también optimiza los parámetros internos de cada distribución (forma y escala), asegurando que cada proporción compita en su mejor versión posible.

---

## ¿Qué muestra el gráfico concretamente?

El **eje X** es el valor del weight, de 0% (modelo 100% Weibull) a 100% (modelo 100% Log-Normal). El **eje Y** es la log-verosimilitud: más alto significa mejor ajuste a los datos reales.

La curva azul sube y baja a medida que se cambia el weight. El hecho de que tenga un pico en un punto intermedio (y no en los extremos) confirma que **ninguna distribución sola es suficiente**: la mezcla gana. En este caso, el óptimo está en weight ≈ 42%, lo que significa que la combinación 42% Log-Normal + 58% Weibull es la que mejor explica los datos de LOS.

La línea punteada naranja marca el nivel de log-verosimilitud del modelo Log-Normal puro. Todo lo que está por encima de esa línea es una mezcla que lo supera, y el pico indica una mejora de +390% respecto a usar Log-Normal sola.

---

## ¿Por qué 42% Log-Normal y 58% Weibull?

El algoritmo no "separa" pacientes en grupos ni etiqueta a nadie. Lo que hace es buscar la forma de curva combinada que mejor calce con la distribución real de los datos. Que el óptimo sea 42-58 significa que esa proporción produce la curva más parecida al histograma real de estadías.

La interpretación clínica se construye *después*, combinando lo que las matemáticas revelan con el conocimiento del contexto hospitalario: los pacientes del componente Log-Normal son probablemente los de estadía predecible y flujo estándar, mientras que los del componente Weibull son los casos complejos donde el tiempo de alta depende de eventos clínicos difíciles de anticipar.

> **En resumen:** el gráfico muestra que los datos de LOS tienen estructura mixta, que una sola distribución no es suficiente para modelarlos, y que la mezcla óptima es 42% Log-Normal + 58% Weibull, encontrada maximizando la log-verosimilitud sobre todos los posibles valores de weight.
