# Análisis de Urgencias y su Influencia en el LOS Hospitalario


---

## Metodología

El análisis parte del `dataset_maestro.csv` (11,951 pacientes), complementado con los archivos granulares `caso_diagnostico.csv` y `caso_procedimiento.csv`. La variable `es_urgencia` es un flag binario construido durante la limpieza de datos: toma valor 1 cuando el paciente tiene al menos un código `UUUUUU` entre sus diagnósticos — código administrativo que identifica admisiones de urgencia sin diagnóstico específico asignado al ingreso.

Para cada comparativa se reportan tests estadísticos robustos a la distribución asimétrica del LOS (Mann-Whitney U, Kolmogorov-Smirnov, Spearman), y se calcula el tamaño del efecto con Cohen's d sobre escala `log1p` para normalizar la variable.

---

## 1. Prevalencia de Urgencias

| Grupo | N | % del total |
|---|---|---|
| Con urgencia | 3,486 | **29.17%** |
| Sin urgencia | 8,465 | 70.83% |
| **Total** | **11,951** | 100% |

Casi 1 de cada 3 pacientes en el dataset es una admisión de urgencia. No es un grupo marginal — es una población lo suficientemente grande como para tener impacto significativo en cualquier modelo predictivo.

---

## 2. Impacto Directo en el LOS

### 2.1 Estadísticas descriptivas

| Métrica | Urgentes | No Urgentes | Diferencia |
|---|---|---|---|
| **Media** | **11.48 días** | 4.41 días | +7.07 días (+160%) |
| **Mediana** | **6.0 días** | 2.0 días | +4.0 días (+200%) |
| Desviación estándar | 18.28 días | 9.75 días | — |
| Mínimo | 0 días | 0 días | — |
| Máximo | 262 días | 224 días | — |
| P25 | 2 días | 1 día | +1 día |
| P75 | 14 días | 4 días | +10 días |
| P90 | 28 días | 8 días | +20 días |
| P95 | 40 días | 15 días | +25 días |
| P99 | 81.5 días | 46.0 días | +35.5 días |

**Conclusión directa:** Un paciente urgente tiene en promedio 2.6x más días de hospitalización que uno no urgente. En mediana, la diferencia es aún mayor: 3x. La brecha se amplifica dramáticamente en los percentiles altos — el P90 de los urgentes (28 días) ya supera el umbral PLOS que define a los pacientes de estancia prolongada.

### 2.2 Skewness y variabilidad

Ambos grupos son asimétricos, pero los no urgentes lo son más (skewness=8.19 vs 5.24 para urgentes). Esto es contraintuitivo: los urgentes tienen mayor media y mayor desviación estándar en términos absolutos, pero la distribución de los no urgentes está más "aplastada" en la base — casi todos se van en 1-2 días, pero los pocos que no lo hacen se van muy lejos.

---

## 3. Significancia Estadística

| Test | Estadístico | p-value | Conclusión |
|---|---|---|---|
| Mann-Whitney U | 20,847,771 | **1.65 × 10⁻²⁸⁸** | Diferencia extremadamente significativa |
| Kolmogorov-Smirnov | D = 0.3643 | **3.70 × 10⁻²⁹²** | Distribuciones completamente distintas |
| Correlación Spearman (es_urgencia ~ los_dias) | r = 0.3321 | **1.60 × 10⁻³⁰⁵** | Relación positiva significativa |
| Cohen's d (escala log1p) | **0.8146** | — | Efecto grande (umbral >0.8) |

La diferencia entre urgentes y no urgentes no es ruido estadístico: los p-values son prácticamente cero y el tamaño del efecto según Cohen's d (0.81) cruza el umbral convencional de "efecto grande" (>0.80). El hecho de que el `es_urgencia` tenga un Spearman r=0.33 con el LOS lo convierte en una de las variables más predictivas del dataset.

---

## 4. Distribución por Tramos de Estancia

| Tramo | Urgentes N | Urgentes % | No Urgentes N | No Urgentes % |
|---|---|---|---|---|
| 0 días | 133 | 3.8% | 117 | 1.4% |
| 1–2 días | 866 | **24.8%** | 4,796 | **56.7%** |
| 3–6 días | 893 | 25.6% | 2,507 | 29.6% |
| 7–13 días | 693 | 19.9% | 550 | 6.5% |
| 14–26 días | 521 | 14.9% | 266 | 3.1% |
| **27+ días (PLOS)** | **380** | **10.9%** | **229** | **2.7%** |

La distribución evidencia una bifurcación estructural:

- Los **no urgentes** se concentran masivamente en los tramos cortos: el 56.7% se va en 1-2 días y el 86.3% en menos de 7 días. Son episodios predecibles y de baja complejidad.
- Los **urgentes** tienen una distribución mucho más uniforme a lo largo de todos los tramos. Solo el 24.8% se va en 1-2 días, y acumulan casi el doble de proporción en cada tramo largo: 19.9% en 7-13 días (vs 6.5%), 14.9% en 14-26 días (vs 3.1%), y 10.9% en PLOS (vs 2.7%).

Visualmente, si se graficara la distribución de ambos grupos, la de no urgentes sería una curva exponencial decreciente muy pronunciada, mientras que la de urgentes sería una curva más plana con una cola derecha mucho más pesada.

---

## 5. Riesgo de Estancia Prolongada (PLOS ≥ 27 días)

| Grupo | N con PLOS | Total grupo | Tasa PLOS |
|---|---|---|---|
| Urgentes | 380 | 3,486 | **10.90%** |
| No urgentes | 229 | 8,465 | 2.71% |
| **Riesgo relativo** | — | — | **4.03x** |

Un paciente urgente tiene **4 veces más probabilidad de convertirse en PLOS** que uno no urgente. Este es quizás el hallazgo más relevante del análisis: la urgencia no solo aumenta el LOS promedio, sino que multiplica por 4 la probabilidad del escenario más costoso y difícil de manejar para el sistema hospitalario.

De los 609 pacientes PLOS totales, 380 (62.4%) son urgencias — a pesar de representar solo el 29.2% de la población total.

---

## 6. Urgentes con LOS = 0

De los 250 pacientes con LOS=0 en todo el dataset, 133 son urgencias (53.2%) y 117 son no urgencias (46.8%).

Esto requiere interpretación clínica cuidadosa. Un urgente con LOS=0 es un paciente que llegó a urgencias, recibió atención el mismo día, y fue dado de alta sin quedar internado. Sus características revelan que son un subgrupo diferenciado dentro de las urgencias:

- **Media de diagnósticos: 8.53** (vs 14.19 diagnósticos en urgentes con LOS > 0)
- Tienen prácticamente la mitad de la complejidad clínica que los urgentes que sí quedan internados
- Representan evaluaciones de urgencia de baja complejidad: triaje rápido, procedimientos diagnósticos ambulatorios, o estabilización sin necesidad de internación

**Implicación para modelos:** Los urgentes con LOS=0 no deben tratarse igual que los urgentes con LOS largo — son clínicamente distintos y el modelo necesita capturar esto. La interacción `es_urgencia × n_diag_total` es un predictor potencialmente poderoso.

---

## 7. Complejidad Clínica: Diagnósticos y Procedimientos

| Variable | Media Urgentes | Media No Urgentes | Ratio |
|---|---|---|---|
| **N° diagnósticos total** | **13.98** | 5.74 | **2.44x** |
| N° diagnósticos primarios | 2.56 | 1.16 | 2.21x |
| N° diagnósticos secundarios | 11.42 | 4.58 | 2.49x |
| N° procedimientos | 2.44 | 2.13 | 1.15x |

Los urgentes llegan con casi 2.5x más diagnósticos que los no urgentes. Esto refleja que una admisión de urgencia implica generalmente un cuadro más complejo: el paciente llega con múltiples comorbilidades activas, condiciones subyacentes que complican el tratamiento, y múltiples diagnósticos secundarios que se registran durante la admisión.

Los procedimientos, en cambio, son casi iguales (2.44 vs 2.13). Esto sugiere que la mayor complejidad de los urgentes se expresa más en diagnósticos que en intervenciones procedurales adicionales — el cuadro clínico es más complicado, pero no necesariamente más intervencionista en número de procedimientos.

### Correlación diagnósticos ~ LOS por grupo

| Variable | r Spearman (Urgentes) | r Spearman (No Urgentes) |
|---|---|---|
| N° diagnósticos total | **0.655** | 0.436 |
| N° diagnósticos secundarios | **0.645** | 0.421 |
| N° diagnósticos primarios | 0.412 | 0.384 |
| N° procedimientos | **0.388** | 0.215 |

La correlación entre número de diagnósticos y LOS es **sustancialmente más fuerte en urgentes**. Esto implica que en la población urgente, la complejidad clínica (medida por diagnósticos) tiene mayor poder explicativo sobre el tiempo de hospitalización. Para los no urgentes, otros factores (tipo de procedimiento, diagnóstico principal específico) probablemente dominan más.

---

## 8. Urgentes por Tramo: Complejidad creciente

| Tramo | N | Diag. Prom. | Proced. Prom. | LOS Prom. |
|---|---|---|---|---|
| 0 días | 133 | 8.53 | 1.35 | 0 |
| 1–2 días | 866 | 7.37 | 1.60 | 1.37 |
| 3–6 días | 893 | 10.92 | 1.96 | 4.30 |
| 7–13 días | 693 | 15.76 | 2.31 | 9.30 |
| 14–26 días | 521 | 20.36 | 2.96 | 18.59 |
| **27+ días (PLOS)** | **380** | **26.12** | **5.42** | **49.64** |

Existe una **escalera de complejidad perfectamente monotónica**: a mayor tramo de LOS, mayor número de diagnósticos y mayor número de procedimientos. Los urgentes PLOS (27+ días) tienen en promedio 26 diagnósticos y 5.4 procedimientos, frente a los 7.4 diagnósticos y 1.6 procedimientos del tramo 1-2 días.

El salto más notable ocurre en procedimientos al llegar a PLOS: de 2.96 en el tramo 14-26 días a 5.42 en el tramo 27+ días (83% de aumento). Esto sugiere que los casos de estancia muy prolongada no solo son más complejos en diagnósticos, sino que también requieren intervenciones activas adicionales (ventilación mecánica, diálisis, drenajes, etc.).

---

## 9. Diagnósticos más Frecuentes en Urgentes

### 9.1 Diagnósticos secundarios (comorbilidades presentes al ingreso)

| Código ICD-10 | Descripción | N pacientes | % de urgentes |
|---|---|---|---|
| I10 | Hipertensión esencial | 812 | 23.3% |
| Z7982 | Uso prolongado de aspirina | 728 | 20.9% |
| E559 | Deficiencia vitamínica inespecífica | 645 | 18.5% |
| E43 | Desnutrición proteico-calórica severa | 611 | 17.5% |
| E440 | Desnutrición proteico-calórica moderada | 554 | 15.9% |
| F17210 | Dependencia al tabaco, cigarrillos | 488 | 14.0% |
| E119 | Diabetes tipo 2 sin complicaciones | 446 | 12.8% |
| Z87891 | Historia personal de dependencia al tabaco | 431 | 12.4% |
| Z7901 | Uso prolongado de anticoagulantes | 427 | 12.2% |
| E669 | Obesidad inespecífica | 397 | 11.4% |
| E7800 | Hipercolesterolemia pura | 380 | 10.9% |
| N390 | Infección del tracto urinario | 371 | 10.6% |
| Z7984 | Uso prolongado de hipoglucemiantes orales | 356 | 10.2% |
| E860 | Deshidratación | 327 | 9.4% |
| W19XXXA | Caída no especificada (inicial) | 324 | 9.3% |

El perfil de comorbilidades de los urgentes describe un **paciente polimórbido crónico**: hipertensión, diabetes, obesidad, tabaquismo, dislipidemia, anticoagulación. Estos no son diagnósticos que causan la urgencia, sino condiciones preexistentes que complican el cuadro. La presencia de desnutrición (E43 + E440) en casi el 33% de los urgentes es clínicamente muy relevante — es un factor de riesgo independiente para estadías prolongadas.

### 9.2 Diagnósticos primarios (causa principal de la urgencia)

| Código ICD-10 | Descripción | N | % de urgentes |
|---|---|---|---|
| I214 | Infarto agudo de miocardio no-ST (NSTEMI) | 105 | 3.0% |
| J440 | EPOC con infección respiratoria baja aguda | 97 | 2.8% |
| J189 | Neumonía inespecífica | 95 | 2.7% |
| J441 | EPOC con exacerbación aguda | 93 | 2.7% |
| Z5189 | Procedimiento posterior (seguimiento) | 92 | 2.6% |
| Z48812 | Control postquirúrgico | 74 | 2.1% |
| K3580 | Apendicitis aguda | 66 | 1.9% |
| N390 | Infección del tracto urinario | 58 | 1.7% |
| I110 | Cardiopatía hipertensiva con insuficiencia cardíaca | 52 | 1.5% |
| J690 | Neumonitis por aspiración de sólidos/líquidos | 44 | 1.3% |

Los principales motivos de urgencia son patologías cardiovasculares (NSTEMI, insuficiencia cardíaca hipertensiva) y respiratorias (EPOC en exacerbación, neumonía, neumonitis por aspiración) — cuadros que clásicamente requieren hospitalizaciones prolongadas y monitoreo continuo.

---

## 10. Procedimientos en Urgentes vs No Urgentes

| Grupo | N procedimientos | N pacientes | Prom/paciente |
|---|---|---|---|
| Urgentes | 8,519 | 3,486 | **2.44** |
| No urgentes | 18,049 | 8,465 | 2.13 |

A pesar de ser un grupo más pequeño, los urgentes reciben más procedimientos per cápita. La distribución por sección ICD-10-PCS en urgentes:

| Sección | Descripción | N | % de proc. urgentes |
|---|---|---|---|
| 0 | Cirugía Médica y Quirúrgica | 5,427 | **63.7%** |
| 3 | Administración (transfusiones, infusiones) | 1,349 | **15.8%** |
| B | Diagnóstico por imagen | 710 | 8.3% |
| 5 | Asistencia extracorpórea / sistémica | 423 | **5.0%** |
| 4 | Medición y monitoreo | 294 | 3.5% |
| 2 | Colocación (vendajes, drenajes) | 112 | 1.3% |

El 15.8% de procedimientos en la sección de Administración (transfusiones, fluidos IV, medicación) es particularmente alto para urgentes — refleja la necesidad de estabilización aguda. La presencia del 5% en Asistencia Extracorpórea (ventilación mecánica, soporte hemodinámico) apunta a los casos más graves.

---

## 11. Diagnósticos en Urgentes con PLOS (la combinación más crítica)

De los 380 urgentes que terminan en PLOS (≥27 días), estos son los diagnósticos secundarios más prevalentes:

| Código | Descripción | N | % de urgentes PLOS |
|---|---|---|---|
| E43 | Desnutrición severa | 163 | **42.9%** |
| E559 | Deficiencia vitamínica | 129 | 33.9% |
| N390 | Infección tracto urinario | 121 | **31.8%** |
| E440 | Desnutrición moderada | 119 | 31.3% |
| Z7982 | Uso prolongado de aspirina | 106 | 27.9% |
| I10 | Hipertensión | 103 | 27.1% |
| Z7901 | Uso prolongado anticoagulantes | 76 | 20.0% |
| Z87891 | Historia de tabaquismo | 71 | 18.7% |
| W19XXXA | Caída no especificada | 67 | 17.6% |
| K449 | Hernia diafragmática | 64 | 16.8% |
| B9620 | Infección por *E. coli* | 64 | 16.8% |
| E119 | Diabetes tipo 2 | 64 | 16.8% |
| F17210 | Dependencia tabaco | 63 | 16.6% |
| E860 | Deshidratación | 58 | 15.3% |
| E7800 | Hipercolesterolemia | 57 | 15.0% |

La **desnutrición severa (E43) aparece en el 42.9% de los urgentes PLOS** — casi uno de cada dos. Junto con la deficiencia vitamínica (33.9%) y la deshidratación (15.3%), emerge un perfil de **paciente urgente de muy alto riesgo: adulto mayor, desnutrido, con infección bacteriana activa (N390, B9620), politratado (anticoagulantes, aspirina, hipoglucemiantes) y con antecedentes de caídas**. Este perfil de "fragilidad clínica" es el que más consistentemente produce estancias extremadamente largas.

---

## 12. Análisis Temporal

### Por año

| Año | Urgentes | Total | % Urgentes |
|---|---|---|---|
| 2017 | 207 | 313 | **66.1%** |
| 2018 | 3,279 | 11,638 | 28.2% |

El año 2017 solo contiene 313 pacientes (probablemente los últimos meses del año), lo que hace que su alta tasa de urgencias (66.1%) no sea representativa del comportamiento general. El grueso del análisis se apoya en 2018.

### Por mes (datos 2018 principalmente)

| Mes | Urgentes | Total | % Urgentes |
|---|---|---|---|
| Enero | 605 | 2,128 | 28.4% |
| Febrero | 510 | 1,834 | 27.8% |
| Marzo | 664 | 2,213 | **30.0%** |
| Abril | 562 | 1,879 | 29.9% |
| Mayo | 560 | 1,992 | 28.1% |
| Junio | 378 | 1,592 | **23.7%** |
| Diciembre | 170 | 250 | **68.0%** |

Los meses de enero a mayo tienen una tasa de urgencias estable (~28-30%). Junio baja a 23.7%, posiblemente porque la demanda total de hospitalización baja. Diciembre tiene 68% de urgencias, pero el N es pequeño (250 pacientes) — podría reflejar que en diciembre se postergan procedimientos electivos y solo atienden urgencias reales.

Los meses julio a noviembre tienen n < 10 pacientes y tasas anómalas (hasta 83% en agosto) — son residuos del dataset de 2017 y no son representativos.

---

## 13. Síntesis y Conclusiones

### Hallazgos principales

**1. La urgencia es el predictor individual más potente del LOS en el dataset.**
Con un Spearman r=0.33 (p≈0), Cohen's d=0.81 (efecto grande), y una diferencia de medias de 7 días, `es_urgencia` es probablemente la feature más importante del modelo — incluso antes de considerar diagnósticos específicos.

**2. Las urgencias duplican la media y triplican la mediana del LOS.**
Media: 11.48 vs 4.41 días (×2.6). Mediana: 6 vs 2 días (×3.0). La mediana es especialmente relevante porque es robusta a outliers — el efecto no lo están produciendo unos pocos casos extremos, sino una diferencia estructural en toda la distribución.

**3. Los urgentes tienen 4× más riesgo de estancia prolongada (PLOS ≥ 27 días).**
Tasa PLOS: 10.90% en urgentes vs 2.71% en no urgentes (RR = 4.03). El 62.4% de todos los pacientes PLOS son urgencias, pese a que solo representan el 29.2% de la población total.

**4. La complejidad clínica en urgentes es 2.5× mayor.**
Los urgentes tienen en promedio 14.0 diagnósticos vs 5.7 en no urgentes. Esta complejidad está correlacionada con el LOS de forma más fuerte en urgentes (r=0.655) que en no urgentes (r=0.436) — el número de diagnósticos "explica más" el LOS cuando el paciente es urgente.

**5. El perfil de urgente-PLOS es identificable: fragilidad clínica.**
La combinación desnutrición + infección bacteriana (UTI, E. coli) + hipertensión + politratamiento crónico + antecedentes de caídas identifica al subgrupo de urgentes con mayor riesgo de hospitalización extremadamente larga. Casi la mitad de los urgentes PLOS presentan desnutrición severa (E43).

**6. Los urgentes con LOS=0 son un subgrupo distinto.**
133 urgentes son dados de alta el mismo día (3.8% de las urgencias). Tienen casi la mitad de diagnósticos que los urgentes con LOS>0 (8.53 vs 14.19). Son evaluaciones de urgencia de baja complejidad, no equivalentes a los urgentes que requieren internación.

**7. El número de procedimientos escala abruptamente en PLOS.**
De 2.96 procedimientos promedio en el tramo 14-26 días a 5.42 en el tramo 27+ días. Este salto (83%) es señal de que los casos PLOS requieren intervenciones activas adicionales que van más allá de la complejidad diagnóstica.

### Implicaciones para el modelo predictivo

| Implicación | Acción recomendada |
|---|---|
| `es_urgencia` es altamente predictivo | Confirmar que el modelo la está usando como feature de alta importancia |
| La interacción urgencia × diagnósticos es fuerte | Considerar feature `es_urgencia × n_diag_total` explícitamente |
| Urgentes-LOS=0 son clínicamente distintos | Evaluar modelo separado o flag adicional para urgente de baja complejidad |
| La tasa PLOS es 4× más alta en urgentes | En estrategias de alerta temprana, priorizar urgentes con comorbilidades graves |
| Los diagnósticos E43, E559, N390 en urgentes predicen PLOS | Considerar features basadas en la presencia de estos códigos específicos |

---

