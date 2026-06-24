# Reporte Consolidado de Analisis de Sensibilidad

## 1. Resumen Ejecutivo

Este informe consolida los cuatro escenarios definidos para evaluar la robustez del pipeline XGBoost operacional de dos etapas.

### Impacto global sobre MAE

| escenario | variante | mae | delta_mae_pct | veredicto |
| --- | --- | --- | --- | --- |
| 1 - Umbral PLOS | LOS >= 7 | 2.8474 | -0.4392 | robusto |
| 1 - Umbral PLOS | LOS >= 14 | 2.8600 | 0.0000 | robusto |
| 1 - Umbral PLOS | LOS >= 21 | 2.8276 | -1.1329 | robusto |
| 1 - Umbral PLOS | LOS >= 27 | 2.8640 | 0.1421 | robusto |
| 2 - Ablation | Full (linea base) | 2.8600 | 0.0000 | robusto |
| 2 - Ablation | Sin Charlson | 2.9109 | 1.7823 | robusto |
| 2 - Ablation | Sin capitulos ICD-10 | 2.8660 | 0.2117 | robusto |
| 2 - Ablation | Solo demografico-operacional | 3.4529 | 20.7306 | sensibilidad alta |
| 2 - Ablation | Sin Clasificador (1 Etapa) | 2.8441 | -0.5547 | robusto |
| 4 - Hiperparametros | Conservadora (mas regularizada) | 2.8752 | 0.5308 | robusto |
| 4 - Hiperparametros | Compleja (menos regularizada) | 2.8863 | 0.9202 | robusto |
| 4 - Hiperparametros | Perturbacion estocastica de muestreo | 2.8912 | 1.0910 | robusto |

**Veredicto general:** El pipeline muestra sensibilidad en al menos un escenario; se debe discutir como limitacion.

## 2. Justificacion Bibliografica

- La definicion de estancia prolongada no es universal; se reportan umbrales entre 7, 14, 24/27 dias y percentiles especificos de cohorte.
- Los estudios de dos etapas para LOS justifican separar una decision binaria de riesgo PLOS y una estimacion continua de dias.
- En datos clinicos desbalanceados, la curva Precision-Recall es mas informativa que mirar solo accuracy o MAE.
- La estabilidad frente a hiperparametros vecinos valida que el resultado no depende de una configuracion demasiado fragil.

Referencias base: Bergstra y Bengio (2012), Chrusciel et al. (2022), Goldstein et al. (2022), Lee et al. (2024), Mahajan et al. (2023), Probst et al. (2019), Saito y Rehmsmeier (2015).

## 3. Analisis por Escenario

### Escenario 1 - Variacion del umbral PLOS

Pregunta: si cambia la definicion clinica de estancia prolongada, se mantiene estable el desempeno del modelo?

| umbral_plos | n_casos | mae | rmse | me | pup | mae_asimetrico | precision_plos | recall_plos | f1_plos | proporcion_plos | n_plos_real | n_plos_pred | delta_mae_pct_vs_umbral_14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.0000 | 2391.0000 | 2.8474 | 7.0502 | -1.0002 | 0.4316 | 4.7712 | 0.8400 | 0.6364 | 0.7241 | 0.2208 | 528.0000 | 400.0000 | -0.4392 |
| 14.0000 | 2391.0000 | 2.8600 | 7.1668 | -0.9624 | 0.4471 | 4.7712 | 0.8000 | 0.5878 | 0.6777 | 0.1167 | 279.0000 | 205.0000 | 0.0000 |
| 21.0000 | 2391.0000 | 2.8276 | 7.0460 | -0.9516 | 0.4517 | 4.7171 | 0.7890 | 0.4914 | 0.6056 | 0.0732 | 175.0000 | 109.0000 | -1.1329 |
| 27.0000 | 2391.0000 | 2.8640 | 7.0406 | -1.0043 | 0.4626 | 4.7982 | 0.7468 | 0.4836 | 0.5871 | 0.0510 | 122.0000 | 79.0000 | 0.1421 |

Desglose por tramo LOS:

| umbral_plos_analizado | tramo | n_casos | mae | rmse | medae | me | pup | mae_asimetrico | pct_error_abs_le_1d | pct_error_abs_le_3d | pct_error_abs_le_7d | los_real_promedio | los_pred_promedio | umbral_plos | proporcion_plos | n_plos_real | n_plos_pred | precision_plos | recall_plos | f1_plos | accuracy_plos | tp_plos | fp_plos | fn_plos | tn_plos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 0-2 | 1183 | 0.7850 | 1.5713 | 0.4549 | 0.6874 | 0.1395 | 0.8338 | 0.7937 | 0.9577 | 0.9941 | 1.2299 | 1.9174 | 7 | 0.0000 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9899 | 0 | 12 | 0 | 1171 |
| 7 | 3-6 | 680 | 1.6100 | 2.6564 | 1.0556 | -0.2221 | 0.6912 | 2.5261 | 0.4779 | 0.8838 | 0.9706 | 4.0324 | 3.8103 | 7 | 0.0000 | 0 | 52 | 0.0000 | 0.0000 | 0.0000 | 0.9235 | 0 | 52 | 0 | 628 |
| 7 | 7-13 | 249 | 4.0652 | 5.0147 | 3.5940 | -1.6973 | 0.7269 | 6.9464 | 0.1205 | 0.4217 | 0.8876 | 9.0281 | 7.3308 | 7 | 1.0000 | 249 | 102 | 1.0000 | 0.4096 | 0.5812 | 0.4096 | 102 | 0 | 147 | 0 |
| 7 | 14+ (PLOS) | 279 | 13.5213 | 19.3870 | 10.3466 | -9.4305 | 0.7742 | 24.9972 | 0.0394 | 0.1111 | 0.3297 | 30.9821 | 21.5515 | 7 | 1.0000 | 279 | 234 | 1.0000 | 0.8387 | 0.9123 | 0.8387 | 234 | 0 | 45 | 0 |
| 14 | 0-2 | 1183 | 0.7935 | 1.6208 | 0.4754 | 0.7041 | 0.1716 | 0.8382 | 0.7946 | 0.9637 | 0.9924 | 1.2299 | 1.9341 | 14 | 0.0000 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9966 | 0 | 4 | 0 | 1179 |
| 14 | 3-6 | 680 | 1.5665 | 2.6294 | 0.9859 | -0.2235 | 0.6868 | 2.4616 | 0.5044 | 0.8912 | 0.9721 | 4.0324 | 3.8089 | 14 | 0.0000 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.9824 | 0 | 12 | 0 | 668 |
| 14 | 7-13 | 249 | 4.0996 | 4.9236 | 3.6032 | -1.7626 | 0.7189 | 7.0307 | 0.1004 | 0.4016 | 0.8755 | 9.0281 | 7.2655 | 14 | 0.0000 | 0 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.8996 | 0 | 25 | 0 | 224 |
| 14 | 14+ (PLOS) | 279 | 13.6682 | 19.7624 | 10.4781 | -9.1154 | 0.7885 | 25.0600 | 0.0573 | 0.1649 | 0.3226 | 30.9821 | 21.8667 | 14 | 1.0000 | 279 | 164 | 1.0000 | 0.5878 | 0.7404 | 0.5878 | 164 | 0 | 115 | 0 |
| 21 | 0-2 | 1183 | 0.7996 | 1.6051 | 0.4900 | 0.7106 | 0.1927 | 0.8441 | 0.7802 | 0.9645 | 0.9924 | 1.2299 | 1.9405 | 21 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 0 | 1183 |
| 21 | 3-6 | 680 | 1.5247 | 2.4806 | 0.9984 | -0.2428 | 0.6779 | 2.4085 | 0.5000 | 0.8941 | 0.9735 | 4.0324 | 3.7895 | 21 | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9985 | 0 | 1 | 0 | 679 |
| 21 | 7-13 | 249 | 4.1507 | 4.9962 | 3.7161 | -1.8366 | 0.7349 | 7.1444 | 0.1084 | 0.3775 | 0.8795 | 9.0281 | 7.1916 | 21 | 0.0000 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.9799 | 0 | 5 | 0 | 244 |
| 21 | 14+ (PLOS) | 279 | 13.4212 | 19.4233 | 10.1565 | -8.9367 | 0.7455 | 24.6001 | 0.0502 | 0.1505 | 0.3369 | 30.9821 | 22.0454 | 21 | 0.6272 | 175 | 103 | 0.8350 | 0.4914 | 0.6187 | 0.6201 | 86 | 17 | 89 | 87 |
| 27 | 0-2 | 1183 | 0.7848 | 1.5354 | 0.4689 | 0.6918 | 0.1986 | 0.8313 | 0.7836 | 0.9628 | 0.9941 | 1.2299 | 1.9217 | 27 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 0 | 1183 |
| 27 | 3-6 | 680 | 1.5545 | 2.4922 | 1.0557 | -0.2596 | 0.6824 | 2.4616 | 0.4706 | 0.8912 | 0.9779 | 4.0324 | 3.7728 | 27 | 0.0000 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.9985 | 0 | 1 | 0 | 679 |
| 27 | 7-13 | 249 | 4.2215 | 5.1445 | 3.8018 | -1.8139 | 0.7430 | 7.2393 | 0.0964 | 0.3695 | 0.8594 | 9.0281 | 7.2142 | 27 | 0.0000 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.9920 | 0 | 2 | 0 | 247 |
| 27 | 14+ (PLOS) | 279 | 13.6606 | 19.3921 | 10.2729 | -9.2886 | 0.7957 | 25.1352 | 0.0538 | 0.1254 | 0.3405 | 30.9821 | 21.6935 | 27 | 0.4373 | 122 | 76 | 0.7763 | 0.4836 | 0.5960 | 0.7133 | 59 | 17 | 63 | 140 |

Interpretacion: El menor MAE aparece con umbral LOS >= 21 (MAE=2.828). La mayor desviacion frente al umbral 14 ocurre con LOS >= 21, delta=-1.1%. 4 de 4 umbrales quedan dentro del margen de robustez de 5%. En este escenario se recalcula `scale_pos_weight` de forma deterministica porque cambia la prevalencia de PLOS; no se realiza re-tuning.

### Escenario 2 - Ablation study

Pregunta: que componentes son indispensables y que se pierde al removerlos?

| variante | pregunta | n_casos | mae | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full (linea base) | Referencia completa del pipeline en dos etapas. | 2391 | 2.8600 | 7.1668 | 0.5878 | 0.6777 | 0.8000 | 4.7712 | 0.0000 |
| Sin Charlson | Mide la dependencia del indice de comorbilidad Charlson. | 2391 | 2.9109 | 7.3094 | 0.5806 | 0.6750 | 0.8060 | 4.8493 | 1.7823 |
| Sin capitulos ICD-10 | Mide el aporte marginal de agrupaciones raras por capitulo. | 2391 | 2.8660 | 7.1804 | 0.6057 | 0.6969 | 0.8204 | 4.7881 | 0.2117 |
| Solo demografico-operacional | Mide cuanto se pierde sin dummies clinicas detalladas. | 2391 | 3.4529 | 8.1420 | 0.4659 | 0.5791 | 0.7647 | 5.8839 | 20.7306 |
| Sin Clasificador (1 Etapa) | Mide si la probabilidad PLOS de la etapa 1 aporta valor neto. | 2391 | 2.8441 | 6.9792 | 0.5986 | 0.7046 | 0.8564 | 4.7981 | -0.5547 |

Interpretacion: La ablacion mas sensible es `Solo demografico-operacional` (MAE=3.453, delta=20.7%). Si una variante degrada poco el MAE, el bloque removido es complementario o redundante; si lo degrada mucho, ese bloque sostiene parte importante de la prediccion operacional. La variante sin clasificador cambia el MAE en -0.6%; ese valor cuantifica el aporte neto de usar la probabilidad PLOS como senal para el regresor.

### Escenario 3 - Punto de operacion del clasificador

Pregunta: que politica de alerta conviene para gestion de camas?

| politica_clinica | umbral_probabilidad | tp | fp | fn | tn | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Politica B - Alta Seguridad / Alto Recall | 0.3500 | 245 | 229 | 34 | 1883 | 0.5169 | 0.8781 | 0.6507 | 0.8900 |
| Politica A - Base / Equilibrio | 0.5000 | 229 | 148 | 50 | 1964 | 0.6074 | 0.8208 | 0.6982 | 0.9172 |
| Politica C - Eficiencia / Alertas Confiables | 0.6500 | 204 | 94 | 75 | 2018 | 0.6846 | 0.7312 | 0.7071 | 0.9293 |

Interpretacion: La politica con mayor recall es `Politica B - Alta Seguridad / Alto Recall` (recall=0.878, FN=34). La politica con mayor precision es `Politica C - Eficiencia / Alertas Confiables` (precision=0.685, FP=94). El mejor F1 entre las politicas predefinidas lo obtiene `Politica C - Eficiencia / Alertas Confiables` (F1=0.707).

Recomendacion: usar Politica B - Alta Seguridad / Alto Recall (umbral=0.35) cuando el hospital priorice no perder pacientes PLOS. Entrega el mayor recall, a costa de mas falsos positivos.

### Escenario 4 - Hiperparametros vecinos

Pregunta: el tuning encontrado esta en una region estable o fragil?

| variante_hiperparametros | n_casos | mae | rmse | recall_plos | f1_plos | precision_plos | mae_asimetrico | delta_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conservadora (mas regularizada) | 2391 | 2.8752 | 7.1739 | 0.5914 | 0.6832 | 0.8088 | 4.8205 | 0.5308 |
| Compleja (menos regularizada) | 2391 | 2.8863 | 7.3926 | 0.6201 | 0.7047 | 0.8160 | 4.8192 | 0.9202 |
| Perturbacion estocastica de muestreo | 2391 | 2.8912 | 7.1004 | 0.5771 | 0.6736 | 0.8090 | 4.8313 | 1.0910 |

Interpretacion: La mayor variacion parametrica observada es `Perturbacion estocastica de muestreo` (MAE=2.891, delta=1.1%). 3 de 3 variantes quedan dentro del margen de robustez de 5%. Esto permite evaluar si el tuning esta en una region plana o si depende de un punto fragil.

## 4. Limitaciones y Mejoras Propuestas

- Los escenarios no re-tunean hiperparametros; esto aisla el efecto de cada perturbacion, pero no mide el mejor rendimiento posible bajo cada nuevo supuesto.
- El holdout permanece fijo para comparabilidad estricta; una validacion externa en otra clinica seria la prueba mas fuerte de generalizacion.
- El punto de operacion del clasificador debe elegirse con gestores de camas, porque el costo de falsos positivos y falsos negativos no es puramente estadistico.

## 5. Veredicto de Robustez y Toma de Decisiones Clinicas

Maxima desviacion absoluta de MAE observada en escenarios 1, 2 y 4: 20.73%.

El pipeline muestra sensibilidad en al menos un escenario; se debe discutir como limitacion.

Recomendacion: usar Politica B - Alta Seguridad / Alto Recall (umbral=0.35) cuando el hospital priorice no perder pacientes PLOS. Entrega el mayor recall, a costa de mas falsos positivos.

## 6. Respuestas Directas por Escenario

- Escenario 1: si las desviaciones frente al umbral 14 son menores a 5%, la conclusion no depende criticamente de definir PLOS como 14 dias.
- Escenario 3: la politica de umbral debe elegirse segun el costo operacional: mayor recall reduce pacientes PLOS no detectados; mayor precision reduce fatiga por alertas.
- Escenario 4: variantes vecinas bajo 5% de delta MAE indican que el tuning es estable; sobre 15% indica fragilidad parametrica.
- Escenario 2 / Full (linea base): MAE=2.860, delta=0.0%. Referencia completa del pipeline en dos etapas.
- Escenario 2 / Sin Charlson: MAE=2.911, delta=1.8%. Mide la dependencia del indice de comorbilidad Charlson.
- Escenario 2 / Sin capitulos ICD-10: MAE=2.866, delta=0.2%. Mide el aporte marginal de agrupaciones raras por capitulo.
- Escenario 2 / Solo demografico-operacional: MAE=3.453, delta=20.7%. Mide cuanto se pierde sin dummies clinicas detalladas.
- Escenario 2 / Sin Clasificador (1 Etapa): MAE=2.844, delta=-0.6%. Mide si la probabilidad PLOS de la etapa 1 aporta valor neto.
- Veredicto cuantitativo: hay variantes con sensibilidad alta; deben reportarse como limitacion critica.
