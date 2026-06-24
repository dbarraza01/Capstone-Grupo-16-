# Paquete de tablas y gráficos - modelo operacional en dos etapas

## Contenido
- `tablas_png/`: tablas listas para pegar en PPT.
- `graficos_png/`: gráficos 16:9 con estilo consistente con la presentación.
- `csv/`: datos fuente, IC95 bootstrap, diferencias pareadas e intervalos de predicción empíricos.
- `metodologia/`: notas de trazabilidad y advertencias metodológicas.

## Archivos clave para presentar
1. `graficos_png/grafico_16_dashboard_resumen_presentacion.png`
2. `tablas_png/tabla_01_resumen_global_ic95.png`
3. `graficos_png/grafico_04_plos14_precision_recall_f1_ic95.png`
4. `graficos_png/grafico_05_mae_por_tramo_ic95.png`
5. `graficos_png/grafico_08_subestimacion_por_tramo_ic95.png`
6. `tablas_png/tabla_05_diferencias_pareadas_xgb_menos_lr.png`
7. `graficos_png/grafico_12_xgb_cobertura_ip90_por_tramo_y_metodo.png`
8. `graficos_png/grafico_13_xgb_ancho_ip90_por_tramo_y_metodo.png`

## Metodología
- PLOS se define como `LOS >= 14` días.
- Tramos: `0-2`, `3-6`, `7-13`, `14+ (PLOS)`.
- Los IC95 de métricas se calcularon mediante bootstrap no paramétrico sobre el holdout, remuestreando pacientes con reemplazo.
- Las diferencias XGB - LR se calcularon con bootstrap pareado por `case_id`.
- Los intervalos de predicción IP90 son empíricos/aparentes: usan cuantiles de residuos del holdout. Sirven para diagnóstico, pero no son conformal prediction plenamente validado.

## Advertencia rigurosa sobre IP90
Para presentar intervalos de predicción como validados se necesitaría separar un set de calibración o guardar predicciones out-of-fold del pipeline completo, incluyendo el regresor. El zip actual trae probabilidades OOF del clasificador, pero no predicciones OOF finales del regresor; por eso no inventé IP conformales.
