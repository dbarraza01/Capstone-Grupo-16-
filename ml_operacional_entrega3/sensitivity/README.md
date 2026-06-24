# Analisis de sensibilidad

Esta carpeta implementa el plan de `sensitivity_analysis_plan.md` para el
pipeline XGBoost operacional de dos etapas.

Los escenarios 1, 2 y 4 reentrenan modelos temporales en memoria. No
sobrescriben los modelos guardados en `ml_operacional_entrega3/modelos_guardados/`.

## Comandos recomendados

Validar rutas, splits y disponibilidad de predicciones sin entrenar:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --dry-run
```

Ejecutar una prueba de humo liviana sin escribir outputs finales:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --smoke-test
```

Ver estado de escenarios y outputs sin ejecutar entrenamientos:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --status
```

Ejecutar la corrida pesada y dejar solo CSVs:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --skip-report
```

Reanudar una corrida pesada sin repetir escenarios ya completados:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --skip-report --skip-existing
```

Generar el informe consolidado cuando ya existan todos los CSVs:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --report-only
```

Validar outputs finales:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --validate-results
```

## Ejecucion por partes

Tambien se puede ejecutar un escenario aislado:

```bash
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --scenario 1 --skip-report
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --scenario 2 --skip-report
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --scenario 3 --skip-report
python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --scenario 4 --skip-report
```

Cuando los cuatro escenarios hayan terminado, generar el reporte con
`--report-only`.

## Salidas esperadas

Los resultados se guardan en `ml_operacional_entrega3/sensitivity/results/`:

1. `escenario_1_resultados.csv`
2. `escenario_1_resultados_por_tramo.csv`
3. `escenario_2_resultados.csv`
4. `escenario_3_curva_pr.csv`
5. `escenario_3_puntos_operacion.csv`
6. `escenario_4_resultados.csv`
7. `reporte_sensibilidad_consolidado.md`

El script maestro imprime el numero total de salidas esperadas, el numero de
salidas generadas en cada corrida y la cantidad total de filas/lineas generadas.
Ademas, cada corrida no-dry-run deja un manifest de trazabilidad en
`results/execution_manifest.json`.
