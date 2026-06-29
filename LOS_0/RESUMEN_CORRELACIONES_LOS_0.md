# Análisis de correlaciones entre diagnósticos y procedimientos en pacientes con LOS = 0

## Datos generales

- Total de pacientes: 250.
- Total de diagnósticos registrados: 1.890.
- Total de procedimientos registrados: 438.
- Diagnósticos principales analizados: 10 más frecuentes.
- Procedimientos analizados: 12 más frecuentes.

## Diagnósticos principales más frecuentes

| Posición | Código | Frecuencia | Porcentaje |
|---:|---|---:|---:|
| 1 | UUUUUU | 133 | 32,0% |
| 2 | N201 | 7 | 1,7% |
| 3 | N132 | 6 | 1,4% |
| 4 | O0010 | 5 | 1,2% |
| 5 | Z09 | 4 | 1,0% |
| 6 | D122 | 4 | 1,0% |
| 7 | R197 | 4 | 1,0% |
| 8 | K2950 | 4 | 1,0% |
| 9 | R109 | 4 | 1,0% |
| 10 | J441 | 4 | 1,0% |

## Procedimientos más frecuentes

| Posición | Código | Frecuencia | Porcentaje |
|---:|---|---:|---:|
| 1 | 0DB98ZX | 18 | 4,1% |
| 2 | 0DJD8ZZ | 17 | 3,9% |
| 3 | 0DB68ZX | 16 | 3,7% |
| 4 | 0DBB8ZX | 12 | 2,7% |
| 5 | 0DBK8ZZ | 12 | 2,7% |
| 6 | 0DJ08ZZ | 11 | 2,5% |
| 7 | 30233N1 | 10 | 2,3% |
| 8 | 0DBH8ZZ | 9 | 2,1% |
| 9 | 0DBL8ZZ | 9 | 2,1% |
| 10 | 0DBN8ZZ | 9 | 2,1% |
| 11 | 0DBP8ZZ | 8 | 1,8% |
| 12 | 0DBM8ZZ | 8 | 1,8% |

## Hallazgos principales

### Coocurrencia

Se identificaron 34 combinaciones entre diagnósticos y procedimientos. La asociación más frecuente fue `UUUUUU` con `0DJ08ZZ`, observada en ocho ocasiones.

### Diagnósticos con mayor diversidad de procedimientos

- `UUUUUU` coocurre con cinco procedimientos diferentes.
- `K2950` coocurre con ocho procedimientos diferentes.

### Procedimientos asociados con más diagnósticos

- `0DB98ZX` se asocia con cuatro diagnósticos principales diferentes.
- `0DJ08ZZ` se asocia con dos diagnósticos principales diferentes.

### Patrones observados

- Varios procedimientos se asocian con múltiples diagnósticos, configurando un patrón de concentración o *hub*.
- Algunos diagnósticos presentan asociaciones con procedimientos más específicos.
- La variabilidad de diagnósticos vinculados con procedimientos evidencia diversidad clínica dentro del grupo con LOS = 0.

## Implicaciones clínicas

- Los procedimientos de las familias `0DB*` y `0DJ*` se utilizan en pacientes con diversos diagnósticos.
- Determinados diagnósticos podrían aportar información para anticipar los procedimientos más probables.
- La matriz de coocurrencia permite identificar asociaciones frecuentes e inusuales entre diagnósticos y procedimientos.

## Archivos asociados

- `04_heatmap_correlaciones_los_0.png`: matriz de coocurrencias.
- `05_procedimientos_por_diagnostico_los_0.png`: cinco procedimientos principales por diagnóstico.
- `06_diagnosticos_por_procedimiento_los_0.png`: cinco diagnósticos principales por procedimiento.
- `correlaciones_diagnostico_procedimiento_los_0.csv`: datos completos de coocurrencia.
