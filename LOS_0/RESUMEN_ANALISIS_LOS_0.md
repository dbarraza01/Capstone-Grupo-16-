# Análisis de pacientes con LOS = 0

## Resumen ejecutivo

La cohorte contiene 250 pacientes con una estancia registrada de cero días, equivalentes al 2,09% del total.

## Diagnósticos

- Registros diagnósticos: 1.890.
- Diagnósticos únicos: 773.
- Diagnósticos principales: 416 (22,0%).
- Diagnósticos secundarios: 1.474 (78,0%).

### Cinco grupos diagnósticos más frecuentes

| Posición | Grupo | Frecuencia | Porcentaje |
|---:|---|---:|---:|
| 1 | Urgencias (`UUU`) | 133 | 7,0% |
| 2 | Uso prolongado de medicamentos (`Z79`) | 88 | 4,7% |
| 3 | Índice de masa corporal (`Z68`) | 53 | 2,8% |
| 4 | Neoplasias benignas y enfermedades de la sangre (`D12`) | 47 | 2,5% |
| 5 | Obesidad (`E66`) | 45 | 2,4% |

## Procedimientos

- Registros de procedimientos: 438.
- Procedimientos únicos: 205.

### Cinco grupos de procedimientos más frecuentes

| Posición | Grupo | Frecuencia | Porcentaje |
|---:|---|---:|---:|
| 1 | Cirugía médica y quirúrgica (`0DB`) | 155 | 35,4% |
| 2 | Cirugía médica y quirúrgica (`0DJ`) | 28 | 6,4% |
| 3 | Administración, incluidas transfusiones e infusiones (`3E0`) | 17 | 3,9% |
| 4 | Cirugía médica y quirúrgica (`0PS`) | 13 | 3,0% |
| 5 | Cirugía médica y quirúrgica (`0HQ`) | 13 | 3,0% |

## Interpretación

### Componente administrativo

Los 416 diagnósticos principales y la presencia de códigos Z son compatibles con casos de baja complejidad, evaluaciones breves e ingresos resueltos durante el mismo día, sin necesidad de hospitalización nocturna.

### Procedimientos diagnósticos

La combinación de procedimientos quirúrgicos, diagnóstico por imagen y administración puede corresponder a atenciones ambulatorias, estudios diagnósticos o intervenciones de corta duración.

### Posibles causas de alta durante el mismo día

1. Cirugías ambulatorias.
2. Procedimientos diagnósticos, como endoscopias y biopsias.
3. Evaluaciones de urgencia, triaje u observación breve.
4. Ingresos administrativos asociados con procedimientos.
5. Traslados internos o egresos administrativos.

### Relevancia clínica

- Este grupo representa una población predominantemente de baja complejidad y corta permanencia.
- Sus características deben considerarse de forma específica en el modelamiento predictivo.
- Los registros no deben interpretarse automáticamente como altas contra indicación médica o abandonos.
- Las estancias de cero días pueden corresponder a atenciones válidas dentro de la operación hospitalaria.

## Archivos asociados

- `01_diagnosticos_procedimientos_los_0.png`.
- `02_principal_vs_secundario_los_0.png`.
- `diagnosticos_detallado_los_0.csv`.
- `procedimientos_detallado_los_0.csv`.
- `RESUMEN_ANALISIS_LOS_0.txt`.
