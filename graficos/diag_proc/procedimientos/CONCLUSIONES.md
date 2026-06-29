# Análisis de Códigos Procedimiento y LOS
**Fecha de análisis:** 2026-03-30

---

## Resumen Ejecutivo

### Estadísticas Generales
- **Total de pacientes analizados:** 11,951
- **Total de registros código-paciente:** 26,155
- **LOS promedio:** 11.06 días
- **LOS mediana:** 3.0 días
- **Umbral outlier (Q3 + 1.5*IQR):** 23.5 días
- **Registros outlier:** 3,465 (13.2%)

---

## Códigos Asociados a Outliers (Estancias Largas)

### Top 5 Códigos con Mayor Probabilidad de Generar LOS Outliers

| Código | Prob. Outlier | N° Pacientes | LOS Promedio | LOS Máximo |
|--------|---------------|--------------|--------------|------------|
| **F07L3ZZ** | 81.8% | 11 | 58.5 días | 147 días |
| **F07Z9FZ** | 78.6% | 14 | 36.1 días | 131 días |
| **F07Z9ZZ** | 71.6% | 81 | 50.0 días | 262 días |
| **05HN33Z** | 70.6% | 34 | 45.6 días | 183 días |
| **5A1955Z** | 66.7% | 102 | 45.0 días | 262 días |

### Interpretación de Outliers
Los códigos listados arriba tienen una **alta probabilidad de estar asociados con estancias hospitalarias prolongadas** (≥23.5 días). Esto puede indicar:

1. **Complejidad clínica:** Condiciones que requieren tratamiento extenso o monitoreo prolongado
2. **Complicaciones:** Diagnósticos/procedimientos con alta tasa de complicaciones postoperatorias
3. **Comorbilidades:** Pacientes con múltiples condiciones que alargan la recuperación

**Conclusión clave:** El código **F07L3ZZ** tiene la mayor probabilidad de outlier (81.8%), apareciendo en 11 pacientes con LOS promedio de 58.5 días.

---

## Códigos Más Frecuentes (Densidad)

### Top 5 Códigos con Mayor Frecuencia

| Código | N° Pacientes | % del Total |
|--------|--------------|-------------|
| **4A1ZXQZ** | 1,027 | 8.59% |
| **10E0XZZ** | 722 | 6.04% |
| **30233N1** | 641 | 5.36% |
| **4A023N7** | 627 | 5.25% |
| **B2111ZZ** | 549 | 4.59% |

### Interpretación de Frecuencia
Los códigos más frecuentes representan las condiciones o procedimientos más comunes en la población hospitalaria analizada.

**Código más frecuente:** **4A1ZXQZ** aparece en 1,027 pacientes (8.59% del total).

---

## Relación Frecuencia vs. Outliers

### Códigos que son FRECUENTES y GENERAN OUTLIERS

No hay códigos que sean simultáneamente top 5 en frecuencia y top 5 en probabilidad de outlier.

**Esto indica que los códigos más comunes NO son los que generan estancias largas**, lo cual es positivo para la gestión hospitalaria (los casos frecuentes son predecibles y de corta duración).

---

## Recomendaciones

### Para Predicción de LOS
1. **Incluir códigos outlier como features importantes:** Los códigos con alta probabilidad de outlier deben tener mayor peso en modelos predictivos
2. **Estratificación por complejidad:** Considerar crear modelos separados para casos con/sin códigos outlier
3. **Features de interacción:** Analizar combinaciones de códigos que amplifiquen el LOS

### Para Gestión Hospitalaria
1. **Priorizar recursos:** Pacientes con códigos outlier requieren planificación anticipada de camas/recursos
2. **Protocolos diferenciados:** Desarrollar protocolos específicos para códigos de alta complejidad
3. **Monitoreo proactivo:** Alertas tempranas cuando se identifican códigos asociados a LOS largo

---

## Archivos Generados

### Gráficos
1. `01_codigos_outliers.png` - Top 20 códigos con mayor probabilidad de outlier
2. `02_boxplot_outliers.png` - Distribución de LOS para códigos outliers
3. `03_codigos_frecuentes.png` - Top 20 códigos más frecuentes
4. `04_violin_frecuentes.png` - Distribución de LOS para códigos frecuentes

### Datos
- `estadisticas_outliers.csv` - Estadísticas completas de códigos outliers
- `estadisticas_frecuencia.csv` - Estadísticas de frecuencia de códigos

---

**Script de análisis:** `analisis_codigos_outliers.py`
