# Análisis de Códigos Diagnóstico y LOS
**Fecha de análisis:** 2026-03-30

---

## 📊 Resumen Ejecutivo

### Estadísticas Generales
- **Total de pacientes analizados:** 11,951
- **Total de registros código-paciente:** 92,141
- **LOS promedio:** 12.93 días
- **LOS mediana:** 5.0 días
- **Umbral outlier (Q3 + 1.5*IQR):** 34.5 días
- **Registros outlier:** 8,491 (9.2%)

---

## 🔴 Códigos Asociados a Outliers (Estancias Largas)

### Top 5 Códigos con Mayor Probabilidad de Generar LOS Outliers

| Código | Prob. Outlier | N° Pacientes | LOS Promedio | LOS Máximo |
|--------|---------------|--------------|--------------|------------|
| **P612** | 80.0% | 15 | 66.7 días | 114 días |
| **P744** | 80.0% | 15 | 63.6 días | 114 días |
| **P284** | 76.9% | 13 | 53.8 días | 103 días |
| **T391X5A** | 76.9% | 13 | 54.8 días | 130 días |
| **P293** | 66.7% | 12 | 61.9 días | 114 días |

### Interpretación de Outliers
Los códigos listados arriba tienen una **alta probabilidad de estar asociados con estancias hospitalarias prolongadas** (≥34.5 días). Esto puede indicar:

1. **Complejidad clínica:** Condiciones que requieren tratamiento extenso o monitoreo prolongado
2. **Complicaciones:** Diagnósticos/procedimientos con alta tasa de complicaciones postoperatorias
3. **Comorbilidades:** Pacientes con múltiples condiciones que alargan la recuperación

**Conclusión clave:** El código **P612** tiene la mayor probabilidad de outlier (80.0%), apareciendo en 15 pacientes con LOS promedio de 66.7 días.

---

## 📈 Códigos Más Frecuentes (Densidad)

### Top 5 Códigos con Mayor Frecuencia

| Código | N° Pacientes | % del Total |
|--------|--------------|-------------|
| **UUUUUU** | 3,486 | 29.17% |
| **I10** | 2,484 | 20.78% |
| **Z7982** | 1,814 | 15.18% |
| **F17210** | 1,336 | 11.18% |
| **E669** | 1,301 | 10.89% |

### Interpretación de Frecuencia
Los códigos más frecuentes representan las condiciones o procedimientos más comunes en la población hospitalaria analizada.

**Código más frecuente:** **UUUUUU** aparece en 3,486 pacientes (29.17% del total).

---

## 🔍 Relación Frecuencia vs. Outliers

### Códigos que son FRECUENTES y GENERAN OUTLIERS

No hay códigos que sean simultáneamente top 5 en frecuencia y top 5 en probabilidad de outlier.

**Esto indica que los códigos más comunes NO son los que generan estancias largas**, lo cual es positivo para la gestión hospitalaria (los casos frecuentes son predecibles y de corta duración).

---

## 💡 Recomendaciones

### Para Predicción de LOS
1. **Incluir códigos outlier como features importantes:** Los códigos con alta probabilidad de outlier deben tener mayor peso en modelos predictivos
2. **Estratificación por complejidad:** Considerar crear modelos separados para casos con/sin códigos outlier
3. **Features de interacción:** Analizar combinaciones de códigos que amplifiquen el LOS

### Para Gestión Hospitalaria
1. **Priorizar recursos:** Pacientes con códigos outlier requieren planificación anticipada de camas/recursos
2. **Protocolos diferenciados:** Desarrollar protocolos específicos para códigos de alta complejidad
3. **Monitoreo proactivo:** Alertas tempranas cuando se identifican códigos asociados a LOS largo

---

## 📁 Archivos Generados

### Gráficos
1. `01_codigos_outliers.png` - Top 20 códigos con mayor probabilidad de outlier
2. `02_boxplot_outliers.png` - Distribución de LOS para códigos outliers
3. `03_codigos_frecuentes.png` - Top 20 códigos más frecuentes
4. `04_violin_frecuentes.png` - Distribución de LOS para códigos frecuentes

### Datos
- `estadisticas_outliers.csv` - Estadísticas completas de códigos outliers
- `estadisticas_frecuencia.csv` - Estadísticas de frecuencia de códigos

---

**Generado automáticamente por:** `analisis_codigos_outliers.py`
