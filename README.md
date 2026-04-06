# Predicción de LOS (Length of Stay) Hospitalario
### Proyecto Capstone - Grupo 16

---

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de **limpieza y preparación de datos** para predecir la duración de estancia hospitalaria (LOS) de pacientes. El objetivo es integrar y limpiar datos de diagnósticos y procedimientos médicos para generar un dataset maestro listo para modelado predictivo.

**Fecha de actualización:** 2026-03-30

---

## 🐍 Scripts Python - Guía Rápida

| Script | Propósito | Entrada | Salida | Orden |
|--------|-----------|---------|--------|-------|
| **limpieza_datos.py** | Limpia, valida ICD-10, integra diagnósticos y procedimientos | CSV crudos (datos_diagnostico, procedimiento_pacientes) | dataset_maestro, caso_diagnostico, caso_procedimiento, pacientes_rechazados | 1️⃣ |
| **analisis.py** | Genera 3 reportes estadísticos completos | dataset_maestro, caso_diagnostico, pacientes_rechazados | reporte_estadistico_*.csv | 2️⃣ |
| **analisis_codigos_outliers.py** | Identifica qué códigos ICD-10 generan estancias largas + visualizaciones | dataset_maestro | Gráficos + CONCLUSIONES.md + estadísticas | 3️⃣ |
| **analisis_distribucion_los.py** | Ajusta distribuciones (Log-Normal, Weibull, Gamma, Mezcla) y genera gráficos QQ | dataset_maestro | PNG: comparación de modelos y criterios AIC/KS | 4️⃣ |
| **visualizacion_los.py** | Histogramas lineales/logarítmicos y boxplots de LOS | dataset_maestro | PNG: distribuciones | 5️⃣ |
| **visualizar_weight_verosimilitud.py** | Gráfico: cómo cambia verosimilitud según peso en mezcla | dataset_maestro | PNG: verosimilitud vs weight | 6️⃣ |
| **icd-10-cm.py** | Diccionarios de referencia de capítulos y categorías ICD-10-CM | - | Python module (para consulta manual) | 🔍 |
| **icd-10-pcs.py** | Función utilitaria para decodificar códigos ICD-10-PCS | - | Python module (para consulta manual) | 🔍 |

---

## 📂 Estructura del Proyecto

```
Capstone-Grupo-16-/
│
├── SCRIPTS DE DATOS (Limpieza y Análisis)
│   ├── limpieza_datos.py                    # ⭐ PRINCIPAL: Limpieza, validación ICD-10, integración
│   ├── analisis.py                          # 📊 Reportes estadísticos (3 archivos: maestro, diag, rechazados)
│   ├── analisis_codigos_outliers.py         # 🆕 Análisis visual de códigos ICD-10 vs outliers
│   ├── analisis_distribucion_los.py         # 🆕 Ajuste: Log-Normal, Weibull, Gamma, Mezcla
│   └── icd-10-cm.py, icd-10-pcs.py         # 📋 Diccionarios de referencia ICD-10
│
├── SCRIPTS DE VISUALIZACIÓN
│   ├── visualizacion_los.py                 # 📈 Histogramas y box plots de LOS
│   └── visualizar_weight_verosimilitud.py   # 📊 Verosimilitud vs pesos en mezcla
│
├── DATOS DE ENTRADA (Crudos)
│   ├── datos_diagnostico.csv                # Diagnósticos crudos (ICD-10-CM)
│   └── procedimiento_pacientes.csv          # Procedimientos crudos + fechas (ICD-10-PCS)
│
├── data/processed/ (Datos Limpios)
│   ├── dataset_maestro.csv                  # ⭐ Dataset principal limpio (11,932 pacientes)
│   ├── caso_diagnostico.csv                 # 📋 Granular: 1 fila por diagnóstico (97,089 registros)
│   ├── caso_procedimiento.csv               # 📋 Granular: 1 fila por procedimiento (26,568 registros)
│   └── pacientes_rechazados.csv             # ❌ Pacientes no válidos (19 pacientes)
│
├── data/reports/ (Reportes Estadísticos)
│   ├── reporte_limpieza.csv                 # Resumen de limpieza (tasa aceptación, promedio LOS)
│   ├── reporte_estadistico_maestro.csv      # 📊 47 métricas del dataset maestro
│   ├── reporte_estadistico_diagnostico.csv  # 📊 70 métricas de diagnósticos
│   └── reporte_estadistico_rechazados.csv   # 📊 12 métricas de rechazo
│
├── graficos/
│   ├── diagnosticos/
│   │   ├── 01_codigos_outliers.png          # Top 20 códigos diagnósticos con prob. outlier
│   │   ├── 02_boxplot_outliers.png          # Distribución LOS por código diagnóstico
│   │   ├── 03_codigos_frecuentes.png        # Top 20 códigos más frecuentes
│   │   ├── 04_violin_frecuentes.png         # Violin plot de códigos frecuentes
│   │   ├── CONCLUSIONES.md                  # 📝 Análisis + recomendaciones diagnósticos
│   │   ├── estadisticas_outliers.csv        # Datos completos outliers
│   │   └── estadisticas_frecuencia.csv      # Datos completos frecuencia
│   │
│   └── procedimientos/
│       ├── 01_codigos_outliers.png          # Top 20 códigos procedimiento con prob. outlier
│       ├── 02_boxplot_outliers.png          # Distribución LOS por código procedimiento
│       ├── 03_codigos_frecuentes.png        # Top 20 códigos más frecuentes
│       ├── 04_violin_frecuentes.png         # Violin plot de códigos frecuentes
│       ├── CONCLUSIONES.md                  # 📝 Análisis + recomendaciones procedimientos
│       ├── estadisticas_outliers.csv        # Datos completos outliers
│       └── estadisticas_frecuencia.csv      # Datos completos frecuencia
│
├── README.md                                # 📖 Este archivo (documentación completa)
├── .DS_Store                                # macOS
└── Datos proyecto LOS.xlsx                  # Datos originales (Excel)
```

### 📊 Flujo de Datos

```
ENTRADA                    PROCESAMIENTO               SALIDA
│                          │                           │
├─ datos_diagnostico.csv ──┤                           ├─ dataset_maestro.csv        ⭐
├─ procedimiento_           limpieza_datos.py ────────►├─ caso_diagnostico.csv
  pacientes.csv             (validación ICD-10)       ├─ caso_procedimiento.csv
                                                       ├─ pacientes_rechazados.csv
                                                       └─ reporte_limpieza.csv

dataset_maestro.csv ──┬─────────────────────────────────►├─ reporte_estadistico_maestro.csv
caso_diagnostico.csv ─┼─ analisis.py (reporting) ─────►├─ reporte_estadistico_diagnostico.csv
caso_procedimiento.csv ─────────────────────────────────►└─ reporte_estadistico_rechazados.csv


dataset_maestro.csv ──┬─ analisis_codigos_outliers.py ─┬─ graficos/diagnosticos/
                      │   (análisis visual)             ├─ graficos/procedimientos/
                      │                                 └─ CONCLUSIONES.md (x2)


dataset_maestro.csv ──┬─ analisis_distribucion_los.py ─►PNG: distribuciones
                      ├─ visualizacion_los.py          └─ comparación modelos
                      └─ visualizar_weight_verosimilitud.py
```

---

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy

### Instalación
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Ejecución (Orden recomendado)

**Paso 1: Limpieza de datos (OBLIGATORIO - primer paso)**
```bash
python3 limpieza_datos.py
```
**Salida:**
- `data/processed/dataset_maestro.csv` (dataset limpio)
- `data/processed/caso_diagnostico.csv` (granular diagnósticos)
- `data/processed/caso_procedimiento.csv` (granular procedimientos)
- `data/processed/pacientes_rechazados.csv` (rechazados)
- `data/reports/reporte_limpieza.csv` (resumen)

**Paso 2: Análisis estadístico (OPCIONAL)**
```bash
python3 analisis.py
```
**Salida:**
- `data/reports/reporte_estadistico_maestro.csv`
- `data/reports/reporte_estadistico_diagnostico.csv`
- `data/reports/reporte_estadistico_rechazados.csv`

**Paso 3: Análisis visual de códigos y outliers (OPCIONAL)**
```bash
python3 analisis_codigos_outliers.py
```
**Salida:** Gráficos + conclusiones en `graficos/diagnosticos/` y `graficos/procedimientos/`

**Paso 4: Análisis de distribución (OPCIONAL)**
```bash
python3 analisis_distribucion_los.py
```
**Salida:** Comparación de modelos probabilísticos

**Paso 5: Visualizaciones adicionales (OPCIONAL)**
```bash
python3 visualizacion_los.py
python3 visualizar_weight_verosimilitud.py
```

---

## 📊 Archivos de Entrada

### 1. `datos_diagnostico.csv`

**Descripción:** Contiene los diagnósticos médicos de los pacientes codificados en ICD-10.

**Formato:** CSV con separador `;`

**Columnas:**

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `CASE` | String | Identificador único del paciente | `13872110` |
| `PrincSec` | String | Tipo de diagnóstico:<br>• `P` = Principal<br>• `S` = Secundario | `P` |
| `Diagnosis` | String | Código ICD-10 del diagnóstico | `E6601` |

**Ejemplo de datos:**
```csv
CASE;PrincSec;Diagnosis
13872110;P;E6601
13872110;S;Z6841
14035188;P;J984
```

**Notas:**
- Un paciente puede tener **múltiples diagnósticos** (primarios y secundarios)
- Códigos **ICD-10-CM** (Clinical Modification) válidos **formato sin decimal**:
  - Formato general: `[A-TV-Z][0-9][0-9A-Z]` opcionalmente seguido de hasta 4 caracteres alfanuméricos
  - Ejemplo: `E6601` (Obesidad mórbida), `S72302E` (Fractura femoral), `Z6841` (IMC), `I10` (Hipertensión)
  - **Nota importante:** Los datos usan nomenclatura ICD-10-CM pero **sin punto decimal** (ej: `E6601` en lugar de `E66.01`)
  - **Excepción COVID:** códigos `U070` y `U071` permitidos
  - **Excepción urgencia:** código `UUUUUU` indica ingreso por urgencia (NO electivo) - se **CONSERVA**
- Códigos rechazados: `AAAAAA`, `DDDDDD` y otros placeholders inválidos (23 registros filtrados)
- **Justificación nomenclatura ICD-10-CM:** Es el estándar internacional para codificación de diagnósticos clínicos, adoptado por la OMS y requerido por sistemas de salud en EE.UU., Canadá y múltiples países. Su estructura jerárquica permite agregación por categorías (ej: E66* = todos los tipos de obesidad). El formato sin decimal es una codificación compacta válida usada en algunos sistemas hospitalarios.

---

### 2. `procedimiento_pacientes.csv`

**Descripción:** Contiene los procedimientos médicos realizados y las fechas de ingreso/egreso.

**Formato:** CSV con separador `;`

**Columnas:**

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `Case` | String | Identificador único del paciente | `13872110` |
| `Procedure` | String | Código del procedimiento médico (ICD-10-PCS) | `0DB64Z3` |
| `Date` | String | Fecha de ingreso (formato `DD-MM-YY`) | `18-01-18` |
| `Release` | String | Fecha de egreso (formato `DD-MM-YY`) | `21-01-18` |

**Ejemplo de datos:**
```csv
Case;Procedure;Date;Release
13872110;0DB64Z3;18-01-18;21-01-18
14035188;0BB64ZZ;11-01-18;14-01-18
```

**Notas:**
- Un paciente puede tener **múltiples procedimientos** en diferentes fechas
- Códigos **ICD-10-PCS** (Procedure Coding System) válidos:
  - Formato: **exactamente 7 caracteres** alfanuméricos
  - Caracteres permitidos: `0-9, A-H, J-N, P-Z` (excluye I y O para evitar confusión con 1 y 0)
  - Ejemplo: `0DB64Z3` (Excisión de duodeno), `0BB64ZZ` (Excisión de lóbulo pulmonar)
  - Cada posición tiene significado: Sección, Sistema corporal, Operación raíz, Parte del cuerpo, Abordaje, Dispositivo, Calificador
- **Justificación nomenclatura ICD-10-PCS:** Sistema estandarizado de codificación de procedimientos usado en hospitales de EE.UU. Su estructura de 7 caracteres permite codificación precisa y única de cada procedimiento quirúrgico, garantizando interoperabilidad entre sistemas de salud y consistencia en facturación.
- El script calcula el **LOS** automáticamente como: `Release - Date` (en días)
- Si un paciente tiene múltiples procedimientos, se toma:
  - **Fecha de ingreso** = fecha más temprana (`min`)
  - **Fecha de egreso** = fecha más tardía (`max`)

---

## 📦 Archivos de Salida

### 1. `dataset_maestro.csv` (Principal)

**Descripción:** Dataset limpio con todos los pacientes válidos, listo para análisis y modelado.

**Formato:** CSV con separador `;`

**Columnas:**

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `case_id` | String | Identificador único del paciente | `13872110` |
| `fecha_ingreso` | Date | Fecha de ingreso (primer procedimiento) | `2018-01-18` |
| `fecha_egreso` | Date | Fecha de egreso (último procedimiento) | `2018-01-21` |
| `los_dias` | Integer | **LOS (Length of Stay)** en días | `3` |
| `es_urgencia` | Integer | Bandera de urgencia (1 = urgencia, 0 = electivo) | `1` |
| `procedimientos` | String | Lista de códigos de procedimientos (separados por `,`) | `0DB64Z3,0BB64ZZ` |
| `n_procedimientos` | Integer | Número total de procedimientos | `2` |
| `diagnosticos_primarios` | String | Lista de diagnósticos primarios (separados por `,`) | `E6601` |
| `diagnosticos_secundarios` | String | Lista de diagnósticos secundarios (separados por `,`) | `Z6841,I119` |
| `n_diag_primarios` | Integer | Número de diagnósticos primarios | `1` |
| `n_diag_secundarios` | Integer | Número de diagnósticos secundarios | `2` |
| `n_diag_total` | Integer | Total de diagnósticos (primarios + secundarios) | `3` |
| `tiene_diag_primario` | Boolean | Flag: ¿Tiene al menos un diagnóstico primario? | `True` |
| `los_negativo` | Boolean | Flag: ¿LOS es negativo? (siempre False en maestro) | `False` |
| `los_cero` | Boolean | Flag: ¿LOS es cero? | `False` |
| `fechas_invalidas` | Boolean | Flag: ¿Hay fechas inválidas? (siempre False en maestro) | `False` |

**Notas importantes:**
- Las columnas de listas (`procedimientos`, `diagnosticos_*`) están en formato string separadas por comas
- Para usar en Python: `df['procedimientos'].str.split(',')` para convertir a lista
- **LOS = 0 SÍ está incluido** (son estancias válidas de admisión/egreso el mismo día)
- **`los_dias`** está posicionado justo después de `fecha_egreso` para facilitar análisis temporal
- **`es_urgencia`** diferencia casos electivos (0) de urgencias (1) - útil para estratificación

---

### 2. `caso_diagnostico.csv` (Granular)

**Descripción:** Diagnósticos a nivel granular - **una fila por diagnóstico**.

**Propósito:** Responde a "¿Qué diagnósticos están presentes en los casos y cómo se relacionan con el LOS?"

**Formato:** CSV con separador `;`

**Columnas:**

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `case_id` | String | Identificador del paciente (llave foránea) | `13872110` |
| `d_code` | String | Código ICD-10-CM limpio del diagnóstico | `E6601` |
| `tipo_d` | String | Tipo de diagnóstico: `P` (primario) o `S` (secundario) | `P` |
| `d_caract_1` | String | Primera letra del código (capítulo general) | `E` |
| `d_caract_3` | String | Primeros 3 caracteres (categoría clínica) | `E66` |

**Ejemplo de datos:**
```csv
case_id;d_code;tipo_d;d_caract_1;d_caract_3
13872110;E6601;P;E;E66
13872110;Z6841;S;Z;Z68
14035188;J984;P;J;J98
14035188;I119;S;I;I11
```

**Estadísticas:**
- **Registros totales:** 97,089 diagnósticos
- **Pacientes únicos:** 11,932
- **Códigos únicos:** 6,108
- **Diagnósticos primarios:** 18,678 (19.2%)
- **Diagnósticos secundarios:** 78,411 (80.8%)
- **Promedio por paciente:** 8.14 diagnósticos

**Usos:**
- Análisis de comorbilidades (diagnósticos secundarios)
- Agregación por capítulo ICD-10 (letra inicial)
- Identificación de diagnósticos principales más frecuentes
- Feature engineering para modelos predictivos

---

### 3. `caso_procedimiento.csv` (Granular)

**Descripción:** Procedimientos a nivel granular - **una fila por procedimiento**.

**Propósito:** Responde a "¿Qué le hicieron al paciente y cómo eso influye en el LOS?"

**Formato:** CSV con separador `;`

**Columnas:**

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `case_id` | String | Identificador del paciente (llave foránea) | `13872110` |
| `p_code` | String | Código ICD-10-PCS del procedimiento | `0DB64Z3` |
| `p_caract_1` | String | Primera letra (tipo de procedimiento) | `0` |
| `p_caract_3` | String | Primeros 3 caracteres | `0DB` |

**Ejemplo de datos:**
```csv
case_id;p_code;p_caract_1;p_caract_3
13872110;0DB64Z3;0;0DB
14035188;0BB64ZZ;0;0BB
14114821;0H0V0JZ;0;0H0
14114821;0J0L0ZZ;0;0J0
```

**Estadísticas:**
- **Registros totales:** 26,568 procedimientos
- **Pacientes únicos:** 11,951
- **Promedio por paciente:** 2.22 procedimientos

**Notas sobre caracteres:**
- **`p_caract_1`** separa grandes categorías:
  - `0` = Médico-quirúrgico
  - `B` = Imagenología
  - `F` = Rehabilitación y medicina física
  - `3` = Administración
- **`p_caract_3`** permite agrupación por sistema corporal y operación

**Usos:**
- Clasificación de complejidad quirúrgica
- Análisis de tipos de procedimientos por caso
- Identificación de procedimientos correlacionados con LOS largo
- Feature engineering para modelos predictivos

---

### 4. `pacientes_rechazados.csv`

**Descripción:** Pacientes que NO pasaron los criterios de validación.

**Formato:** CSV con separador `;`

**Columnas principales:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `case_id` | String | Identificador del paciente rechazado |
| `motivo_rechazo` | String | Razón(es) de rechazo (separadas por `\|`) |
| `codigos_diagnostico_invalidos` | String | Códigos inválidos detectados (ej: AAAAAA) |
| `los_dias` | Integer/NaN | LOS calculado (puede ser nulo) |
| `n_procedimientos` | Integer | Número de procedimientos |
| `n_diag_total` | Integer | Número de diagnósticos |
| `fecha_ingreso` | Date/NaT | Fecha de ingreso |
| `fecha_egreso` | Date/NaT | Fecha de egreso |

**Motivos de rechazo posibles:**

| Motivo | Descripción |
|--------|-------------|
| `contiene_codigo_diagnostico_invalido` | **NUEVO:** Paciente contiene códigos inválidos como AAAAAA |
| `falta_en_diagnosticos` | Paciente existe en procedimientos pero NO en diagnósticos |
| `falta_en_procedimientos` | Paciente existe en diagnósticos pero NO en procedimientos |
| `fechas_invalidas` | Fechas de ingreso o egreso son inválidas (NaT) |
| `los_negativo` | Fecha de egreso es anterior a fecha de ingreso (error en datos) |

**Estadísticas actuales:**
- **Pacientes rechazados:** 19 (0.16%)
- **Todos rechazados por:** `contiene_codigo_diagnostico_invalido` (código AAAAAA)

**Motivos que NO causan rechazo:**
- ✅ **LOS = 0** (son datos válidos)

---

### 5. `reporte_limpieza.csv`

**Descripción:** Resumen ejecutivo con métricas de calidad de la limpieza.

**Formato:** CSV con separador `;`

**Métricas incluidas:**

| Métrica | Descripción |
|---------|-------------|
| `pacientes_totales` | Total de pacientes únicos en la unión de ambos archivos |
| `pacientes_maestro` | Pacientes válidos en el dataset maestro |
| `pacientes_rechazados` | Pacientes que no pasaron validación |
| `tasa_aceptacion_pct` | Porcentaje de pacientes aceptados |
| `pacientes_con_los_cero` | Pacientes con estancia de 0 días (conservados) |
| `pacientes_falta_diagnosticos` | Pacientes sin datos de diagnósticos |
| `pacientes_falta_procedimientos` | Pacientes sin datos de procedimientos |
| `pacientes_fechas_invalidas` | Pacientes con fechas mal formateadas |
| `pacientes_los_negativo` | Pacientes con LOS negativo (error en datos) |
| `los_promedio_dias` | Promedio de LOS en el dataset maestro |
| `procedimientos_promedio` | Promedio de procedimientos por paciente |
| `diagnosticos_promedio` | Promedio de diagnósticos por paciente |

**Ejemplo:**
```csv
metrica;valor
pacientes_totales;11951
pacientes_maestro;11932
pacientes_rechazados;19
pacientes_rechazados_por_codigos_invalidos;19
tasa_aceptacion_pct;99.84
los_promedio_dias;6.44
```

---

## 📈 Reportes Estadísticos (analisis.py)

El script `analisis.py` genera reportes estadísticos detallados estilo `summary()` de R para cada dataset.

### 1. `reporte_estadistico_maestro.csv`

**Descripción:** Análisis estadístico completo del dataset maestro.

**Formato:** CSV con separador `;` (columnas: `seccion`, `variable`, `valor`)

**Secciones incluidas:**

#### **GENERAL**
| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `numero_pacientes` | Total de pacientes en dataset maestro | 11,932 |
| `periodo_inicio` | Fecha más temprana de ingreso | 2017-07-12 |
| `periodo_fin` | Fecha más tardía de egreso | 2018-06-30 |

#### **LOS (Length of Stay)**
| Variable | Descripción |
|----------|-------------|
| `count` | Número de observaciones |
| `mean` | Promedio de días de estancia |
| `std` | Desviación estándar |
| `min`, `percentil_25`, `median`, `percentil_75`, `max` | Cuartiles |
| `percentil_90`, `percentil_95`, `percentil_99` | Percentiles superiores |

**Ejemplo:**
```csv
seccion;variable;valor
LOS;mean;6.44
LOS;median;3.0
LOS;percentil_90;15.0
LOS;percentil_95;27.0
```

#### **LOS_RANGOS**
Distribución de pacientes por rangos de estancia:
- `los_0`: Pacientes con LOS = 0 días
- `los_1_3`: Estancias cortas (1-3 días)
- `los_4_7`: Estancias medias (4-7 días)
- `los_8_14`: Estancias largas (8-14 días)
- `los_15_30`: Estancias muy largas (15-30 días)
- `los_mas_30`: Estancias prolongadas (>30 días)

#### **URGENCIA**
| Variable | Descripción |
|----------|-------------|
| `pacientes_urgencia` | Número de ingresos por urgencia (es_urgencia=1) |
| `pacientes_electivos` | Número de ingresos electivos (es_urgencia=0) |
| `porcentaje_urgencias` | % de ingresos por urgencia |
| `los_promedio_urgencia` | LOS promedio para urgencias |
| `los_promedio_electivo` | LOS promedio para electivos |

**Insight importante:**
```csv
URGENCIA;los_promedio_urgencia;11.44
URGENCIA;los_promedio_electivo;4.39
```
→ Las urgencias tienen **2.6x más LOS** que ingresos electivos.

#### **PROCEDIMIENTOS** y **DIAGNOSTICOS**
Estadísticas descriptivas de número de procedimientos/diagnósticos por paciente:
- `mean`, `std`, `min`, `median`, `max`
- Distribuciones por rangos

#### **CORRELACIONES**
| Variable | Descripción |
|----------|-------------|
| `los_vs_procedimientos` | Correlación de Pearson entre LOS y número de procedimientos |
| `los_vs_diagnosticos` | Correlación de Pearson entre LOS y número de diagnósticos |

---

### 2. `reporte_estadistico_diagnostico.csv`

**Descripción:** Análisis estadístico de diagnósticos granulares.

**Formato:** CSV con separador `;` (columnas: `seccion`, `variable`, `valor`)

**Secciones incluidas:**

#### **GENERAL**
| Variable | Valor |
|----------|-------|
| `numero_diagnosticos` | 97,089 |
| `pacientes_unicos` | 11,932 |
| `codigos_unicos` | 6,108 |
| `diagnosticos_por_paciente_promedio` | 8.14 |

#### **TIPO**
| Variable | Valor |
|----------|-------|
| `primarios` | 18,678 |
| `secundarios` | 78,411 |
| `porcentaje_primarios` | 19.24% |

#### **TOP_CODIGOS**
Top 20 códigos diagnósticos más frecuentes:
```csv
seccion;variable;valor
TOP_CODIGOS;top_01;UUUUUU (3472)
TOP_CODIGOS;top_02;I10 (2525)
TOP_CODIGOS;top_03;Z7982 (1846)
```

**Interpretación:**
- `UUUUUU`: Código de urgencia (conservado intencionalmente)
- `I10`: Hipertensión esencial
- `Z7982`: IMC alto

#### **TOP_CAPITULOS**
Top 20 capítulos ICD-10 más frecuentes (primera letra):
- `Z`: Factores que influyen en el estado de salud
- `I`: Enfermedades del sistema circulatorio
- `E`: Enfermedades endocrinas, nutricionales y metabólicas
- `J`: Enfermedades del sistema respiratorio

#### **TOP_CATEGORIAS**
Top 20 categorías ICD-10 más frecuentes (primeros 3 caracteres).

#### **DISTRIBUCION**
| Variable | Descripción |
|----------|-------------|
| `min_diag_por_paciente` | Mínimo número de diagnósticos |
| `max_diag_por_paciente` | Máximo número de diagnósticos |
| `median_diag_por_paciente` | Mediana de diagnósticos |

---

### 3. `reporte_estadistico_rechazados.csv`

**Descripción:** Análisis estadístico de pacientes rechazados.

**Formato:** CSV con separador `;` (columnas: `seccion`, `variable`, `valor`)

**Secciones incluidas:**

#### **GENERAL**
| Variable | Valor |
|----------|-------|
| `pacientes_rechazados` | 19 |

#### **MOTIVOS**
Distribución de motivos de rechazo:
```csv
seccion;variable;valor
MOTIVOS;motivo_01;contiene_codigo_diagnostico_invalido (19)
```

#### **CODIGOS_INVALIDOS**
| Variable | Descripción |
|----------|-------------|
| `total_codigos_invalidos` | Número total de códigos inválidos encontrados |
| `codigos_unicos` | Códigos inválidos únicos |
| `codigo_01`, `codigo_02`, ... | Top códigos inválidos con frecuencia |

**Ejemplo:**
```csv
CODIGOS_INVALIDOS;codigo_01;AAAAAA (19)
```

#### **LOS**, **PROCEDIMIENTOS**, **DIAGNOSTICOS**
Estadísticas descriptivas de pacientes rechazados (si aplica).

---

### Cómo Usar los Reportes Estadísticos

```python
import pandas as pd

# Cargar reporte maestro
df_reporte = pd.read_csv('data/reports/reporte_estadistico_maestro.csv', sep=';')

# Ver estadísticas de LOS
los_stats = df_reporte[df_reporte['seccion'] == 'LOS']
print(los_stats)

# Ver correlaciones
corr_stats = df_reporte[df_reporte['seccion'] == 'CORRELACIONES']
print(corr_stats)

# Ver top diagnósticos
df_diag = pd.read_csv('data/reports/reporte_estadistico_diagnostico.csv', sep=';')
top_codigos = df_diag[df_diag['seccion'] == 'TOP_CODIGOS']
print(top_codigos)
```

---

## 🔧 Explicación del Script `analisis.py`

El script `analisis.py` está estructurado en **4 funciones principales**:

### **FUNCIÓN 1: `analizar_dataset_maestro()`**

**Propósito:** Generar reporte estadístico completo del dataset maestro.

**Proceso:**
1. Carga `dataset_maestro.csv`
2. Convierte columnas numéricas (`los_dias`, `es_urgencia`, `n_procedimientos`, etc.)
3. Calcula estadísticas descriptivas (mean, std, percentiles)
4. Genera distribuciones por rangos (LOS, procedimientos, diagnósticos)
5. Calcula estadísticas por tipo de ingreso (urgencia vs electivo)
6. Calcula correlaciones entre variables
7. Retorna DataFrame con todas las estadísticas

**Estadísticas clave:**
- 47 métricas totales
- 6 secciones: GENERAL, LOS, LOS_RANGOS, URGENCIA, PROCEDIMIENTOS, DIAGNOSTICOS, CORRELACIONES

### **FUNCIÓN 2: `analizar_caso_diagnostico()`**

**Propósito:** Generar reporte estadístico de diagnósticos granulares.

**Proceso:**
1. Carga `caso_diagnostico.csv`
2. Calcula estadísticas generales (total, pacientes únicos, códigos únicos)
3. Analiza distribución de tipos (primarios vs secundarios)
4. Genera Top 20 de códigos más frecuentes
5. Genera Top 20 de capítulos (primera letra)
6. Genera Top 20 de categorías (primeros 3 caracteres)
7. Calcula distribución de diagnósticos por paciente

**Estadísticas clave:**
- ~70 métricas totales
- Secciones: GENERAL, TIPO, TOP_CODIGOS, TOP_CAPITULOS, TOP_CATEGORIAS, DISTRIBUCION

### **FUNCIÓN 3: `analizar_pacientes_rechazados()`**

**Propósito:** Generar reporte estadístico de pacientes rechazados.

**Proceso:**
1. Carga `pacientes_rechazados.csv`
2. Cuenta pacientes rechazados
3. Analiza distribución de motivos de rechazo
4. Identifica códigos inválidos encontrados
5. Calcula estadísticas de LOS/procedimientos/diagnósticos (si aplican)

**Estadísticas clave:**
- ~12 métricas totales
- Secciones: GENERAL, MOTIVOS, CODIGOS_INVALIDOS, LOS, PROCEDIMIENTOS, DIAGNOSTICOS

### **FUNCIÓN 4: `guardar_reportes()`**

**Propósito:** Guardar los 3 reportes en archivos CSV.

**Proceso:**
1. Crea directorio `data/reports/` si no existe
2. Exporta cada reporte con separador `;`
3. Imprime resumen de estadísticas generadas

---

## 🔧 Explicación del Script `limpieza_datos.py`

El script está estructurado en **5 funciones principales** que se ejecutan secuencialmente:

---

### **FUNCIÓN 1: `cargar_datos_crudos()`**

**Propósito:** Leer los archivos CSV de entrada.

**Proceso:**
```python
# Cargar archivo de diagnósticos con separador ; y todo como strings
df_diagnosticos = pd.read_csv(ARCHIVO_DIAGNOSTICOS, sep=';', dtype=str)

# Cargar archivo de procedimientos con separador ; y todo como strings
df_procedimientos = pd.read_csv(ARCHIVO_PROCEDIMIENTOS, sep=';', dtype=str)
```

**¿Por qué `dtype=str`?**
- Evita que pandas convierta automáticamente tipos (ej: `01234` → `1234`)
- Preserva códigos ICD-10 con ceros a la izquierda
- Permite validación manual posterior

---

### **FUNCIÓN 2: `limpiar_diagnosticos(df)`**

**Propósito:** Validar códigos ICD-10 y agregar diagnósticos por paciente.

#### **Bloque 1: Estandarización de columnas**
```python
# Convertir todas las columnas a minúsculas para consistencia
df.columns = df.columns.str.lower()

# Renombrar 'case' → 'case_id' para claridad
df = df.rename(columns={'case': 'case_id'})
```

#### **Bloque 2: Validación de PrincSec**
```python
# Crear columna booleana: True si es 'P' o 'S', False en caso contrario
df['princsec_valido'] = df['princsec'].isin({'P', 'S'})
```

#### **Bloque 3: Validación de códigos ICD-10-CM (sin decimal)**
```python
# REGEX para ICD-10-CM (Clinical Modification) - Formato SIN DECIMAL
# Formato: [A-TV-Z][0-9][0-9A-Z] opcionalmente seguido de hasta 4 caracteres alfanuméricos
# Excepción 1: códigos U07 (COVID) permiten U070 o U071
# Excepción 2: código UUUUUU (urgencia) se CONSERVA
ICD10_CM_REGEX = r'^(?:[A-TV-Z][0-9][0-9A-Z](?:[0-9A-Z]{0,4})?|U07[01]?|UUUUUU)$'

# Validar cada código contra el patrón
df['diagnosis_valido'] = df['diagnosis'].str.match(ICD10_CM_REGEX, na=False)

# Ejemplos válidos: E6601, S72302E, I10, U071, UUUUUU (urgencia)
# Ejemplos inválidos: AAAAAA, DDDDDD, 123ABC, E6 (muy corto)
```

#### **Bloque 4: Filtrado de registros inválidos (NO pacientes completos)**
```python
# DECISIÓN CRÍTICA: Solo eliminar REGISTROS con códigos inválidos
# NO eliminar pacientes completos
df_valido = df[df['princsec_valido'] & df['diagnosis_valido']].copy()

# Ejemplo 1:
# Paciente 12345 tiene diagnósticos: E6601 (válido), AAAAAA (inválido), I119 (válido)
# Resultado: Se conserva paciente con E6601 e I119, se elimina solo AAAAAA

# Ejemplo 2:
# Paciente 67890 tiene: UUUUUU (urgencia, válido), E6601 (válido)
# Resultado: Se conservan ambos códigos (UUUUUU indica admisión por urgencia)
```

#### **Bloque 5: Agregación por paciente**
```python
# Separar diagnósticos primarios (P)
df_primarios = df_valido[df_valido['princsec'] == 'P']

# Agrupar por paciente y convertir a lista
diag_primarios = (
    df_primarios
    .groupby('case_id')['diagnosis']
    .apply(list)  # Convierte múltiples filas en una lista
    .rename('diagnosticos_primarios')
)

# Ejemplo:
# case_id | diagnosis
# 12345   | E6601
# 12345   | I119
# Resultado: case_id=12345, diagnosticos_primarios=['E6601', 'I119']
```

#### **Bloque 6: Creación de métricas**
```python
# Contar número de diagnósticos
df_paciente['n_diag_primarios'] = df_paciente['diagnosticos_primarios'].apply(len)

# Flag: tiene al menos un diagnóstico primario
df_paciente['tiene_diag_primario'] = df_paciente['n_diag_primarios'] > 0
```

---

### **FUNCIÓN 3: `limpiar_procedimientos(df)`**

**Propósito:** Validar códigos ICD-10-PCS, validar fechas, calcular LOS y agregar por paciente.

#### **Bloque 1: Validación de códigos ICD-10-PCS**
```python
# REGEX para ICD-10-PCS (Procedure Coding System)
# Formato: exactamente 7 caracteres alfanuméricos [0-9A-HJ-NP-Z]
# Excluye letras I, O para evitar confusión con dígitos 1, 0
ICD10_PCS_REGEX = r'^[0-9A-HJ-NP-Z]{7}$'

# Validar cada código contra el patrón
df['procedimiento_valido'] = (
    (df['procedimiento'].notna()) &
    (df['procedimiento'] != '') &
    (df['procedimiento'].str.match(ICD10_PCS_REGEX, na=False))
)

# Ejemplos válidos: 0DB64Z3, 0BB64ZZ, 02H60JZ
# Ejemplos inválidos: 0DB64Z (6 caracteres), 0DB64Z3X (8 caracteres), OIIOIIO (contiene I/O)

# Filtrar solo registros con códigos válidos
df = df[df['procedimiento_valido']].copy()
```

#### **Bloque 2: Conversión de fechas**
```python
# Convertir strings 'DD-MM-YY' a objetos datetime
# dayfirst=True indica que el día va primero
df['fecha_ingreso'] = pd.to_datetime(
    df['fecha_ingreso'],
    format='%d-%m-%y',
    errors='coerce'  # Convierte fechas inválidas a NaT (Not a Time)
)

# Ejemplo: '18-01-18' → datetime(2018, 1, 18)
```

#### **Bloque 3: Agregación de fechas por paciente**
```python
# Para cada paciente, obtener:
# - Fecha de ingreso más temprana (primer procedimiento)
# - Fecha de egreso más tardía (último procedimiento)

fechas_por_paciente = df.groupby('case_id').agg({
    'fecha_ingreso': 'min',  # Mínimo = más temprana
    'fecha_egreso': 'max'    # Máximo = más tardía
})

# Ejemplo:
# Paciente 12345 tiene procedimientos en:
# - 10-01-18 (ingreso) → 12-01-18 (egreso)
# - 15-01-18 (ingreso) → 20-01-18 (egreso)
# Resultado: fecha_ingreso=10-01-18, fecha_egreso=20-01-18
```

#### **Bloque 4: Cálculo de LOS**
```python
# LOS = diferencia en días entre egreso e ingreso
df_paciente['los_dias'] = (
    (df_paciente['fecha_egreso'] - df_paciente['fecha_ingreso']).dt.days
)

# Flags de validación
df_paciente['los_negativo'] = df_paciente['los_dias'] < 0  # Error en datos
df_paciente['los_cero'] = df_paciente['los_dias'] == 0     # Estancia de 1 día

# Ejemplo:
# fecha_ingreso=10-01-18, fecha_egreso=13-01-18 → los_dias=3
# fecha_ingreso=10-01-18, fecha_egreso=10-01-18 → los_dias=0 (VÁLIDO)
# fecha_ingreso=10-01-18, fecha_egreso=08-01-18 → los_dias=-2 (INVÁLIDO)
```

---

### **FUNCIÓN 4: `integrar_datos(df_diagnosticos, df_procedimientos)`**

**Propósito:** Unir diagnósticos y procedimientos, y aplicar criterios de rechazo.

#### **Bloque 1: Merge OUTER**
```python
# outer join: incluye TODOS los pacientes de ambos archivos
df = df_procedimientos.merge(
    df_diagnosticos,
    on='case_id',
    how='outer',
    indicator=True  # Crea columna '_merge' con origen de cada fila
)

# Valores de '_merge':
# - 'both': paciente en ambos archivos (ESPERADO)
# - 'left_only': solo en procedimientos (RECHAZADO)
# - 'right_only': solo en diagnósticos (RECHAZADO)
```

#### **Bloque 2: Criterios de rechazo**
```python
# CRITERIO 1: Falta en diagnósticos
mask_solo_proc = df['_merge'] == 'left_only'
# → Paciente tiene procedimientos pero NO diagnósticos → RECHAZADO

# CRITERIO 2: Falta en procedimientos
mask_solo_diag = df['_merge'] == 'right_only'
# → Paciente tiene diagnósticos pero NO procedimientos → RECHAZADO

# CRITERIO 3: Fechas inválidas
mask_fechas_invalidas = df['fechas_invalidas'].fillna(False)
# → Fechas son NaT (inválidas) → RECHAZADO

# CRITERIO 4: LOS negativo
mask_los_negativo = df['los_negativo'].fillna(False)
# → Fecha egreso < fecha ingreso → RECHAZADO

# CRITERIOS QUE NO RECHAZAN:
# ✅ LOS = 0 → ACEPTADO (admisión/egreso el mismo día)
# ✅ Códigos ICD-10 inválidos filtrados → ACEPTADO (con códigos válidos restantes)
```

#### **Bloque 3: Separación maestro vs rechazados**
```python
# Crear lista de razones de rechazo para cada paciente
df['razones_rechazo'] = [[] for _ in range(len(df))]

# Agregar razón si cumple criterio
df.loc[mask_solo_proc, 'razones_rechazo'] = ...
# → razones_rechazo = ['falta_en_diagnosticos']

# Marcar como rechazado si tiene al menos una razón
df['es_rechazado'] = df['razones_rechazo'].apply(len) > 0

# Separar
df_maestro = df[~df['es_rechazado']].copy()
df_rechazados = df[df['es_rechazado']].copy()
```

---

### **FUNCIÓN 5: `guardar_resultados(df_maestro, df_rechazados, df_completo)`**

**Propósito:** Exportar resultados a archivos CSV.

#### **Bloque 1: Conversión de listas a strings**
```python
# CSV no soporta listas nativamente, convertir a strings
# ['E6601', 'I119'] → 'E6601,I119'

for col in ['procedimientos', 'diagnosticos_primarios', 'diagnosticos_secundarios']:
    df[col] = df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else '')

# Para leer de nuevo en Python:
# df['procedimientos'].str.split(',')
```

#### **Bloque 2: Creación de reporte de calidad**
```python
# Crear DataFrame con métricas resumidas
reporte = {
    'metrica': ['pacientes_totales', 'pacientes_maestro', ...],
    'valor': [len(df_completo), len(df_maestro), ...]
}

df_reporte = pd.DataFrame(reporte)
df_reporte.to_csv('reporte_limpieza.csv', index=False, sep=';')
```

---

## 📊 Resultados de la Limpieza Actual

**Ejecución: 2026-03-30**

| Métrica | Valor |
|---------|-------|
| **Pacientes totales** | 11,951 |
| **Pacientes en dataset maestro** | 11,932 (99.84%) ✅ |
| **Pacientes rechazados** | 19 (0.16%) por código AAAAAA |
| **Registros diagnóstico con AAAAAA** | 23 registros en 19 pacientes |
| **Códigos UUUUUU conservados** | 3,472 (urgencias) ✅ |
| **Pacientes con urgencia (es_urgencia=1)** | 3,472 (29.1%) |
| **Pacientes electivos (es_urgencia=0)** | 8,460 (70.9%) |
| **Pacientes con LOS = 0** | 250 (conservados) |
| **LOS promedio** | 6.44 días |
| **LOS promedio urgencias** | 11.44 días |
| **LOS promedio electivos** | 4.39 días |
| **LOS máximo** | 262 días |
| **Procedimientos promedio** | 2.22 por paciente |
| **Diagnósticos promedio** | 8.14 por paciente |
| **Diagnósticos granulares generados** | 97,089 registros |
| **Procedimientos granulares generados** | 26,568 registros |
| **Códigos diagnósticos únicos** | 6,108 |

---

## 🎯 Decisiones de Diseño Importantes

### ✅ **Por qué usar nomenclaturas ICD-10-CM e ICD-10-PCS**

**Nomenclaturas adoptadas:**
- **ICD-10-CM** (Clinical Modification): Para diagnósticos médicos
- **ICD-10-PCS** (Procedure Coding System): Para procedimientos quirúrgicos

**Razones técnicas:**

1. **Estándares internacionales oficiales:**
   - ICD-10-CM es el estándar global adoptado por la OMS (Organización Mundial de la Salud)
   - ICD-10-PCS es el estándar estadounidense para facturación y codificación hospitalaria
   - Garantizan **interoperabilidad** entre sistemas de salud de diferentes instituciones

2. **Validación robusta y precisa:**
   - ICD-10-CM validado en formato sin decimal (ej: `E6601` en lugar de `E66.01`)
   - ICD-10-PCS valida exactamente 7 caracteres, evitando códigos incompletos o mal formados
   - Excluye caracteres ambiguos (I, O) que pueden confundirse con dígitos
   - **Conserva código UUUUUU** (urgencia) por su valor clínico predictivo

3. **Prevención de datos erróneos:**
   - Filtra códigos placeholder inválidos como `AAAAAA`, `DDDDDD`, etc.
   - **Conserva UUUUUU** (urgencia) como dato clínico válido
   - Detecta códigos truncados o con caracteres extra
   - Maneja espacios en blanco que pueden causar fallos en joins

4. **Trazabilidad y reproducibilidad:**
   - Nomenclaturas documentadas públicamente (CMS, CDC)
   - Permite auditoría y verificación de categorías diagnósticas
   - Facilita agregación jerárquica (ej: E66.* = todos los tipos de obesidad)

**Impacto en calidad de datos:**
- Reducción de falsos positivos en validación
- Mayor precisión en categorización de diagnósticos y procedimientos
- Compatibilidad con sistemas estándar de análisis de datos hospitalarios

---

### ✅ **Por qué NO se rechazaron pacientes por códigos ICD-10 inválidos**

**Problema identificado:** Algunos códigos como `AAAAAA` son placeholders inválidos que deben filtrarse.

**Solución adoptada:** Se **filtran solo los códigos inválidos**, pero se **conserva el paciente** con sus códigos válidos.

**Nota especial sobre UUUUUU:** Este código **se conserva** porque indica admisión por urgencia (no electiva), lo cual es información clínica valiosa para el modelo predictivo.

**Ejemplos:**
```
Paciente 12345:
- Diagnósticos originales: E6601 (válido), AAAAAA (placeholder), I119 (válido)
- Resultado: Conserva paciente con [E6601, I119] ✅

Paciente 67890:
- Diagnósticos originales: UUUUUU (urgencia), E6601 (válido)
- Resultado: Conserva paciente con [UUUUUU, E6601] ✅ (ambos códigos válidos)
```

---

### ✅ **Por qué LOS = 0 NO se rechaza**

**Interpretación:** LOS = 0 significa que el paciente ingresó y egresó **el mismo día**.

**Casos reales:**
- Cirugías ambulatorias
- Procedimientos diagnósticos rápidos
- Observación de emergencia con alta el mismo día

**Dato:** 250 pacientes (2.1%) tienen LOS = 0 → **Son datos válidos y útiles para el modelo**.

---

### ✅ **Por qué se simplificó el código**

**Sistema anterior:**
- 17 archivos Python
- Múltiples carpetas (`src/config`, `src/cleaning`, etc.)
- Uso de `pathlib`, `settings.py`, `__init__.py` vacíos
- 200+ líneas distribuidas en módulos

**Sistema actual:**
- **1 solo archivo** Python (`limpieza_datos.py`)
- **NO requiere** imports complejos ni configuración
- **Más fácil de entender** y modificar
- **Mismo resultado** (mejor incluso)

**Ventajas:**
- ✅ Más rápido de ejecutar
- ✅ Más fácil de debuggear
- ✅ No genera `__pycache__`
- ✅ Comentarios línea por línea para aprendizaje

---

## 🔍 Validación de Datos

### Verificar códigos ICD-10-CM filtrados (diagnósticos)
```python
import pandas as pd

df = pd.read_csv('datos_diagnostico.csv', sep=';', dtype=str)
df['Diagnosis'] = df['Diagnosis'].str.strip()

# Ver códigos inválidos únicos
import re
regex = r'^(?:[A-TV-Z][0-9][0-9A-Z](?:[0-9A-Z]{0,4})?|U07[01]?|UUUUUU)$'
invalidos = df[~df['Diagnosis'].str.match(regex, na=False)]
print(invalidos['Diagnosis'].value_counts())

# Output esperado:
# AAAAAA      23

# Verificar que UUUUUU se conserva (urgencia)
uuuuuu_count = df[df['Diagnosis'] == 'UUUUUU'].shape[0]
print(f'\nCódigos UUUUUU (urgencia) conservados: {uuuuuu_count:,}')
# Output esperado: 3,486
```

### Verificar códigos ICD-10-PCS filtrados (procedimientos)
```python
df = pd.read_csv('procedimiento_pacientes.csv', sep=';')

# Ver códigos inválidos únicos
regex = r'^[0-9A-HJ-NP-Z]{7}$'
invalidos = df[~df['Procedure'].str.match(regex, na=False)]
print(f"Códigos inválidos: {len(invalidos)}")
print(invalidos['Procedure'].value_counts().head(10))
```

### Verificar pacientes con LOS = 0
```python
df_maestro = pd.read_csv('data/processed/dataset_maestro.csv', sep=';')

los_cero = df_maestro[df_maestro['los_dias'] == 0]
print(f"Pacientes con LOS=0: {len(los_cero)}")
print(los_cero[['case_id', 'fecha_ingreso', 'fecha_egreso', 'n_procedimientos']])
```

---

## 📝 Siguiente Paso: Modelado

Con el dataset maestro limpio, el siguiente paso es:

1. **Análisis exploratorio (EDA)**
   - Distribución de LOS
   - Correlación entre diagnósticos y procedimientos
   - Identificación de outliers

2. **Feature Engineering**
   - One-hot encoding de códigos ICD-10
   - Categorización de procedimientos
   - Variables temporales (mes, día de la semana)

3. **Modelado Predictivo**
   - Regresión (predecir días exactos)
   - Clasificación (categorías: corta/media/larga estancia)
   - Modelos candidatos: RandomForest, XGBoost, Redes Neuronales

---

## 👥 Equipo

**Capstone - Grupo 16**
Universidad: [Nombre Universidad]
Fecha: Marzo 2026

---

## 📞 Contacto

Para preguntas o problemas con el código, contactar al equipo de desarrollo.

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico del curso Capstone.

---

**Última actualización:** 2026-03-30
**Versión del script:** 3.0 (Análisis estadístico + archivos granulares)
**Mejoras clave:**
- ✅ Validación ICD-10-CM formato sin decimal (E6601 en lugar de E66.01)
- ✅ Validación ICD-10-PCS exacta (7 caracteres)
- ✅ **NUEVO:** Bandera `es_urgencia` para diferenciar urgencias de electivos
- ✅ **NUEVO:** Archivos granulares `caso_diagnostico.csv` y `caso_procedimiento.csv`
- ✅ **NUEVO:** Script `analisis.py` con reportes estadísticos detallados
- ✅ Rechazo completo de pacientes con código AAAAAA (19 pacientes)
- ✅ Conserva código UUUUUU (urgencia) - 3,472 pacientes
- ✅ 99.84% tasa de aceptación de pacientes
- ✅ LOS=0 conservados como datos válidos
