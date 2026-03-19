# Predicción de LOS (Length of Stay) Hospitalario
### Proyecto Capstone - Grupo 16

---

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de **limpieza y preparación de datos** para predecir la duración de estancia hospitalaria (LOS) de pacientes. El objetivo es integrar y limpiar datos de diagnósticos y procedimientos médicos para generar un dataset maestro listo para modelado predictivo.

**Fecha de actualización:** 2026-03-18

---

## 📂 Estructura del Proyecto

```
Capstone-Grupo-16-/
│
├── datos_diagnostico.csv              # Datos crudos de diagnósticos (ICD-10)
├── procedimiento_pacientes.csv        # Datos crudos de procedimientos y fechas
│
├── limpieza_datos.py                  # ⭐ SCRIPT PRINCIPAL DE LIMPIEZA
│
├── data/
│   ├── processed/
│   │   ├── dataset_maestro.csv        # Dataset final limpio (salida principal)
│   │   └── pacientes_rechazados.csv   # Pacientes rechazados con justificación
│   │
│   └── reports/
│       └── reporte_limpieza.csv       # Métricas de calidad de la limpieza
│
└── README.md                          # Este archivo
```

---

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos
- Python 3.7+
- pandas

### Instalación
```bash
pip install pandas
```

### Ejecución
```bash
python3 limpieza_datos.py
```

### Salida Esperada
El script genera **3 archivos CSV** en las carpetas correspondientes:

1. **`data/processed/dataset_maestro.csv`** - Dataset limpio listo para modelado
2. **`data/processed/pacientes_rechazados.csv`** - Pacientes no válidos con razones
3. **`data/reports/reporte_limpieza.csv`** - Resumen ejecutivo de la limpieza

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
- Códigos ICD-10 válidos: letra (A-Z excepto U) + 2-6 caracteres alfanuméricos
- Códigos como `UUUUUU` o `AAAAAA` son placeholders de diagnósticos desconocidos (se filtran)

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
- El script calc ula el **LOS** automáticamente como: `Release - Date` (en días)
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
| `procedimientos` | String | Lista de códigos de procedimientos (separados por `,`) | `0DB64Z3,0BB64ZZ` |
| `n_procedimientos` | Integer | Número total de procedimientos | `2` |
| `los_dias` | Integer | **LOS (Length of Stay)** en días | `3` |
| `los_negativo` | Boolean | Flag: ¿LOS es negativo? (siempre False en maestro) | `False` |
| `los_cero` | Boolean | Flag: ¿LOS es cero? | `False` |
| `fechas_invalidas` | Boolean | Flag: ¿Hay fechas inválidas? (siempre False en maestro) | `False` |
| `diagnosticos_primarios` | String | Lista de diagnósticos primarios (separados por `,`) | `E6601` |
| `diagnosticos_secundarios` | String | Lista de diagnósticos secundarios (separados por `,`) | `Z6841,I119` |
| `n_diag_primarios` | Integer | Número de diagnósticos primarios | `1` |
| `n_diag_secundarios` | Integer | Número de diagnósticos secundarios | `2` |
| `n_diag_total` | Integer | Total de diagnósticos (primarios + secundarios) | `3` |
| `tiene_diag_primario` | Boolean | Flag: ¿Tiene al menos un diagnóstico primario? | `True` |

**Notas importantes:**
- Las columnas de listas (`procedimientos`, `diagnosticos_*`) están en formato string separadas por comas
- Para usar en Python: `df['procedimientos'].str.split(',')` para convertir a lista
- **LOS = 0 SÍ está incluido** (son estancias válidas de admisión/egreso el mismo día)

---

### 2. `pacientes_rechazados.csv`

**Descripción:** Pacientes que NO pasaron los criterios de validación.

**Formato:** CSV con separador `;`

**Columnas principales:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `case_id` | String | Identificador del paciente rechazado |
| `motivo_rechazo` | String | Razón(es) de rechazo (separadas por `\|`) |
| `los_dias` | Integer/NaN | LOS calculado (puede ser nulo) |
| `n_procedimientos` | Integer | Número de procedimientos |
| `n_diag_total` | Integer | Número de diagnósticos |
| `fecha_ingreso` | Date/NaT | Fecha de ingreso |
| `fecha_egreso` | Date/NaT | Fecha de egreso |

**Motivos de rechazo posibles:**

| Motivo | Descripción |
|--------|-------------|
| `falta_en_diagnosticos` | Paciente existe en procedimientos pero NO en diagnósticos |
| `falta_en_procedimientos` | Paciente existe en diagnósticos pero NO en procedimientos |
| `fechas_invalidas` | Fechas de ingreso o egreso son inválidas (NaT) |
| `los_negativo` | Fecha de egreso es anterior a fecha de ingreso (error en datos) |

**Motivos que NO causan rechazo:**
- ✅ **LOS = 0** (son datos válidos)
- ✅ **Códigos ICD-10 inválidos** (se filtran pero se conserva el paciente con sus códigos válidos)

---

### 3. `reporte_limpieza.csv`

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
pacientes_maestro;11951
pacientes_rechazados;0
tasa_aceptacion_pct;100.0
los_promedio_dias;6.5
```

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

#### **Bloque 3: Validación de códigos ICD-10**
```python
# REGEX para ICD-10: Letra (excepto U) + 2-6 caracteres alfanuméricos
ICD10_REGEX = r'^[A-TV-Z][0-9]{1,2}[A-Z0-9]{0,5}$'

# Validar cada código contra el patrón
df['diagnosis_valido'] = df['diagnosis'].str.match(ICD10_REGEX, na=False)

# Ejemplos válidos: E6601, S72302E, I10
# Ejemplos inválidos: UUUUUU, AAAAAA, 123ABC
```

#### **Bloque 4: Filtrado de registros inválidos (NO pacientes completos)**
```python
# DECISIÓN CRÍTICA: Solo eliminar REGISTROS con códigos inválidos
# NO eliminar pacientes completos
df_valido = df[df['princsec_valido'] & df['diagnosis_valido']].copy()

# Ejemplo:
# Paciente 12345 tiene diagnósticos: E6601 (válido), UUUUUU (inválido), I119 (válido)
# Resultado: Se conserva paciente con E6601 e I119, se elimina solo UUUUUU
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

**Propósito:** Validar fechas, calcular LOS y agregar por paciente.

#### **Bloque 1: Conversión de fechas**
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

#### **Bloque 2: Agregación de fechas por paciente**
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

#### **Bloque 3: Cálculo de LOS**
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

**Ejecución: 2026-03-18**

| Métrica | Valor |
|---------|-------|
| **Pacientes totales** | 11,951 |
| **Pacientes en dataset maestro** | 11,951 (100%) ✅ |
| **Pacientes rechazados** | 0 (0%) |
| **Registros diagnóstico inválidos filtrados** | 3,509 (códigos UUUUUU, AAAAAA) |
| **Pacientes con LOS = 0** | 250 (conservados) |
| **LOS promedio** | 6.5 días |
| **LOS máximo** | 262 días |
| **Procedimientos promedio** | 2.2 por paciente |
| **Diagnósticos promedio** | 7.9 por paciente |

---

## 🎯 Decisiones de Diseño Importantes

### ✅ **Por qué NO se rechazaron pacientes por códigos ICD-10 inválidos**

**Problema anterior:** El sistema antiguo rechazaba TODO el paciente si tenía un solo código `UUUUUU` (desconocido), perdiendo el 29% de los datos.

**Solución actual:** Se **filtran solo los códigos inválidos**, pero se **conserva el paciente** con sus códigos válidos.

**Ejemplo:**
```
Paciente 12345:
- Diagnósticos originales: E6601 (válido), UUUUUU (inválido), I119 (válido)
- Sistema ANTERIOR: Rechaza paciente completo ❌
- Sistema ACTUAL: Conserva paciente con [E6601, I119] ✅
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

### Verificar códigos ICD-10 filtrados
```python
import pandas as pd

df = pd.read_csv('datos_diagnostico.csv', sep=';')

# Ver códigos inválidos únicos
import re
regex = r'^[A-TV-Z][0-9]{1,2}[A-Z0-9]{0,5}$'
invalidos = df[~df['Diagnosis'].str.match(regex, na=False)]
print(invalidos['Diagnosis'].value_counts())

# Output esperado:
# UUUUUU    3486
# AAAAAA      23
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

**Última actualización:** 2026-03-18
**Versión del script:** 2.0 (Simplificado)
**Mejoras clave:**
- ✅ Recuperados 3,486 pacientes (29% más de datos)
- ✅ Código simplificado de 17 archivos a 1
- ✅ Documentación línea por línea completa
- ✅ LOS=0 conservados como datos válidos
