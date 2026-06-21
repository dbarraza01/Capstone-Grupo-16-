"""
procesamiento_features_v2.py
============================
Pipeline v2 de ingeniería de features para predicción de LOS hospitalario.

Cambios respecto a v1:
1. UUUUUU excluido de la matriz diagnóstica (no es código clínico).
2. Fallback a capítulo usa prefijo diag_rare_cap_* para distinguir de categorías reales.
3. Procedimientos: umbral de código completo = 10 (más conservador); fallback a sección
   usa prefijo proc_rare_sec_*.
4. Features de repetición/carga de códigos (6 columnas numéricas).
5. Validación descriptiva de los_dias y análisis de outliers (P95).
6. Reportes ampliados.

Decisiones documentadas:
- Se usan diagnósticos primarios Y secundarios (autorizado por el profesor).
- No se implementa estratificación por servicio (trabajo futuro).
- Conteos brutos para features de repetición (RF/GBM no requieren normalización).
- Outlier = P95. No se crean features de outlier en el dataset principal (riesgo de leakage).
"""

import pandas as pd
import numpy as np
import os

print("🚀 Iniciando procesamiento de features v2 para Machine Learning...\n")

# ============================================================
# CONFIGURACIÓN
# ============================================================
os.makedirs("ml/feature_engineering/processed_v2", exist_ok=True)
os.makedirs("ml/feature_engineering/reports_features", exist_ok=True)

UMBRAL_DIAG = 20           # Umbral para diagnósticos (código completo y cat. 3 chars)
UMBRAL_PROC_CODE = 10      # Umbral para procedimientos — código completo (más bajo que v1)
UMBRAL_PROC_CAT3 = 20      # Umbral para procedimientos — categoría 3 chars
CODIGO_URGENCIA = "UUUUUU"

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print("📂 Cargando datasets...")
df_maestro = pd.read_csv("data/processed/dataset_maestro.csv", sep=";")
df_diag = pd.read_csv("data/processed/caso_diagnostico.csv", sep=";", dtype=str)
df_proc = pd.read_csv("data/processed/caso_procedimiento.csv", sep=";", dtype=str)

print(f"   Dataset maestro: {len(df_maestro)} pacientes, {len(df_maestro.columns)} columnas")
print(f"   Registros diagnósticos: {len(df_diag)}")
print(f"   Registros procedimentales: {len(df_proc)}")

# Limpieza básica de strings
for df, col in [(df_diag, 'd_code'), (df_diag, 'd_caract_3'), (df_diag, 'd_caract_1'),
                (df_proc, 'p_code'), (df_proc, 'p_caract_3'), (df_proc, 'p_caract_1')]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Nulos en datos fuente
print("\n📋 Nulos en datos fuente:")
print(f"   Diagnósticos: {df_diag.isnull().sum().to_dict()}")
print(f"   Procedimientos: {df_proc.isnull().sum().to_dict()}")

# ============================================================
# 2. TRATAMIENTO DE UUUUUU
# ============================================================
print(f"\n🏥 Analizando código '{CODIGO_URGENCIA}'...")
mask_uuuuuu = df_diag['d_code'] == CODIGO_URGENCIA
n_registros_uu = mask_uuuuuu.sum()
n_pacientes_uu = df_diag.loc[mask_uuuuuu, 'case_id'].nunique()
total_pacientes = df_maestro['case_id'].nunique()

# Verificar si aparece como primario o secundario
if 'tipo_d' in df_diag.columns:
    tipo_uu = df_diag.loc[mask_uuuuuu, 'tipo_d'].value_counts().to_dict()
else:
    tipo_uu = {"info_no_disponible": n_registros_uu}

print(f"   Registros con {CODIGO_URGENCIA}: {n_registros_uu}")
print(f"   Pacientes únicos con {CODIGO_URGENCIA}: {n_pacientes_uu} ({100*n_pacientes_uu/total_pacientes:.1f}%)")
print(f"   Distribución por tipo: {tipo_uu}")
print(f"   → DECISIÓN: Excluir de la matriz diagnóstica predictiva.")

# Crear flag auxiliar (NO va al dataset principal por defecto)
pacientes_uu = set(df_diag.loc[mask_uuuuuu, 'case_id'].unique())

# Excluir UUUUUU de los diagnósticos antes de procesar
df_diag_clean = df_diag[~mask_uuuuuu].copy()
print(f"   Registros diagnósticos después de excluir {CODIGO_URGENCIA}: {len(df_diag_clean)}")

# ============================================================
# 3. PROCESAMIENTO DE DIAGNÓSTICOS (v2)
# ============================================================
print(f"\n🏥 Procesando {len(df_diag_clean)} diagnósticos...")

# Frecuencias por paciente único a cada nivel
freq_d_code = df_diag_clean.groupby('d_code')['case_id'].nunique().to_dict()
freq_d_3 = df_diag_clean.groupby('d_caract_3')['case_id'].nunique().to_dict()
freq_d_1 = df_diag_clean.groupby('d_caract_1')['case_id'].nunique().to_dict()

print(f"   Códigos únicos d_code: {len(freq_d_code)}")
print(f"   Categorías 3-char únicas: {len(freq_d_3)}")
print(f"   Capítulos 1-char únicos: {len(freq_d_1)}")

def agrupar_diagnostico_v2(row):
    """
    Regla jerárquica v2 para diagnósticos:
    1. Código completo si >= UMBRAL_DIAG pacientes
    2. Categoría 3 chars si >= UMBRAL_DIAG pacientes
    3. Capítulo 1 char con prefijo rare_cap_ (distingue de categorías reales)
    """
    code = row['d_code']
    cat3 = row['d_caract_3']
    cap1 = row['d_caract_1']

    if freq_d_code.get(code, 0) >= UMBRAL_DIAG:
        return code, "codigo_completo"
    elif freq_d_3.get(cat3, 0) >= UMBRAL_DIAG:
        return cat3, "categoria_3_char"
    else:
        return f"rare_cap_{cap1}", "capitulo_rare"

# Aplicar regla
resultados_diag = df_diag_clean.apply(agrupar_diagnostico_v2, axis=1)
df_diag_clean['grupo_final'] = [r[0] for r in resultados_diag]
df_diag_clean['nivel_agrupacion'] = [r[1] for r in resultados_diag]

# Reportes de mapeo
print("📊 Generando reportes de diagnósticos v2...")
mapeo_diag = df_diag_clean[['d_code', 'd_caract_3', 'd_caract_1', 'grupo_final', 'nivel_agrupacion']].drop_duplicates()
mapeo_diag['pacientes_d_code'] = mapeo_diag['d_code'].map(freq_d_code)
mapeo_diag['pacientes_d_caract_3'] = mapeo_diag['d_caract_3'].map(freq_d_3)
mapeo_diag['pacientes_d_caract_1'] = mapeo_diag['d_caract_1'].map(freq_d_1)
mapeo_diag.to_csv("ml/feature_engineering/reports_features/mapeo_reemplazos_diagnosticos.csv", sep=";", index=False)

# Reporte de frecuencias finales
frec_final_diag = df_diag_clean.groupby('grupo_final')['case_id'].nunique().reset_index(name='pacientes_unicos')
frec_final_diag.sort_values('pacientes_unicos', ascending=False).to_csv(
    "ml/feature_engineering/reports_features/reporte_frecuencias_diagnosticos.csv", sep=";", index=False)

# Matriz binaria
print("⚙️  Creando matriz binaria de diagnósticos v2...")
df_diag_clean['grupo_final_col'] = 'diag_' + df_diag_clean['grupo_final']
matriz_diag = pd.crosstab(df_diag_clean['case_id'], df_diag_clean['grupo_final_col'])
matriz_diag = (matriz_diag > 0).astype(int)
matriz_diag.to_csv("ml/feature_engineering/processed_v2/features_diagnosticos_agrupados_v2.csv", sep=";")

# ============================================================
# 4. PROCESAMIENTO DE PROCEDIMIENTOS (v2)
# ============================================================
print(f"\n💉 Procesando {len(df_proc)} procedimientos...")

freq_p_code = df_proc.groupby('p_code')['case_id'].nunique().to_dict()
freq_p_3 = df_proc.groupby('p_caract_3')['case_id'].nunique().to_dict()
freq_p_1 = df_proc.groupby('p_caract_1')['case_id'].nunique().to_dict()

print(f"   Códigos únicos p_code: {len(freq_p_code)}")
print(f"   Categorías 3-char únicas: {len(freq_p_3)}")
print(f"   Secciones 1-char únicas: {len(freq_p_1)}")

def agrupar_procedimiento_v2(row):
    """
    Regla jerárquica v2 para procedimientos:
    1. Código completo si >= UMBRAL_PROC_CODE pacientes (10, más bajo que diag)
    2. Categoría 3 chars si >= UMBRAL_PROC_CAT3 pacientes (20)
    3. Sección 1 char con prefijo rare_sec_ (distingue de categorías reales)
    """
    code = row['p_code']
    cat3 = row['p_caract_3']
    sec1 = row['p_caract_1']

    if freq_p_code.get(code, 0) >= UMBRAL_PROC_CODE:
        return code, "codigo_completo"
    elif freq_p_3.get(cat3, 0) >= UMBRAL_PROC_CAT3:
        return cat3, "categoria_3_char"
    else:
        return f"rare_sec_{sec1}", "seccion_rare"

resultados_proc = df_proc.apply(agrupar_procedimiento_v2, axis=1)
df_proc['grupo_final'] = [r[0] for r in resultados_proc]
df_proc['nivel_agrupacion'] = [r[1] for r in resultados_proc]

# Reportes
print("📊 Generando reportes de procedimientos v2...")
mapeo_proc = df_proc[['p_code', 'p_caract_3', 'p_caract_1', 'grupo_final', 'nivel_agrupacion']].drop_duplicates()
mapeo_proc['pacientes_p_code'] = mapeo_proc['p_code'].map(freq_p_code)
mapeo_proc['pacientes_p_caract_3'] = mapeo_proc['p_caract_3'].map(freq_p_3)
mapeo_proc['pacientes_p_caract_1'] = mapeo_proc['p_caract_1'].map(freq_p_1)
mapeo_proc.to_csv("ml/feature_engineering/reports_features/mapeo_reemplazos_procedimientos.csv", sep=";", index=False)

frec_final_proc = df_proc.groupby('grupo_final')['case_id'].nunique().reset_index(name='pacientes_unicos')
frec_final_proc.sort_values('pacientes_unicos', ascending=False).to_csv(
    "ml/feature_engineering/reports_features/reporte_frecuencias_procedimientos.csv", sep=";", index=False)

# Matriz binaria
print("⚙️  Creando matriz binaria de procedimientos v2...")
df_proc['grupo_final_col'] = 'proc_' + df_proc['grupo_final']
matriz_proc = pd.crosstab(df_proc['case_id'], df_proc['grupo_final_col'])
matriz_proc = (matriz_proc > 0).astype(int)
matriz_proc.to_csv("ml/feature_engineering/processed_v2/features_procedimientos_agrupados_v2.csv", sep=";")

# ============================================================
# 5. FEATURES DE REPETICIÓN / CARGA DE CÓDIGOS
# ============================================================
print("\n🔄 Calculando features de repetición de códigos...")

# --- Diagnósticos ---
# Conteo de registros por paciente (un código repetido N veces)
rep_diag_por_paciente = df_diag_clean.groupby('case_id').agg(
    total_registros_diag=('d_code', 'size'),
    codigos_unicos_diag=('d_code', 'nunique'),
    grupos_unicos_diag=('grupo_final', 'nunique')
).reset_index()
rep_diag_por_paciente['n_diag_codigos_repetidos'] = (
    rep_diag_por_paciente['total_registros_diag'] - rep_diag_por_paciente['codigos_unicos_diag']
)
# Repetición máxima de un grupo diagnóstico en un paciente
max_rep_diag = df_diag_clean.groupby(['case_id', 'grupo_final']).size().reset_index(name='count')
max_rep_diag_max = max_rep_diag.groupby('case_id')['count'].max().reset_index(name='max_repeticion_diag_grupo')

rep_diag = rep_diag_por_paciente.merge(max_rep_diag_max, on='case_id', how='left')

# --- Procedimientos ---
rep_proc_por_paciente = df_proc.groupby('case_id').agg(
    total_registros_proc=('p_code', 'size'),
    codigos_unicos_proc=('p_code', 'nunique'),
    grupos_unicos_proc=('grupo_final', 'nunique')
).reset_index()
rep_proc_por_paciente['n_proc_codigos_repetidos'] = (
    rep_proc_por_paciente['total_registros_proc'] - rep_proc_por_paciente['codigos_unicos_proc']
)
max_rep_proc = df_proc.groupby(['case_id', 'grupo_final']).size().reset_index(name='count')
max_rep_proc_max = max_rep_proc.groupby('case_id')['count'].max().reset_index(name='max_repeticion_proc_grupo')

rep_proc = rep_proc_por_paciente.merge(max_rep_proc_max, on='case_id', how='left')

# Reporte de repeticiones
rep_merge = rep_diag[['case_id', 'n_diag_codigos_repetidos', 'grupos_unicos_diag', 'max_repeticion_diag_grupo']].merge(
    rep_proc[['case_id', 'n_proc_codigos_repetidos', 'grupos_unicos_proc', 'max_repeticion_proc_grupo']],
    on='case_id', how='outer'
)
rep_merge.to_csv("ml/feature_engineering/reports_features/reporte_repeticiones_codigos.csv", sep=";", index=False)
print(f"   Pacientes con repeticiones de diagnósticos: {(rep_merge['n_diag_codigos_repetidos'] > 0).sum()}")
print(f"   Pacientes con repeticiones de procedimientos: {(rep_merge['n_proc_codigos_repetidos'] > 0).sum()}")

# ============================================================
# 6. VALIDACIÓN DEL TARGET (los_dias)
# ============================================================
print("\n📏 Validación de la variable objetivo: los_dias")
los = df_maestro['los_dias'].dropna()

percentiles = {
    'min': los.min(),
    'p5': los.quantile(0.05),
    'p10': los.quantile(0.10),
    'p25': los.quantile(0.25),
    'mediana': los.median(),
    'media': los.mean(),
    'p75': los.quantile(0.75),
    'p90': los.quantile(0.90),
    'p95': los.quantile(0.95),
    'p99': los.quantile(0.99),
    'max': los.max(),
    'std': los.std(),
    'n_total': len(los),
    'n_nulos': df_maestro['los_dias'].isnull().sum(),
}

p95_value = los.quantile(0.95)
n_outliers_p95 = (los >= p95_value).sum()
percentiles['umbral_outlier_p95'] = p95_value
percentiles['n_outliers_p95'] = n_outliers_p95
percentiles['pct_outliers_p95'] = 100 * n_outliers_p95 / len(los)

# Guardar reporte
reporte_los = pd.DataFrame([percentiles])
reporte_los.to_csv("ml/feature_engineering/reports_features/reporte_target_los.csv", sep=";", index=False)

for k, v in percentiles.items():
    print(f"   {k}: {v}")

# ============================================================
# 7. ANÁLISIS DE OUTLIERS (P95)
# ============================================================
print(f"\n📌 Análisis de outliers (LOS >= P95 = {p95_value} días)...")
outlier_case_ids = set(df_maestro.loc[los >= p95_value, 'case_id'].astype(str).values)

# Diagnósticos frecuentes en outliers
diag_outliers = df_diag_clean[df_diag_clean['case_id'].astype(str).isin(outlier_case_ids)]
if len(diag_outliers) > 0:
    top_diag_outliers = diag_outliers.groupby('grupo_final')['case_id'].nunique().reset_index(name='pacientes_outlier')
    top_diag_outliers['tipo'] = 'diagnostico'
    top_diag_outliers = top_diag_outliers.sort_values('pacientes_outlier', ascending=False)
else:
    top_diag_outliers = pd.DataFrame(columns=['grupo_final', 'pacientes_outlier', 'tipo'])

# Procedimientos frecuentes en outliers
proc_outliers = df_proc[df_proc['case_id'].astype(str).isin(outlier_case_ids)]
if len(proc_outliers) > 0:
    top_proc_outliers = proc_outliers.groupby('grupo_final')['case_id'].nunique().reset_index(name='pacientes_outlier')
    top_proc_outliers['tipo'] = 'procedimiento'
    top_proc_outliers = top_proc_outliers.sort_values('pacientes_outlier', ascending=False)
else:
    top_proc_outliers = pd.DataFrame(columns=['grupo_final', 'pacientes_outlier', 'tipo'])

reporte_outliers = pd.concat([top_diag_outliers.head(30), top_proc_outliers.head(30)], ignore_index=True)
reporte_outliers.to_csv("ml/feature_engineering/reports_features/reporte_codigos_outliers.csv", sep=";", index=False)
print(f"   Pacientes outlier analizados: {len(outlier_case_ids)}")
print(f"   Top diagnóstico en outliers: {top_diag_outliers.iloc[0]['grupo_final'] if len(top_diag_outliers) > 0 else 'N/A'}")
print(f"   Top procedimiento en outliers: {top_proc_outliers.iloc[0]['grupo_final'] if len(top_proc_outliers) > 0 else 'N/A'}")

# ============================================================
# 8. CREACIÓN DEL DATASET FINAL ML v2
# ============================================================
print("\n🔗 Integrando features en el dataset final v2...")

# Convertir IDs a string para el merge
df_maestro['case_id'] = df_maestro['case_id'].astype(str)
matriz_diag.index = matriz_diag.index.astype(str)
matriz_proc.index = matriz_proc.index.astype(str)

# Features temporales
df_maestro['fecha_ingreso'] = pd.to_datetime(df_maestro['fecha_ingreso'], errors='coerce')
df_maestro['mes_ingreso'] = df_maestro['fecha_ingreso'].dt.month.fillna(-1).astype(int)
df_maestro['dia_semana_ingreso'] = df_maestro['fecha_ingreso'].dt.dayofweek.fillna(-1).astype(int)

# Filtro de columnas leakage y de texto crudo
columnas_a_eliminar = [
    'fecha_ingreso', 'fecha_egreso',
    'los_cero', 'los_negativo', 'fechas_invalidas',
    'procedimientos', 'diagnosticos_primarios', 'diagnosticos_secundarios'
]
df_ml = df_maestro.drop(columns=[col for col in columnas_a_eliminar if col in df_maestro.columns])

# Merge con matrices binarias
df_ml = df_ml.merge(matriz_diag, on='case_id', how='left')
df_ml = df_ml.merge(matriz_proc, on='case_id', how='left')

# Merge con features de repetición
cols_rep_diag = ['case_id', 'n_diag_codigos_repetidos', 'grupos_unicos_diag', 'max_repeticion_diag_grupo']
cols_rep_proc = ['case_id', 'n_proc_codigos_repetidos', 'grupos_unicos_proc', 'max_repeticion_proc_grupo']

rep_diag['case_id'] = rep_diag['case_id'].astype(str)
rep_proc['case_id'] = rep_proc['case_id'].astype(str)

df_ml = df_ml.merge(rep_diag[cols_rep_diag], on='case_id', how='left')
df_ml = df_ml.merge(rep_proc[cols_rep_proc], on='case_id', how='left')

# Rellenar NaNs
cols_diag = matriz_diag.columns.tolist()
cols_proc = matriz_proc.columns.tolist()
df_ml[cols_diag + cols_proc] = df_ml[cols_diag + cols_proc].fillna(0).astype(int)

# Rellenar NaNs en features de repetición con 0
cols_rep = ['n_diag_codigos_repetidos', 'grupos_unicos_diag', 'max_repeticion_diag_grupo',
            'n_proc_codigos_repetidos', 'grupos_unicos_proc', 'max_repeticion_proc_grupo']
for c in cols_rep:
    if c in df_ml.columns:
        df_ml[c] = df_ml[c].fillna(0).astype(int)

# Guardar
path_final = "ml/feature_engineering/processed_v2/model_data_ml_v2.csv"
df_ml.to_csv(path_final, sep=";", index=False)

# ============================================================
# 9. RESUMEN FINAL
# ============================================================
# Identificar categorías de columnas
cols_base = ['case_id', 'los_dias', 'es_urgencia', 'n_procedimientos',
             'n_diag_primarios', 'n_diag_secundarios', 'n_diag_total',
             'tiene_diag_primario', 'mes_ingreso', 'dia_semana_ingreso']
cols_repeticion = [c for c in df_ml.columns if c in cols_rep]
cols_diag_final = [c for c in df_ml.columns if c.startswith('diag_')]
cols_proc_final = [c for c in df_ml.columns if c.startswith('proc_')]
cols_other = [c for c in df_ml.columns if c not in cols_base + cols_repeticion + cols_diag_final + cols_proc_final]

# Densidad de la matriz binaria
matrix_binary_cols = cols_diag_final + cols_proc_final
densidad = df_ml[matrix_binary_cols].mean().mean() * 100

# Distribución de jerarquías
dist_diag = df_diag_clean[['d_code', 'nivel_agrupacion']].drop_duplicates()['nivel_agrupacion'].value_counts()
dist_proc = df_proc[['p_code', 'nivel_agrupacion']].drop_duplicates()['nivel_agrupacion'].value_counts()

print("\n" + "=" * 60)
print("✅ PROCESAMIENTO v2 FINALIZADO CON ÉXITO")
print("=" * 60)
print(f"Pacientes procesados: {len(df_ml)}")
print(f"Total columnas finales: {len(df_ml.columns)}")
print(f"  - Features diagnósticas (diag_*): {len(cols_diag_final)}")
print(f"  - Features procedimentales (proc_*): {len(cols_proc_final)}")
print(f"  - Features base: {len([c for c in cols_base if c in df_ml.columns])}")
print(f"  - Features de repetición: {len(cols_repeticion)}")
print(f"  - Otras: {len(cols_other)}")
print(f"Densidad de la matriz binaria: {densidad:.2f}%")
print(f"Nulos restantes en dataset final: {df_ml.isnull().sum().sum()}")

print(f"\n--- Tratamiento de {CODIGO_URGENCIA} ---")
print(f"Pacientes con {CODIGO_URGENCIA}: {n_pacientes_uu} ({100*n_pacientes_uu/total_pacientes:.1f}%)")
print(f"Distribución por tipo: {tipo_uu}")
print(f"Decisión: EXCLUIDO de la matriz diagnóstica predictiva.")

print("\n--- Distribución de Jerarquías Diagnósticas (v2) ---")
print(dist_diag)

print("\n--- Distribución de Jerarquías Procedimentales (v2) ---")
print(dist_proc)

print(f"\n--- Target los_dias ---")
print(f"Media: {los.mean():.2f}, Mediana: {los.median():.1f}, Std: {los.std():.2f}")
print(f"P95: {p95_value}, Outliers (>= P95): {n_outliers_p95} ({100*n_outliers_p95/len(los):.1f}%)")

print(f"\n📁 Dataset final v2: {path_final}")
print(f"   → Para entrenar: X = todas las columnas EXCEPTO 'case_id' y 'los_dias'")
print(f"   → Target: y = 'los_dias'")
