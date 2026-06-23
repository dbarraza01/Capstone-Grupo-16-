"""
procesamiento_features_v3.py
============================
Fase 2 del pipeline de predicción de LOS — Ingeniería de Features v3

Calcula el Índice de Comorbilidad de Charlson y las categorías de Elixhauser
usando la librería `comorbidipy` (MIT License), que implementa internamente
el algoritmo de Quan et al. (2005) con los mapeos oficiales ICD-10.

Genera dos datasets para comparación en la Fase 3:
  - Escenario B: Dataset v2 completo + columna charlson_index
  - Escenario C: Variables base + Procedimientos + 31 categorías Elixhauser
                 (reemplaza los ~1,300 grupos ICD dispersos de la v2)

Fuente del algoritmo:
  Librería comorbidipy v0.5.0 (https://github.com/vvcb/comorbidipy)
  Mapeos internos basados en:
    Quan H, Sundararajan V, Halfon P, et al.
    "Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10
     administrative data." Medical Care, 2005;43(11):1130-1139.

  Pesos de Elixhauser basados en:
    van Walraven C, Austin PC, Jennings A, Quan H, Forster AJ.
    "A modification of the Elixhauser comorbidity measures into a point
     system for hospital death using administrative data."
    Medical Care, 2009;47(6):626-633.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from comorbidipy import comorbidity

# ============================================================================
# Rutas
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_maestro.csv"
V2_DATA_PATH  = BASE_DIR / "ml" / "feature_engineering" / "processed_v2" / "model_data_ml_v2.csv"
OUT_DIR       = BASE_DIR / "ml" / "feature_engineering" / "processed_v3"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Carga de datos
# ============================================================================
def load_data():
    """Lee el dataset maestro (para los códigos ICD originales) y la base v2."""
    print("Leyendo datos maestro y base procesada v2...")
    df_raw = pd.read_csv(RAW_DATA_PATH, sep=';')
    df_v2  = pd.read_csv(V2_DATA_PATH, sep=';')

    # Limpiar campos de diagnóstico
    df_raw['diagnosticos_primarios']   = df_raw['diagnosticos_primarios'].fillna('').astype(str).str.strip().str.upper()
    df_raw['diagnosticos_secundarios'] = df_raw['diagnosticos_secundarios'].fillna('').astype(str).str.strip().str.upper()

    print(f"  Dataset maestro: {len(df_raw)} pacientes")
    print(f"  Dataset v2:      {len(df_v2)} pacientes, {len(df_v2.columns)} columnas")
    return df_raw, df_v2

# ============================================================================
# 2. Preparar formato "long" para comorbidipy
# ============================================================================
def build_long_format(df_raw):
    """
    comorbidipy espera un DataFrame con columnas 'id', 'code' y 'age'.
    Explota los diagnósticos primarios y secundarios en filas individuales.
    Como no tenemos la edad de los pacientes, usamos age=0 para que
    comorbidipy calcule el score sin ajuste por edad.
    """
    print("Construyendo formato long (una fila por código por paciente)...")
    records = []
    for _, row in df_raw.iterrows():
        case_id = row['case_id']
        codes = []

        # Diagnóstico primario (1 código por paciente)
        if row['diagnosticos_primarios']:
            codes.append(row['diagnosticos_primarios'])

        # Diagnósticos secundarios (separados por coma)
        if row['diagnosticos_secundarios']:
            codes.extend([c.strip() for c in row['diagnosticos_secundarios'].split(',') if c.strip()])

        # Deduplica: un mismo código no debe contarse dos veces por paciente
        for code in set(codes):
            records.append({'id': case_id, 'code': code, 'age': 0})

    df_long = pd.DataFrame(records)
    print(f"  Registros generados: {len(df_long)} (filas código-paciente)")
    print(f"  Pacientes únicos:   {df_long['id'].nunique()}")
    return df_long

# ============================================================================
# 3. Calcular índices con comorbidipy
# ============================================================================
def calculate_indices(df_long):
    """
    Usa comorbidipy para calcular:
      - Charlson (variant='quan', icd='icd10')
      - Elixhauser (variant='quan', icd='icd10', weighting='vw')

    Los mapeos ICD-10 → categorías clínicas vienen incluidos dentro de
    comorbidipy y corresponden al algoritmo de Quan et al. (2005).
    """

    # --- Charlson ---
    print("\nCalculando Índice de Charlson (comorbidipy, Quan ICD-10)...")
    df_charlson = comorbidity(
        df_long,
        id='id',
        code='code',
        age='age',
        score='charlson',
        icd='icd10',
        variant='quan'
    )
    # Renombrar columnas para claridad
    df_charlson = df_charlson.rename(columns={
        'id': 'case_id',
        'comorbidity_score': 'charlson_index'
    })
    # Eliminar age y age_adj_comorbidity_score (no tenemos edad real)
    cols_to_drop = [c for c in ['age', 'age_adj_comorbidity_score'] if c in df_charlson.columns]
    df_charlson = df_charlson.drop(columns=cols_to_drop)

    print(f"  Categorías Charlson detectadas: {len([c for c in df_charlson.columns if c not in ['case_id', 'charlson_index']])}")
    print(f"  Distribución charlson_index:")
    print(df_charlson['charlson_index'].value_counts().sort_index().to_string())

    # --- Elixhauser ---
    print("\nCalculando Índice de Elixhauser (comorbidipy, Quan ICD-10, pesos van Walraven)...")
    df_elixhauser = comorbidity(
        df_long,
        id='id',
        code='code',
        age='age',
        score='elixhauser',
        icd='icd10',
        variant='quan',
        weighting='vw'  # van Walraven weights
    )
    df_elixhauser = df_elixhauser.rename(columns={
        'id': 'case_id',
        'comorbidity_score': 'elixhauser_score'
    })
    cols_to_drop = [c for c in ['age', 'age_adj_comorbidity_score'] if c in df_elixhauser.columns]
    df_elixhauser = df_elixhauser.drop(columns=cols_to_drop)

    # Prefixear las categorías de Elixhauser para evitar confusión con las de Charlson
    elix_cat_cols = [c for c in df_elixhauser.columns if c not in ['case_id', 'elixhauser_score']]
    rename_map = {c: f"elix_{c}" for c in elix_cat_cols}
    df_elixhauser = df_elixhauser.rename(columns=rename_map)

    n_cats = len(elix_cat_cols)
    print(f"  Categorías Elixhauser detectadas: {n_cats}")

    return df_charlson, df_elixhauser

# ============================================================================
# 4. Generar datasets de escenarios
# ============================================================================
def main():
    df_raw, df_v2 = load_data()
    df_long = build_long_format(df_raw)
    df_charlson, df_elixhauser = calculate_indices(df_long)

    # ========================================================================
    # ESCENARIO B: Dataset v2 completo + charlson_index
    # ========================================================================
    print("\n" + "="*60)
    print("Generando Escenario B (v2 + Charlson)...")
    print("="*60)

    df_b = pd.merge(
        df_v2,
        df_charlson[['case_id', 'charlson_index']],
        on='case_id',
        how='left'
    )
    df_b['charlson_index'] = df_b['charlson_index'].fillna(0).astype(int)

    out_b = OUT_DIR / "model_data_v3_escenario_B_charlson.csv"
    df_b.to_csv(out_b, sep=';', index=False)
    print(f"✅ Escenario B: {len(df_b)} filas, {len(df_b.columns)} columnas")
    print(f"   Guardado en: {out_b.name}")

    # ========================================================================
    # ESCENARIO C: Base + Procedimientos + Elixhauser (sin grupos ICD v2)
    # ========================================================================
    print("\n" + "="*60)
    print("Generando Escenario C (Base + Procedimientos + Elixhauser)...")
    print("="*60)

    # Conservar todo MENOS las columnas de diagnóstico de la v2
    cols_to_keep = [col for col in df_v2.columns if not col.startswith('diag_')]
    df_c_base = df_v2[cols_to_keep].copy()

    # Agregar todas las columnas de Elixhauser (categorías binarias + score)
    df_c = pd.merge(
        df_c_base,
        df_elixhauser,
        on='case_id',
        how='left'
    )
    # Llenar nulos con 0
    elix_cols = [c for c in df_c.columns if c.startswith('elix_') or c == 'elixhauser_score']
    for col in elix_cols:
        df_c[col] = df_c[col].fillna(0)

    out_c = OUT_DIR / "model_data_v3_escenario_C_elixhauser.csv"
    df_c.to_csv(out_c, sep=';', index=False)
    print(f"✅ Escenario C: {len(df_c)} filas, {len(df_c.columns)} columnas")
    print(f"   Guardado en: {out_c.name}")

    # ========================================================================
    # Resumen comparativo
    # ========================================================================
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"  Escenario A (v2 original):  {len(df_v2.columns)} columnas")
    print(f"  Escenario B (v2 + Charlson): {len(df_b.columns)} columnas (+1)")
    print(f"  Escenario C (Elixhauser):    {len(df_c.columns)} columnas")
    print(f"\nProcesamiento Fase 2 completado exitosamente.")

if __name__ == "__main__":
    main()
