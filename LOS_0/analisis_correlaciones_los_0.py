"""
Análisis de correlaciones entre Diagnósticos y Procedimientos para pacientes con LOS=0
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import os

# Configuración
OUTPUT_DIR = "."
DATA_DIR = "../data/processed"

# Cargar datos
df_maestro = pd.read_csv(f"{DATA_DIR}/dataset_maestro.csv", dtype=str, sep=';')
df_diag = pd.read_csv(f"{DATA_DIR}/caso_diagnostico.csv", dtype=str, sep=';')
df_proc = pd.read_csv(f"{DATA_DIR}/caso_procedimiento.csv", dtype=str, sep=';')

# Filtrar solo LOS=0
df_maestro['los_dias'] = pd.to_numeric(df_maestro['los_dias'], errors='coerce')
los_0_patients = df_maestro[df_maestro['los_dias'] == 0].copy()
los_0_ids = set(los_0_patients['case_id'].unique())

diag_los_0 = df_diag[df_diag['case_id'].isin(los_0_ids)].copy()
proc_los_0 = df_proc[df_proc['case_id'].isin(los_0_ids)].copy()

print(f"Análisis de {len(los_0_patients)} pacientes con LOS=0")
print(f"Diagnósticos: {len(diag_los_0)}")
print(f"Procedimientos: {len(proc_los_0)}\n")

# =============================================================================
# 1. ANÁLISIS DE DIAGNÓSTICOS PRINCIPALES
# =============================================================================

principal_diag = diag_los_0[diag_los_0['tipo_d'] == 'P'].copy()
top_principal = principal_diag['d_code'].value_counts().head(10)

print("TOP 10 DIAGNÓSTICOS PRINCIPALES:")
for idx, (code, count) in enumerate(top_principal.items(), 1):
    pct = 100 * count / len(principal_diag)
    print(f"  {idx}. {code}: {count} ({pct:.1f}%)")

# =============================================================================
# 2. MATRIZ DE CO-OCURRENCIA: DIAGNÓSTICO -> PROCEDIMIENTO
# =============================================================================

# Crear mapeo paciente -> diagnósticos principales
patient_principal_diag = {}
for _, row in principal_diag.iterrows():
    case_id = row['case_id']
    code = row['d_code']
    if case_id not in patient_principal_diag:
        patient_principal_diag[case_id] = []
    patient_principal_diag[case_id].append(code)

# Crear mapeo paciente -> procedimientos
patient_procedures = {}
for _, row in proc_los_0.iterrows():
    case_id = row['case_id']
    code = row['p_code']
    if case_id not in patient_procedures:
        patient_procedures[case_id] = []
    patient_procedures[case_id].append(code)

# Crear matriz de co-ocurrencia
top_proc = proc_los_0['p_code'].value_counts().head(12)

# Contar co-ocurrencias (diagnóstico-procedimiento por paciente)
coocurrence_matrix = {}
for diag in top_principal.index:
    coocurrence_matrix[diag] = {}
    for proc in top_proc.index:
        coocurrence_matrix[diag][proc] = 0

# Contar cuántas veces cada diagnóstico aparece con cada procedimiento
for patient_id in los_0_ids:
    diags = patient_principal_diag.get(patient_id, [])
    procs = patient_procedures.get(patient_id, [])

    for diag in diags:
        if diag in top_principal.index:
            for proc in procs:
                if proc in top_proc.index:
                    coocurrence_matrix[diag][proc] += 1

# Convertir a DataFrame
cooc_df = pd.DataFrame(coocurrence_matrix).T
cooc_df = cooc_df[[col for col in top_proc.index if col in cooc_df.columns]]

print("\nMATRIZ DE CO-OCURRENCIA (primeras 5 diagnósticos):")
print(cooc_df.head())

# =============================================================================
# 3. GRÁFICO 1: HEATMAP DE CORRELACIONES
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

# Crear heatmap
sns.heatmap(cooc_df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Frecuencia de Co-ocurrencia'},
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Matriz de Co-ocurrencia: Diagnósticos Principales vs Procedimientos (TOP 10 vs TOP 12)',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Códigos de Procedimientos (ICD-10-PCS)', fontsize=11, fontweight='bold')
ax.set_ylabel('Diagnósticos Principales (ICD-10-CM)', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_heatmap_correlaciones_los_0.png', dpi=150, bbox_inches='tight')
print("✓ Generado: 04_heatmap_correlaciones_los_0.png")
plt.close()

# =============================================================================
# 4. GRÁFICO 2: PROCEDIMIENTOS ASOCIADOS POR DIAGNÓSTICO PRINCIPAL
# =============================================================================

# Para cada diagnóstico principal en TOP 5, mostrar sus procedimientos más frecuentes
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
axes = axes.flatten()

top_5_diag = top_principal.head(5).index

for idx, diag in enumerate(top_5_diag):
    ax = axes[idx]

    # Obtener pacientes con este diagnóstico
    patients_with_diag = []
    for patient_id, diag_list in patient_principal_diag.items():
        if diag in diag_list:
            patients_with_diag.append(patient_id)

    # Obtener procedimientos de estos pacientes
    procs_in_patients = []
    for patient_id in patients_with_diag:
        procs_in_patients.extend(patient_procedures.get(patient_id, []))

    # Contar frecuencias
    proc_counts = pd.Series(procs_in_patients).value_counts().head(8)

    # Gráfico de barras
    colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(proc_counts)))
    ax.barh(range(len(proc_counts)), proc_counts.values, color=colors, edgecolor='black', linewidth=0.7)

    ax.set_yticks(range(len(proc_counts)))
    ax.set_yticklabels(proc_counts.index, fontsize=10, fontweight='bold')
    ax.set_xlabel('Frecuencia', fontsize=10, fontweight='bold')
    ax.set_title(f'{diag} (n={len(patients_with_diag)} pacientes)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Agregar valores
    for i, val in enumerate(proc_counts.values):
        pct = 100 * val / len(procs_in_patients)
        ax.text(val + 0.1, i, f'{int(val)} ({pct:.1f}%)', va='center', fontsize=9)

# Desactivar el sexto subplot
axes[5].axis('off')

plt.suptitle('Procedimientos Asociados a Diagnósticos Principales (TOP 5) en LOS=0',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_procedimientos_por_diagnostico_los_0.png', dpi=150, bbox_inches='tight')
print("✓ Generado: 05_procedimientos_por_diagnostico_los_0.png")
plt.close()

# =============================================================================
# 5. GRÁFICO 3: DIAGNÓSTICOS ASOCIADOS A PROCEDIMIENTOS (INVERSO)
# =============================================================================

# Crear mapeo inverso: paciente -> diagnósticos
patient_all_diag = {}
for _, row in diag_los_0.iterrows():
    case_id = row['case_id']
    code = row['d_code']
    if case_id not in patient_all_diag:
        patient_all_diag[case_id] = []
    patient_all_diag[case_id].append(code)

# Para cada procedimiento en TOP 5, mostrar sus diagnósticos más frecuentes
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
axes = axes.flatten()

top_5_proc = top_proc.head(5).index

for idx, proc in enumerate(top_5_proc):
    ax = axes[idx]

    # Obtener pacientes con este procedimiento
    patients_with_proc = []
    for patient_id, proc_list in patient_procedures.items():
        if proc in proc_list:
            patients_with_proc.append(patient_id)

    # Obtener diagnósticos principales de estos pacientes
    diags_in_patients = []
    for patient_id in patients_with_proc:
        # Solo diagnósticos principales
        patient_diags = principal_diag[principal_diag['case_id'] == patient_id]['d_code'].tolist()
        diags_in_patients.extend(patient_diags)

    # Contar frecuencias
    diag_counts = pd.Series(diags_in_patients).value_counts().head(8)

    if len(diag_counts) == 0:
        ax.text(0.5, 0.5, 'Sin diagnósticos', ha='center', va='center', fontsize=10)
        ax.set_title(f'{proc} (n={len(patients_with_proc)} pacientes)', fontsize=11, fontweight='bold')
        continue

    # Gráfico de barras
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(diag_counts)))
    ax.barh(range(len(diag_counts)), diag_counts.values, color=colors, edgecolor='black', linewidth=0.7)

    ax.set_yticks(range(len(diag_counts)))
    ax.set_yticklabels(diag_counts.index, fontsize=10, fontweight='bold')
    ax.set_xlabel('Frecuencia', fontsize=10, fontweight='bold')
    ax.set_title(f'{proc} (n={len(patients_with_proc)} pacientes)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Agregar valores
    for i, val in enumerate(diag_counts.values):
        pct = 100 * val / len(diags_in_patients) if len(diags_in_patients) > 0 else 0
        ax.text(val + 0.1, i, f'{int(val)} ({pct:.1f}%)', va='center', fontsize=9)

# Desactivar el sexto subplot
axes[5].axis('off')

plt.suptitle('Diagnósticos Principales Asociados a Procedimientos (TOP 5) en LOS=0',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_diagnosticos_por_procedimiento_los_0.png', dpi=150, bbox_inches='tight')
print("✓ Generado: 06_diagnosticos_por_procedimiento_los_0.png")
plt.close()

# =============================================================================
# 6. CSV DE CORRELACIONES: RESUMEN DETALLADO
# =============================================================================

# Crear CSV con las correlaciones más importantes
correlations_data = []

for diag in cooc_df.index:
    row = cooc_df.loc[diag]
    for proc in cooc_df.columns:
        count = row[proc]
        if count > 0:
            diag_count = top_principal[diag]
            proc_count = top_proc[proc]

            # Calcular tasas
            pct_in_diag = 100 * count / diag_count
            pct_in_proc = 100 * count / proc_count

            correlations_data.append({
                'Diagnóstico': diag,
                'Procedimiento': proc,
                'Co-ocurrencias': int(count),
                '% de pacientes con diagnóstico que reciben procedimiento': f'{pct_in_diag:.1f}%',
                '% de pacientes con procedimiento que tienen diagnóstico': f'{pct_in_proc:.1f}%',
                'Frecuencia_Diagnóstico': int(diag_count),
                'Frecuencia_Procedimiento': int(proc_count)
            })

correlations_df = pd.DataFrame(correlations_data).sort_values('Co-ocurrencias', ascending=False)
correlations_df.to_csv(f'{OUTPUT_DIR}/correlaciones_diagnostico_procedimiento_los_0.csv', index=False, sep=';')

print("\n✓ Generado: correlaciones_diagnostico_procedimiento_los_0.csv")
print(f"  Total de combinaciones encontradas: {len(correlations_df)}")
print("\nTOP 15 Correlaciones Diagnóstico-Procedimiento:")
print(correlations_df.head(15).to_string(index=False))

# =============================================================================
# 7. RESUMEN ESTADÍSTICO
# =============================================================================

summary_text = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║           ANÁLISIS DE CORRELACIONES: DIAGNÓSTICOS-PROCEDIMIENTOS          ║
║                         PACIENTES CON LOS=0                                ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 DATOS GENERALES:
   • Total de pacientes: {len(los_0_patients)}
   • Total de diagnósticos registrados: {len(diag_los_0)}
   • Total de procedimientos registrados: {len(proc_los_0)}
   • Diagnósticos principales analizados: TOP 10
   • Procedimientos analizados: TOP 12

🏥 DIAGNÓSTICOS PRINCIPALES (TOP 10):
"""

for idx, (code, count) in enumerate(top_principal.items(), 1):
    pct = 100 * count / len(principal_diag)
    summary_text += f"   {idx}. {code}: {count} ({pct:.1f}%)\n"

summary_text += f"\n🔧 PROCEDIMIENTOS (TOP 12):\n"

for idx, (code, count) in enumerate(top_proc.items(), 1):
    pct = 100 * count / len(proc_los_0)
    summary_text += f"   {idx}. {code}: {count} ({pct:.1f}%)\n"

summary_text += f"""
📈 HALLAZGOS CLAVE:

1. CO-OCURRENCIA:
   • Total de combinaciones diagnóstico-procedimiento: {len(correlations_df)}
   • Correlación más fuerte: {correlations_df.iloc[0]['Diagnóstico']} + {correlations_df.iloc[0]['Procedimiento']} ({int(correlations_df.iloc[0]['Co-ocurrencias'])} veces)

2. DIAGNÓSTICOS MÁS VERSÁTILES:
   • {cooc_df.sum(axis=1).idxmax()}: co-ocurre con {(cooc_df.loc[cooc_df.sum(axis=1).idxmax()] > 0).sum()} procedimientos diferentes
   • {cooc_df.sum(axis=1).nlargest(2).index[1]}: co-ocurre con {(cooc_df.loc[cooc_df.sum(axis=1).nlargest(2).index[1]] > 0).sum()} procedimientos diferentes

3. PROCEDIMIENTOS MÁS APLICADOS:
   • {cooc_df.sum(axis=0).idxmax()}: se asocia con {(cooc_df[cooc_df.sum(axis=0).idxmax()] > 0).sum()} diagnósticos principales diferentes
   • {cooc_df.sum(axis=0).nlargest(2).index[1]}: se asocia con {(cooc_df[cooc_df.sum(axis=0).nlargest(2).index[1]] > 0).sum()} diagnósticos principales diferentes

4. PATRONES OBSERVADOS:
   • Muchos procedimientos se asocian con múltiples diagnósticos (patrón de "hub")
   • Algunos diagnósticos tienen procedimientos más específicos (patrón de "especialización")
   • La variablidad diagnóstica en procedimientos sugiere diversidad clínica en LOS=0

5. IMPLICACIONES CLÍNICAS:
   • Procedimientos comunes (0DB*, 0DJ*) son versátiles para múltiples diagnósticos
   • Diagnósticos específicos pueden predecir procedimientos más probables
   • La matriz permite identificar combinaciones esperadas vs inusuales

════════════════════════════════════════════════════════════════════════════════
Gráficos generados:
  ✓ 04_heatmap_correlaciones_los_0.png - Matriz de co-ocurrencias
  ✓ 05_procedimientos_por_diagnostico_los_0.png - Procedimientos por diagnóstico (TOP 5)
  ✓ 06_diagnosticos_por_procedimiento_los_0.png - Diagnósticos por procedimiento (TOP 5)

CSV generado:
  ✓ correlaciones_diagnostico_procedimiento_los_0.csv - Datos completos de correlaciones
════════════════════════════════════════════════════════════════════════════════
"""

with open(f'{OUTPUT_DIR}/RESUMEN_CORRELACIONES_LOS_0.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Generado: RESUMEN_CORRELACIONES_LOS_0.txt")
print("\n" + summary_text)

print("\n✅ Análisis de correlaciones completado!")
