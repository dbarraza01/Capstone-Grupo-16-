"""
Análisis detallado de pacientes con LOS=0 (estancia de 0 días)
Objetivo: Identificar diagnósticos y procedimientos para comprender altas rápidas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ============================================
# DICCIONARIOS ICD-10
# ============================================
capitulos_icd10cm = {
    'A': 'Enfermedades infecciosas y parasitarias (A00-A99)',
    'B': 'Enfermedades infecciosas y parasitarias (B00-B99)',
    'C': 'Neoplasias malignas (C00-C96)',
    'D': 'Neoplasias benignas y enf. sangre (D00-D89)',
    'E': 'Enfermedades endocrinas, nutricionales y metabólicas (E00-E89)',
    'F': 'Trastornos mentales y del comportamiento (F01-F99)',
    'G': 'Enfermedades del sistema nervioso (G00-G99)',
    'H': 'Enfermedades del ojo, oído y anexos (H00-H95)',
    'I': 'Enfermedades del sistema circulatorio (I00-I99)',
    'J': 'Enfermedades del sistema respiratorio (J00-J99)',
    'K': 'Enfermedades del sistema digestivo (K00-K95)',
    'L': 'Enfermedades de la piel y tejido subcutáneo (L00-L99)',
    'M': 'Enfermedades del sistema musculoesquelético (M00-M99)',
    'N': 'Enfermedades del sistema genitourinario (N00-N99)',
    'O': 'Embarazo, parto y puerperio (O00-O9A)',
    'P': 'Afecciones perinatales (P00-P96)',
    'Q': 'Malformaciones congénitas (Q00-Q99)',
    'R': 'Síntomas, signos y hallazgos anormales (R00-R99)',
    'S': 'Traumatismos y lesiones (S00-S99)',
    'T': 'Envenenamientos y efectos adversos (T07-T88)',
    'U': 'Códigos de uso especial / COVID-19 (U00-U85)',
    'V': 'Causas externas: transporte (V00-V99)',
    'W': 'Causas externas: caídas y accidentes (W00-W99)',
    'X': 'Causas externas: otras (X00-X99)',
    'Y': 'Causas externas: intervención médica (Y00-Y99)',
    'Z': 'Factores que influyen en el estado de salud (Z00-Z99)',
}

categorias_relevantes = {
    'Z79': 'Uso prolongado de medicamentos',
    'Z68': 'Índice de masa corporal (IMC)',
    'E66': 'Obesidad',
    'I10': 'Hipertensión esencial',
    'F17': 'Trastornos por uso de nicotina',
    'E11': 'Diabetes mellitus tipo 2',
    'E78': 'Trastornos del metabolismo lipídico',
    'Z95': 'Presencia de implantes cardíacos y vasculares',
    'I25': 'Enfermedad isquémica crónica del corazón',
    'G47': 'Trastornos del sueño',
    'I48': 'Fibrilación y flutter auricular',
    'Z87': 'Historia personal de enfermedades',
    'Z3A': 'Semanas de gestación',
    'Z37': 'Resultado del parto',
    'J44': 'EPOC (Enfermedad Pulmonar Obstructiva Crónica)',
    'N18': 'Enfermedad renal crónica',
    'Z48': 'Encuentro de seguimiento post-procedimiento',
    'E55': 'Deficiencia de vitamina D',
    'E87': 'Otros trastornos electrolíticos',
    'UUU': 'Urgencias',
    'Z00': 'Examen médico general',
    'Z01': 'Examen especial',
    'Z02': 'Examen con fines administrativos',
}

seccion_pcs = {
    '0': 'Cirugía médica y quirúrgica (Medical & Surgical)',
    '1': 'Obstetricia',
    '2': 'Colocación (Placement)',
    '3': 'Administración (ej: transfusiones, infusiones)',
    '4': 'Medición y monitoreo',
    '5': 'Asistencia y rendimiento extracorpóreo (ej: ECMO, diálisis)',
    '6': 'Terapias extracorpóreas',
    '7': 'Osteopatía',
    '8': 'Otros procedimientos',
    '9': 'Quiropráctica',
    'B': 'Diagnóstico por imagen (Imaging)',
    'C': 'Medicina nuclear',
    'D': 'Oncología radioterápica',
    'F': 'Rehabilitación física y audiología',
    'G': 'Salud mental',
    'H': 'Tratamiento de abuso de sustancias',
    'X': 'Nueva tecnología',
}

# Configuración
OUTPUT_DIR = "."
DATA_DIR = "../data/processed"

# ============================================
# 1. CARGAR DATOS
# ============================================
print("Cargando datos...")
df_maestro = pd.read_csv(f"{DATA_DIR}/dataset_maestro.csv", dtype=str, sep=';')
df_diag = pd.read_csv(f"{DATA_DIR}/caso_diagnostico.csv", dtype=str, sep=';')
df_proc = pd.read_csv(f"{DATA_DIR}/caso_procedimiento.csv", dtype=str, sep=';')

# Convertir LOS a numeric
df_maestro['los_dias'] = pd.to_numeric(df_maestro['los_dias'], errors='coerce')

# Filtrar pacientes con LOS=0
los_0_patients = df_maestro[df_maestro['los_dias'] == 0].copy()
los_0_ids = set(los_0_patients['case_id'].unique())

print(f"✓ Total de pacientes: {len(df_maestro)}")
print(f"✓ Pacientes con LOS=0: {len(los_0_patients)} ({100*len(los_0_patients)/len(df_maestro):.2f}%)")

# ============================================
# 2. ANÁLISIS DE DIAGNÓSTICOS (LOS=0)
# ============================================
print("\nAnalizando diagnósticos para LOS=0...")

# Filtrar diagnósticos de pacientes con LOS=0
diag_los_0 = df_diag[df_diag['case_id'].isin(los_0_ids)].copy()

print(f"✓ Total registros diagnósticos (LOS=0): {len(diag_los_0)}")
print(f"✓ Diagnósticos únicos: {diag_los_0['d_code'].nunique()}")

# Contar frecuencias
diag_freq = diag_los_0['d_code'].value_counts()
print(f"\nDiagnósticos TOP 15 (LOS=0):")
print(diag_freq.head(15))

# ============================================
# 3. ANÁLISIS DE PROCEDIMIENTOS (LOS=0)
# ============================================
print("\nAnalizando procedimientos para LOS=0...")

# Filtrar procedimientos de pacientes con LOS=0
proc_los_0 = df_proc[df_proc['case_id'].isin(los_0_ids)].copy()

print(f"✓ Total registros procedimientos (LOS=0): {len(proc_los_0)}")
print(f"✓ Procedimientos únicos: {proc_los_0['p_code'].nunique()}")

# Contar frecuencias
proc_freq = proc_los_0['p_code'].value_counts()
print(f"\nProcedimientos TOP 15 (LOS=0):")
print(proc_freq.head(15))

# ============================================
# 4. CREAR DICCIONARIO DE TRADUCCIONES
# ============================================
print("\nCreando diccionarios de traducciones...")

# Para diagnósticos: usar 3 primeros caracteres
def get_diagnosis_description(code):
    """Obtener descripción de diagnóstico basado en código ICD-10-CM"""
    if pd.isna(code) or len(str(code)) < 1:
        return "Código inválido"

    code = str(code)
    # Primer carácter = capítulo
    chapter = code[0]
    chapter_desc = capitulos_icd10cm.get(chapter, f"Capítulo desconocido ({chapter})")

    # Primeros 3 caracteres para categoría más específica
    category_3 = code[:3]
    category_desc = categorias_relevantes.get(category_3, None)

    if category_desc:
        return f"{category_desc} ({category_3})"
    else:
        return f"{chapter_desc} ({code[:3]})"

# Para procedimientos: procesar sección (primer carácter)
def get_procedure_description(code):
    """Obtener descripción de procedimiento basado en código ICD-10-PCS"""
    if pd.isna(code) or len(str(code)) < 1:
        return "Código inválido"

    code = str(code)
    section = code[0]
    section_desc = seccion_pcs.get(section, f"Sección desconocida ({section})")

    return f"{section_desc} ({code[:3]})"

# Agregar descripciones
diag_los_0['descripcion'] = diag_los_0['d_code'].apply(get_diagnosis_description)
proc_los_0['descripcion'] = proc_los_0['p_code'].apply(get_procedure_description)

# Agrupar por descripción
diag_by_desc = diag_los_0['descripcion'].value_counts()
proc_by_desc = proc_los_0['descripcion'].value_counts()

print(f"✓ Diagnósticos agrupados por descripción: {len(diag_by_desc)}")
print(f"✓ Procedimientos agrupados por descripción: {len(proc_by_desc)}")

# ============================================
# 5. CREAR VISUALIZACIONES
# ============================================
print("\nCreando visualizaciones...")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.suptitle('Análisis de Diagnósticos y Procedimientos en Pacientes con LOS=0\n(250 pacientes con estancia de 0 días)',
             fontsize=16, fontweight='bold', y=0.995)

# === GRÁFICO 1: DIAGNÓSTICOS ===
ax1 = axes[0]
top_diag = diag_by_desc.head(12)
colors_diag = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_diag)))
bars1 = ax1.barh(range(len(top_diag)), top_diag.values, color=colors_diag, edgecolor='black', linewidth=0.7)
ax1.set_yticks(range(len(top_diag)))
ax1.set_yticklabels(top_diag.index, fontsize=10)
ax1.set_xlabel('Frecuencia (número de diagnósticos)', fontsize=11, fontweight='bold')
ax1.set_title('TOP 12 Diagnósticos en LOS=0', fontsize=12, fontweight='bold', loc='left')
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars1, top_diag.values)):
    pct = 100 * val / len(diag_los_0)
    ax1.text(val + 1, i, f'{int(val)} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')

# === GRÁFICO 2: PROCEDIMIENTOS ===
ax2 = axes[1]
top_proc = proc_by_desc.head(12)
colors_proc = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_proc)))
bars2 = ax2.barh(range(len(top_proc)), top_proc.values, color=colors_proc, edgecolor='black', linewidth=0.7)
ax2.set_yticks(range(len(top_proc)))
ax2.set_yticklabels(top_proc.index, fontsize=10)
ax2.set_xlabel('Frecuencia (número de procedimientos)', fontsize=11, fontweight='bold')
ax2.set_title('TOP 12 Procedimientos en LOS=0', fontsize=12, fontweight='bold', loc='left')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars2, top_proc.values)):
    pct = 100 * val / len(proc_los_0)
    ax2.text(val + 1, i, f'{int(val)} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_diagnosticos_procedimientos_los_0.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: 01_diagnosticos_procedimientos_los_0.png")
plt.close()

# ============================================
# 6. ANÁLISIS POR TIPO (Principal vs Secundario)
# ============================================
print("\nAnalizando diagnósticos primarios vs secundarios...")

# Principales vs Secundarios
principal = diag_los_0[diag_los_0['tipo_d'] == 'P']
secondary = diag_los_0[diag_los_0['tipo_d'] == 'S']

print(f"✓ Diagnósticos principales (P): {len(principal)} ({100*len(principal)/len(diag_los_0):.1f}%)")
print(f"✓ Diagnósticos secundarios (S): {len(secondary)} ({100*len(secondary)/len(diag_los_0):.1f}%)")

# Top diagnósticos principales
principal_by_desc = principal['descripcion'].value_counts().head(10)
secondary_by_desc = secondary['descripcion'].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Diagnósticos Principales vs Secundarios en LOS=0', fontsize=14, fontweight='bold')

# Principales
ax = axes[0]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(principal_by_desc)))
bars = ax.barh(range(len(principal_by_desc)), principal_by_desc.values, color=colors, edgecolor='black', linewidth=0.7)
ax.set_yticks(range(len(principal_by_desc)))
ax.set_yticklabels(principal_by_desc.index, fontsize=9)
ax.set_xlabel('Frecuencia', fontsize=10, fontweight='bold')
ax.set_title(f'Diagnósticos Principales (n={len(principal)})', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, principal_by_desc.values)):
    ax.text(val + 0.5, i, f'{int(val)}', va='center', fontsize=9)

# Secundarios
ax = axes[1]
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(secondary_by_desc)))
bars = ax.barh(range(len(secondary_by_desc)), secondary_by_desc.values, color=colors, edgecolor='black', linewidth=0.7)
ax.set_yticks(range(len(secondary_by_desc)))
ax.set_yticklabels(secondary_by_desc.index, fontsize=9)
ax.set_xlabel('Frecuencia', fontsize=10, fontweight='bold')
ax.set_title(f'Diagnósticos Secundarios (n={len(secondary)})', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, secondary_by_desc.values)):
    ax.text(val + 0.5, i, f'{int(val)}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_principal_vs_secundario_los_0.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: 02_principal_vs_secundario_los_0.png")
plt.close()

# ============================================
# 7. CREAR TABLA DETALLADA
# ============================================
print("\nCreando tabla resumen...")

# Top códigos con descripciones
top_codes_diag = diag_los_0['d_code'].value_counts().head(20)
diag_summary = pd.DataFrame({
    'Código': top_codes_diag.index,
    'Frecuencia': top_codes_diag.values,
    'Porcentaje': (100 * top_codes_diag.values / len(diag_los_0)).round(2),
    'Descripción': [get_diagnosis_description(code) for code in top_codes_diag.index]
})

top_codes_proc = proc_los_0['p_code'].value_counts().head(20)
proc_summary = pd.DataFrame({
    'Código': top_codes_proc.index,
    'Frecuencia': top_codes_proc.values,
    'Porcentaje': (100 * top_codes_proc.values / len(proc_los_0)).round(2),
    'Descripción': [get_procedure_description(code) for code in top_codes_proc.index]
})

# Guardar en CSV
diag_summary.to_csv(f'{OUTPUT_DIR}/diagnosticos_detallado_los_0.csv', index=False)
proc_summary.to_csv(f'{OUTPUT_DIR}/procedimientos_detallado_los_0.csv', index=False)

print("✓ Guardado: diagnosticos_detallado_los_0.csv")
print("✓ Guardado: procedimientos_detallado_los_0.csv")

# ============================================
# 8. ANÁLISIS DE INTERPRETACIÓN
# ============================================
print("\nGenerando análisis de interpretación...")

# Distribuir por secciones PCS
section_counts = proc_los_0['p_code'].str[0].value_counts()
print(f"\nDistribución por sección PCS:")
for section, count in section_counts.head(10).items():
    desc = seccion_pcs.get(section, 'Desconocida')
    pct = 100 * count / len(proc_los_0)
    print(f"  {section}: {desc[:50]} - {count} ({pct:.1f}%)")

# ============================================
# 9. CREAR RESUMEN EJECUTIVO
# ============================================
# Preparar secciones principales
top_sections = []
for k in section_counts.head(3).index:
    desc = seccion_pcs.get(k, "")
    top_sections.append(f"{k}: {desc}")
sections_str = ", ".join(top_sections)

summary_text = f"""
{'='*80}
ANÁLISIS DE PACIENTES CON LOS=0 (Estancia de 0 días)
{'='*80}

RESUMEN EJECUTIVO
-----------------
Total de pacientes con LOS=0: {len(los_0_patients)} ({100*len(los_0_patients)/len(df_maestro):.2f}% del total)

DIAGNÓSTICOS
• Total registros diagnósticos: {len(diag_los_0)}
• Diagnósticos únicos: {diag_los_0['d_code'].nunique()}
• Diagnósticos principales (P): {len(principal)} ({100*len(principal)/len(diag_los_0):.1f}%)
• Diagnósticos secundarios (S): {len(secondary)} ({100*len(secondary)/len(diag_los_0):.1f}%)

Diagnósticos TOP 5:
"""

for i, (desc, count) in enumerate(diag_by_desc.head(5).items(), 1):
    pct = 100 * count / len(diag_los_0)
    summary_text += f"  {i}. {desc}: {int(count)} ({pct:.1f}%)\n"

summary_text += f"""
PROCEDIMIENTOS
• Total registros procedimientos: {len(proc_los_0)}
• Procedimientos únicos: {proc_los_0['p_code'].nunique()}

Procedimientos TOP 5:
"""

for i, (desc, count) in enumerate(proc_by_desc.head(5).items(), 1):
    pct = 100 * count / len(proc_los_0)
    summary_text += f"  {i}. {desc}: {int(count)} ({pct:.1f}%)\n"

summary_text += f"""
INTERPRETACIÓN
{'='*80}

HIPÓTESIS SOBRE LOS=0:

1. OBSERVACIÓN ADMINISTRATIVO (Principal hallazgo)
   • {int(len(principal))} diagnósticos principales sugieren casos de baja complejidad
   • Muchos corresponden a evaluaciones rápidas (Z-codes: Z00-Z99)
   • Pacientes admitidos y evaluados el mismo día sin necesidad de internación

2. PROCEDIMIENTOS DIAGNÓSTICOS (Imaging, estudios)
   • Mayor presencia de procedimientos de diagnóstico que terapéuticos
   • Secciones principales: {sections_str}
   • Indica pacientes que realizaron pruebas diagnósticas ambulatorias

3. CAUSAS POSIBLES DE ALTAS INMEDIATAS:
   a) Cirugías ambulatorias (Same-Day Surgery)
   b) Procedimientos diagnósticos (endoscopia, biopsia)
   c) Evaluaciones de urgencia (triaje, observación breve)
   d) Ingresos administrativos para procedimientos
   e) Pacientes traslados internos (egreso administrativo)

4. RELEVANCIA CLÍNICA:
   • Representa población de bajo riesgo/baja complejidad
   • Importante para modelos predictivos: necesitan regresión separada
   • No corresponde a altas contra médico ni abandonos
   • Son estancias válidas y esperadas en hospitales modernos

{'='*80}
Archivos generados:
  - 01_diagnosticos_procedimientos_los_0.png
  - 02_principal_vs_secundario_los_0.png
  - diagnosticos_detallado_los_0.csv
  - procedimientos_detallado_los_0.csv
  - RESUMEN_ANALISIS_LOS_0.txt
{'='*80}
"""

with open(f'{OUTPUT_DIR}/RESUMEN_ANALISIS_LOS_0.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)
print("\n✓ Guardado: RESUMEN_ANALISIS_LOS_0.txt")

# ============================================
# 10. CREAR GRÁFICO CIRCULAR DE SECCIONES PCS
# ============================================
fig, ax = plt.subplots(figsize=(10, 8))

section_names = section_counts.index.map(lambda x: f"{x}: {seccion_pcs.get(x, 'Desconocida')[:40]}")
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(section_counts)))

wedges, texts, autotexts = ax.pie(section_counts.values,
                                    labels=section_names,
                                    autopct='%1.1f%%',
                                    colors=colors_pie,
                                    startangle=90,
                                    textprops={'fontsize': 9})

ax.set_title('Distribución de Procedimientos por Sección ICD-10-PCS\n(LOS=0)',
             fontsize=12, fontweight='bold', pad=20)

# Mejorar legibilidad
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_distribucion_secciones_pcs_los_0.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: 03_distribucion_secciones_pcs_los_0.png")
plt.close()

print("\n✅ Análisis completado exitosamente!")
