"""
Análisis de Complejidad vs LOS:
- Gráficos de boxplot mostrando cómo LOS varía según:
    1. Número total de diagnósticos
    2. Número de diagnósticos secundarios
    3. Número de procedimientos

Hipótesis: Mayor complejidad acumulada → Mayor LOS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
DATA_PATH = '../../data/processed/dataset_maestro.csv'
OUTPUT_DIAG = 'diagnosticos/'
OUTPUT_PROC = 'procedimientos/'

# Crear directorios
Path(OUTPUT_DIAG).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PROC).mkdir(parents=True, exist_ok=True)

# Cargar datos
print("📊 Cargando dataset maestro...")
df = pd.read_csv(DATA_PATH, sep=';', dtype=str)
df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')

# Contar diagnósticos y procedimientos
print("🔢 Calculando complejidad...")

# Contar diagnósticos totales
df['n_diagnosticos_total'] = df.apply(
    lambda row: len([c.strip() for col in ['diagnosticos_primarios', 'diagnosticos_secundarios']
                        for c in str(row[col]).split(',') if c.strip()]),
    axis=1
)

# Contar diagnósticos secundarios
df['n_diagnosticos_secundarios'] = df.apply(
    lambda row: len([c.strip() for c in str(row['diagnosticos_secundarios']).split(',') if c.strip()]),
    axis=1
) if 'diagnosticos_secundarios' in df.columns else 0

# Contar procedimientos
df['n_procedimientos'] = df.apply(
    lambda row: len([c.strip() for c in str(row['procedimientos']).split(',') if c.strip()]) if pd.notna(row['procedimientos']) and row['procedimientos'] != '' else 0,
    axis=1
)

print(f"✅ Datos procesados")
print(f"   n_diagnosticos_total: {df['n_diagnosticos_total'].min()}-{df['n_diagnosticos_total'].max()}")
print(f"   n_diagnosticos_secundarios: {df['n_diagnosticos_secundarios'].min()}-{df['n_diagnosticos_secundarios'].max()}")
print(f"   n_procedimientos: {df['n_procedimientos'].min()}-{df['n_procedimientos'].max()}")

# Función para crear tramos
def crear_tramos(serie, nombre_tramo):
    """Crear categorías: 1, 2-3, 4-5, 6+"""
    tramos = pd.cut(serie, bins=[0, 1, 3, 5, np.inf],
                    labels=['1', '2-3', '4-5', '6+'],
                    right=True)
    return tramos

# Crear tramos para cada variable
df['tramo_diag_total'] = crear_tramos(df['n_diagnosticos_total'], 'n_diagnosticos_total')
df['tramo_diag_sec'] = crear_tramos(df['n_diagnosticos_secundarios'], 'n_diagnosticos_secundarios')
df['tramo_proc'] = crear_tramos(df['n_procedimientos'], 'n_procedimientos')

# ========== GRÁFICOS DE DIAGNÓSTICOS ==========

print("\n🎨 Generando gráficos de diagnósticos...")

# GRÁFICO 1: LOS vs Número Total de Diagnósticos
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

sns.boxplot(
    data=df,
    x='tramo_diag_total',
    y='los_dias',
    ax=ax,
    palette='Blues',
    showfliers=True
)

ax.set_xlabel('Número Total de Diagnósticos', fontsize=12, fontweight='bold')
ax.set_ylabel('LOS (días)', fontsize=12, fontweight='bold')
ax.set_title('Relación entre Complejidad Diagnóstica Total y LOS',
                fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, df['los_dias'].quantile(0.95))
ax.grid(axis='y', alpha=0.3)

# Añadir estadísticas
for i, tramo in enumerate(['1', '2-3', '4-5', '6+']):
    subset = df[df['tramo_diag_total'] == tramo]['los_dias']
    mediana = subset.median()
    ax.text(i, mediana + 1, f'n={len(subset)}\nMed={mediana:.0f}',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIAG}05_los_vs_diagnosticos_total.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ 05_los_vs_diagnosticos_total.png")

# GRÁFICO 2: LOS vs Número de Diagnósticos Secundarios
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

sns.boxplot(
    data=df,
    x='tramo_diag_sec',
    y='los_dias',
    ax=ax,
    palette='Oranges',
    showfliers=True
)

ax.set_xlabel('Número de Diagnósticos Secundarios', fontsize=12, fontweight='bold')
ax.set_ylabel('LOS (días)', fontsize=12, fontweight='bold')
ax.set_title('Relación entre Diagnósticos Secundarios y LOS',
                fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, df['los_dias'].quantile(0.95))
ax.grid(axis='y', alpha=0.3)

# Añadir estadísticas
for i, tramo in enumerate(['1', '2-3', '4-5', '6+']):
    subset = df[df['tramo_diag_sec'] == tramo]['los_dias']
    mediana = subset.median()
    ax.text(i, mediana + 1, f'n={len(subset)}\nMed={mediana:.0f}',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIAG}06_los_vs_diagnosticos_secundarios.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ 06_los_vs_diagnosticos_secundarios.png")

# ========== GRÁFICO DE PROCEDIMIENTOS ==========

print("\n🎨 Generando gráficos de procedimientos...")

# GRÁFICO 3: LOS vs Número de Procedimientos
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

sns.boxplot(
    data=df,
    x='tramo_proc',
    y='los_dias',
    ax=ax,
    palette='Greens',
    showfliers=True
)

ax.set_xlabel('Número de Procedimientos', fontsize=12, fontweight='bold')
ax.set_ylabel('LOS (días)', fontsize=12, fontweight='bold')
ax.set_title('Relación entre Número de Procedimientos y LOS',
                fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, df['los_dias'].quantile(0.95))
ax.grid(axis='y', alpha=0.3)

# Añadir estadísticas
for i, tramo in enumerate(['1', '2-3', '4-5', '6+']):
    subset = df[df['tramo_proc'] == tramo]['los_dias']
    mediana = subset.median()
    ax.text(i, mediana + 1, f'n={len(subset)}\nMed={mediana:.0f}',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_PROC}05_los_vs_procedimientos.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ 05_los_vs_procedimientos.png")

# Imprimir resumen estadístico
print("\n📊 RESUMEN ESTADÍSTICO:")
print("\n" + "="*70)
print("LOS por Número Total de Diagnósticos")
print("="*70)
for tramo in ['1', '2-3', '4-5', '6+']:
    subset = df[df['tramo_diag_total'] == tramo]['los_dias']
    print(f"{tramo:>4} diagnósticos: n={len(subset):5d} | Med={subset.median():5.1f} | Media={subset.mean():5.2f} | Max={subset.max():5.0f}")

print("\n" + "="*70)
print("LOS por Número de Diagnósticos Secundarios")
print("="*70)
for tramo in ['1', '2-3', '4-5', '6+']:
    subset = df[df['tramo_diag_sec'] == tramo]['los_dias']
    print(f"{tramo:>4} secundarios: n={len(subset):5d} | Med={subset.median():5.1f} | Media={subset.mean():5.2f} | Max={subset.max():5.0f}")

print("\n" + "="*70)
print("LOS por Número de Procedimientos")
print("="*70)
for tramo in ['1', '2-3', '4-5', '6+']:
    subset = df[df['tramo_proc'] == tramo]['los_dias']
    print(f"{tramo:>4} procedimientos: n={len(subset):5d} | Med={subset.median():5.1f} | Media={subset.mean():5.2f} | Max={subset.max():5.0f}")

print("\n✅ Análisis de complejidad completado!")
print(f"\n📁 Resultados guardados en:")
print(f"   - {OUTPUT_DIAG}")
print(f"   - {OUTPUT_PROC}")
