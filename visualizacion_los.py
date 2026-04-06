"""
Visualización de la distribución de LOS (Length of Stay) hospitalario
Genera gráficas descriptivas con estadísticas completas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Cargar datos
print("Cargando dataset maestro...")
df = pd.read_csv('data/processed/dataset_maestro.csv', sep=';', dtype=str)
df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')

# Eliminar NaN si los hay
los = df['los_dias'].dropna()

print(f"\n{'='*60}")
print("ESTADÍSTICAS DESCRIPTIVAS COMPLETAS - LOS (días)")
print(f"{'='*60}")

# Estadísticas básicas
print(f"\n{'Medidas de tendencia central':^60}")
print(f"{'─'*60}")
print(f"Media:                    {los.mean():.2f} días")
print(f"Mediana (P50):            {los.median():.2f} días")
print(f"Moda:                     {los.mode()[0]:.0f} días")

print(f"\n{'Medidas de dispersión':^60}")
print(f"{'─'*60}")
print(f"Desviación estándar:      {los.std():.2f} días")
print(f"Varianza:                 {los.var():.2f}")
print(f"Mínimo:                   {los.min():.0f} días")
print(f"Máximo:                   {los.max():.0f} días")
print(f"Rango:                    {los.max() - los.min():.0f} días")
print(f"IQR (Q3-Q1):              {los.quantile(0.75) - los.quantile(0.25):.2f} días")

print(f"\n{'Medidas de forma':^60}")
print(f"{'─'*60}")
skew = stats.skew(los)
kurt = stats.kurtosis(los)
print(f"Skewness (asimetría):     {skew:.4f}")
print(f"Kurtosis (curtosis):      {kurt:.4f}")

if skew > 1:
    print(f"  → Distribución MUY sesgada a la derecha")
elif skew > 0.5:
    print(f"  → Distribución moderadamente sesgada a la derecha")

if kurt > 3:
    print(f"  → Distribución leptocúrtica (colas pesadas, pico pronunciado)")

print(f"\n{'Percentiles clave':^60}")
print(f"{'─'*60}")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    valor = los.quantile(p/100)
    print(f"P{p:2d}:                      {valor:.2f} días")

print(f"\n{'Información adicional':^60}")
print(f"{'─'*60}")
print(f"N (pacientes):            {len(los):,}")
print(f"Casos LOS = 0:            {(los == 0).sum():,} ({(los == 0).sum()/len(los)*100:.2f}%)")
print(f"Casos LOS > 30 días:      {(los > 30).sum():,} ({(los > 30).sum()/len(los)*100:.2f}%)")
print(f"Casos LOS > 90 días:      {(los > 90).sum():,} ({(los > 90).sum()/len(los)*100:.2f}%)")
print(f"{'='*60}\n")

# ============================================================================
# GRÁFICA 1: Histograma en escala lineal
# ============================================================================
print("Generando gráfica 1: Histograma en escala lineal...")

fig, ax = plt.subplots(figsize=(14, 8))

# Histograma con bins automáticos optimizados
n_bins = min(100, int(np.sqrt(len(los))))
counts, bins, patches = ax.hist(los, bins=n_bins,
                                    color='steelblue',
                                    alpha=0.7,
                                    edgecolor='black',
                                    linewidth=0.5)

# Líneas de media y mediana
media = los.mean()
mediana = los.median()

ax.axvline(media, color='red', linestyle='--', linewidth=2.5,
            label=f'Media = {media:.2f} días', alpha=0.8)
ax.axvline(mediana, color='green', linestyle='--', linewidth=2.5,
            label=f'Mediana = {mediana:.2f} días', alpha=0.8)

# Títulos y etiquetas
ax.set_xlabel('Días de Estancia Hospitalaria (LOS)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frecuencia (Número de Pacientes)', fontsize=13, fontweight='bold')
ax.set_title('Distribución de Length of Stay (LOS) - Escala Lineal\n' +
                f'Skewness = {skew:.3f} | Kurtosis = {kurt:.3f} | N = {len(los):,}',
                fontsize=15, fontweight='bold', pad=20)

# Leyenda con estadísticas
legend_text = (
    f'Media = {media:.2f} días\n'
    f'Mediana = {mediana:.2f} días\n'
    f'Desv. Est. = {los.std():.2f} días\n'
    f'Mín = {los.min():.0f} | Máx = {los.max():.0f}\n'
    f'P95 = {los.quantile(0.95):.2f} días'
)

ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Añadir cuadro de texto con estadísticas
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.65, legend_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# Grid y ajustes
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.set_xlim(left=0)
plt.tight_layout()

# Guardar
output_path_1 = 'graficos/01_distribucion_los_escala_lineal.png'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: {output_path_1}")
plt.close()

# ============================================================================
# GRÁFICA 2: Histograma con transformación logarítmica
# ============================================================================
print("Generando gráfica 2: Histograma con transformación log(1+x)...")

fig, ax = plt.subplots(figsize=(14, 8))

# Aplicar transformación log(1+x)
los_log = np.log1p(los)

# Histograma en escala log
n_bins_log = min(80, int(np.sqrt(len(los))))
counts_log, bins_log, patches_log = ax.hist(los_log, bins=n_bins_log,
                                                color='darkorange',
                                                alpha=0.7,
                                                edgecolor='black',
                                                linewidth=0.5)

# Líneas de media y mediana (transformadas)
media_log = np.log1p(media)
mediana_log = np.log1p(mediana)

ax.axvline(media_log, color='red', linestyle='--', linewidth=2.5,
            label=f'Media = {media:.2f} días (log = {media_log:.2f})', alpha=0.8)
ax.axvline(mediana_log, color='green', linestyle='--', linewidth=2.5,
            label=f'Mediana = {mediana:.2f} días (log = {mediana_log:.2f})', alpha=0.8)

# Añadir líneas de percentiles clave
p95_val = los.quantile(0.95)
p99_val = los.quantile(0.99)
ax.axvline(np.log1p(p95_val), color='purple', linestyle=':', linewidth=2,
            label=f'P95 = {p95_val:.2f} días', alpha=0.7)
ax.axvline(np.log1p(p99_val), color='brown', linestyle=':', linewidth=2,
            label=f'P99 = {p99_val:.2f} días', alpha=0.7)

# Títulos y etiquetas
ax.set_xlabel('log(1 + LOS) - Días de Estancia Transformados', fontsize=13, fontweight='bold')
ax.set_ylabel('Frecuencia (Número de Pacientes)', fontsize=13, fontweight='bold')
ax.set_title('Distribución de LOS con Transformación Logarítmica log(1+x)\n' +
                'Revela la estructura completa de la distribución sesgada',
                fontsize=15, fontweight='bold', pad=20)

# Leyenda
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

# Añadir etiquetas en el eje X con valores originales
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())

# Crear ticks con valores representativos
valores_originales = [0, 1, 3, 7, 14, 30, 60, 90, 180, 262]
valores_log = [np.log1p(v) for v in valores_originales]
ax2.set_xticks(valores_log)
ax2.set_xticklabels([f'{v}' for v in valores_originales], fontsize=9)
ax2.set_xlabel('Valores Originales de LOS (días)', fontsize=11, fontweight='bold', labelpad=10)

# Cuadro de texto explicativo
explanation_text = (
    'Transformación log(1+x):\n'
    '• Expande valores pequeños\n'
    '• Comprime valores grandes\n'
    '• Revela distribución completa\n'
    f'• Sesgo reducido en escala log'
)
props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props2)

# Grid y ajustes
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
plt.tight_layout()

# Guardar
output_path_2 = 'graficos/02_distribucion_los_transformacion_logaritmica.png'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: {output_path_2}")
plt.close()

# ============================================================================
# GRÁFICA 3 (BONUS): Box plot con percentiles
# ============================================================================
print("Generando gráfica 3 (bonus): Box plot con percentiles...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Box plot en escala lineal
bp1 = ax1.boxplot([los], vert=False, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

# Añadir puntos de percentiles
percentiles_vals = [los.quantile(p/100) for p in [10, 25, 50, 75, 90, 95, 99]]
colors_p = ['blue', 'green', 'red', 'green', 'blue', 'purple', 'brown']
labels_p = ['P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']

for val, color, label in zip(percentiles_vals, colors_p, labels_p):
    ax1.axvline(val, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.text(val, 1.15, f'{label}\n{val:.1f}d', ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold')

ax1.set_xlabel('Días de Estancia (LOS)', fontsize=12, fontweight='bold')
ax1.set_title('Box Plot - Escala Lineal\nVisualizando Outliers y Percentiles',
                fontsize=13, fontweight='bold')
ax1.set_yticks([])
ax1.grid(True, alpha=0.3, axis='x')

# Box plot en escala log
bp2 = ax2.boxplot([los_log], vert=False, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

ax2.set_xlabel('log(1 + LOS)', fontsize=12, fontweight='bold')
ax2.set_title('Box Plot - Escala Logarítmica\nDistribución Normalizada',
                fontsize=13, fontweight='bold')
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# Axis secundario con valores originales
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
valores_originales_box = [0, 1, 3, 7, 14, 30, 90, 262]
valores_log_box = [np.log1p(v) for v in valores_originales_box]
ax2_top.set_xticks(valores_log_box)
ax2_top.set_xticklabels([f'{v}d' for v in valores_originales_box], fontsize=8)

plt.suptitle(f'Análisis de Percentiles - LOS (N = {len(los):,} pacientes)',
                fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

output_path_3 = 'graficos/03_boxplot_percentiles_los.png'
plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: {output_path_3}")
plt.close()

print("\n" + "="*60)
print("VISUALIZACIÓN COMPLETADA")
print("="*60)
print(f"Se generaron 3 gráficas en la carpeta 'graficos/':")
print(f"  1. {output_path_1}")
print(f"  2. {output_path_2}")
print(f"  3. {output_path_3}")
print("="*60)
