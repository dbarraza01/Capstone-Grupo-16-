"""
Análisis de códigos ICD-10 y su relación con LOS (Length of Stay)
- Identifica códigos asociados a outliers (estancias largas)
- Genera gráficos de densidad de códigos frecuentes
- Crea conclusiones automáticas en archivos .md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = 'data/processed/dataset_maestro.csv'
OUTPUT_DIAG = 'graficos/diagnosticos/'
OUTPUT_PROC = 'graficos/procedimientos/'

# ========================
# FUNCIONES DE ANÁLISIS
# ========================

def cargar_datos():
    """Carga el dataset maestro."""
    df = pd.read_csv(DATA_PATH, sep=';', dtype=str)
    df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')
    return df

def explotar_codigos(df, columnas, tipo):
    """
    Explota las listas de códigos en filas individuales.

    Args:
        df: DataFrame con dataset maestro
        columnas: lista de columnas con códigos (ej: ['diagnosticos_primarios', 'diagnosticos_secundarios'])
        tipo: 'diagnostico' o 'procedimiento'

    Returns:
        DataFrame con una fila por cada código-paciente
    """
    registros = []

    for _, row in df.iterrows():
        case_id = row['case_id']
        los = row['los_dias']
        es_urgencia = row['es_urgencia']

        codigos_set = set()

        for col in columnas:
            valor = row[col]
            if pd.notna(valor) and valor != '':
                codigos = [c.strip() for c in str(valor).split(',') if c.strip()]
                codigos_set.update(codigos)

        for codigo in codigos_set:
            registros.append({
                'case_id': case_id,
                'codigo': codigo,
                'los_dias': los,
                'es_urgencia': es_urgencia,
                'tipo': tipo
            })

    return pd.DataFrame(registros)

def identificar_outliers_iqr(df, columna='los_dias'):
    """
    Identifica outliers usando método IQR (Interquartile Range).

    Returns:
        umbral_outlier: valor de LOS a partir del cual se considera outlier
    """
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    umbral_outlier = Q3 + 1.5 * IQR
    return umbral_outlier

def analizar_codigos_outliers(df_codigos, umbral_outlier, top_n=20):
    """
    Analiza qué códigos están más asociados con outliers.

    Returns:
        DataFrame con estadísticas de códigos y probabilidad de outlier
    """
    stats = df_codigos.groupby('codigo').agg({
        'los_dias': ['count', 'mean', 'median', 'std', 'max'],
        'case_id': 'nunique'
    }).reset_index()

    stats.columns = ['codigo', 'n_registros', 'los_mean', 'los_median', 'los_std', 'los_max', 'n_pacientes']

    # Contar cuántas veces el código aparece en outliers
    df_outliers = df_codigos[df_codigos['los_dias'] >= umbral_outlier]
    outliers_count = df_outliers.groupby('codigo').size().reset_index(name='n_outliers')

    stats = stats.merge(outliers_count, on='codigo', how='left')
    stats['n_outliers'] = stats['n_outliers'].fillna(0).astype(int)
    stats['prob_outlier'] = stats['n_outliers'] / stats['n_registros']

    # Filtrar códigos con al menos 10 apariciones para análisis confiable
    stats = stats[stats['n_registros'] >= 10]

    # Ordenar por probabilidad de outlier
    stats_sorted = stats.sort_values('prob_outlier', ascending=False)

    return stats_sorted.head(top_n)

def graficar_top_codigos_outliers(stats, tipo, umbral_outlier):
    """
    Gráfico de barras: códigos con mayor probabilidad de generar outliers.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    codes = stats['codigo'].values[::-1]
    probs = stats['prob_outlier'].values[::-1] * 100

    bars = ax.barh(range(len(codes)), probs, color='crimson', alpha=0.7)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes)
    ax.set_xlabel('Probabilidad de Outlier (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Código {tipo.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Top 20 Códigos {tipo.capitalize()} Asociados a LOS Outliers (≥{umbral_outlier:.1f} días)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(axis='x', alpha=0.3)

    # Añadir valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig

def graficar_boxplot_codigos(df_codigos, stats, tipo):
    """
    Boxplot: distribución de LOS para top códigos por probabilidad de outlier.
    """
    top_codes = stats['codigo'].head(15).values
    df_top = df_codigos[df_codigos['codigo'].isin(top_codes)].copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Ordenar por mediana de LOS
    order = df_top.groupby('codigo')['los_dias'].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df_top,
        y='codigo',
        x='los_dias',
        order=order,
        ax=ax,
        palette='Set2',
        showfliers=True
    )

    ax.set_xlabel('LOS (días)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Código {tipo.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Distribución de LOS por Código {tipo.capitalize()} (Top 15 Outliers)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig

def graficar_densidad_frecuencia(df_codigos, tipo, top_n=20):
    """
    Gráfico de barras: códigos más frecuentes (densidad).
    """
    freq = df_codigos['codigo'].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))

    codes = freq.index.values[::-1]
    counts = freq.values[::-1]

    bars = ax.barh(range(len(codes)), counts, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes)
    ax.set_xlabel('Número de Pacientes', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Código {tipo.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Top {top_n} Códigos {tipo.capitalize()} Más Frecuentes',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(axis='x', alpha=0.3)

    # Añadir valores en las barras
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + max(counts)*0.01, i, f'{count:,}', va='center', fontsize=9)

    plt.tight_layout()
    return fig

def graficar_violin_top_frecuentes(df_codigos, tipo, top_n=10):
    """
    Violin plot: distribución de LOS para códigos más frecuentes.
    """
    top_codes = df_codigos['codigo'].value_counts().head(top_n).index
    df_top = df_codigos[df_codigos['codigo'].isin(top_codes)].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Ordenar por frecuencia
    order = df_top['codigo'].value_counts().index

    sns.violinplot(
        data=df_top,
        y='codigo',
        x='los_dias',
        order=order,
        ax=ax,
        palette='muted',
        cut=0
    )

    ax.set_xlabel('LOS (días)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Código {tipo.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Distribución de LOS - Top {top_n} Códigos {tipo.capitalize()} Frecuentes',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, df_codigos['los_dias'].quantile(0.95))

    plt.tight_layout()
    return fig

def generar_conclusiones_md(stats_outliers, freq_top, df_codigos, tipo, umbral_outlier, output_path):
    """
    Genera archivo .md con conclusiones del análisis.
    """
    total_pacientes = df_codigos['case_id'].nunique()
    total_registros = len(df_codigos)
    los_mean = df_codigos['los_dias'].mean()
    los_median = df_codigos['los_dias'].median()
    n_outliers = len(df_codigos[df_codigos['los_dias'] >= umbral_outlier])
    pct_outliers = (n_outliers / len(df_codigos)) * 100

    # Top 5 códigos outliers
    top5_outliers = stats_outliers.head(5)

    # Top 5 códigos frecuentes
    top5_freq = freq_top.head(5)

    md_content = f"""# Análisis de Códigos {tipo.capitalize()} y LOS
**Fecha de análisis:** 2026-03-30

---

## 📊 Resumen Ejecutivo

### Estadísticas Generales
- **Total de pacientes analizados:** {total_pacientes:,}
- **Total de registros código-paciente:** {total_registros:,}
- **LOS promedio:** {los_mean:.2f} días
- **LOS mediana:** {los_median:.1f} días
- **Umbral outlier (Q3 + 1.5*IQR):** {umbral_outlier:.1f} días
- **Registros outlier:** {n_outliers:,} ({pct_outliers:.1f}%)

---

## 🔴 Códigos Asociados a Outliers (Estancias Largas)

### Top 5 Códigos con Mayor Probabilidad de Generar LOS Outliers

| Código | Prob. Outlier | N° Pacientes | LOS Promedio | LOS Máximo |
|--------|---------------|--------------|--------------|------------|
"""

    for _, row in top5_outliers.iterrows():
        md_content += f"| **{row['codigo']}** | {row['prob_outlier']*100:.1f}% | {row['n_pacientes']:,} | {row['los_mean']:.1f} días | {row['los_max']:.0f} días |\n"

    md_content += f"""
### Interpretación de Outliers
Los códigos listados arriba tienen una **alta probabilidad de estar asociados con estancias hospitalarias prolongadas** (≥{umbral_outlier:.1f} días). Esto puede indicar:

1. **Complejidad clínica:** Condiciones que requieren tratamiento extenso o monitoreo prolongado
2. **Complicaciones:** Diagnósticos/procedimientos con alta tasa de complicaciones postoperatorias
3. **Comorbilidades:** Pacientes con múltiples condiciones que alargan la recuperación

**Conclusión clave:** El código **{top5_outliers.iloc[0]['codigo']}** tiene la mayor probabilidad de outlier ({top5_outliers.iloc[0]['prob_outlier']*100:.1f}%), apareciendo en {top5_outliers.iloc[0]['n_pacientes']:,} pacientes con LOS promedio de {top5_outliers.iloc[0]['los_mean']:.1f} días.

---

## 📈 Códigos Más Frecuentes (Densidad)

### Top 5 Códigos con Mayor Frecuencia

| Código | N° Pacientes | % del Total |
|--------|--------------|-------------|
"""

    for code, count in top5_freq.items():
        pct = (count / total_pacientes) * 100
        md_content += f"| **{code}** | {count:,} | {pct:.2f}% |\n"

    md_content += f"""
### Interpretación de Frecuencia
Los códigos más frecuentes representan las condiciones o procedimientos más comunes en la población hospitalaria analizada.

**Código más frecuente:** **{top5_freq.index[0]}** aparece en {top5_freq.values[0]:,} pacientes ({(top5_freq.values[0]/total_pacientes)*100:.2f}% del total).

---

## 🔍 Relación Frecuencia vs. Outliers

### Códigos que son FRECUENTES y GENERAN OUTLIERS
"""

    # Códigos que aparecen en ambos tops
    codes_freq = set(top5_freq.index)
    codes_outliers = set(top5_outliers['codigo'].values)
    codes_both = codes_freq & codes_outliers

    if codes_both:
        md_content += f"\nLos siguientes códigos son **frecuentes Y generan outliers** (alta prioridad para gestión hospitalaria):\n\n"
        for code in codes_both:
            freq_count = top5_freq[code]
            outlier_row = top5_outliers[top5_outliers['codigo'] == code].iloc[0]
            md_content += f"- **{code}**: {freq_count:,} pacientes, {outlier_row['prob_outlier']*100:.1f}% prob. outlier\n"
    else:
        md_content += "\nNo hay códigos que sean simultáneamente top 5 en frecuencia y top 5 en probabilidad de outlier.\n"
        md_content += "\n**Esto indica que los códigos más comunes NO son los que generan estancias largas**, lo cual es positivo para la gestión hospitalaria (los casos frecuentes son predecibles y de corta duración).\n"

    md_content += f"""
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
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✅ Conclusiones guardadas en: {output_path}")

# ========================
# FUNCIÓN PRINCIPAL
# ========================

def analizar_tipo(df, columnas, tipo, output_dir):
    """
    Ejecuta análisis completo para diagnósticos o procedimientos.
    """
    print(f"\n{'='*60}")
    print(f"🔍 ANALIZANDO {tipo.upper()}")
    print(f"{'='*60}\n")

    # 1. Explotar códigos
    print(f"📋 Explotando códigos...")
    df_codigos = explotar_codigos(df, columnas, tipo)
    print(f"   Total registros código-paciente: {len(df_codigos):,}")
    print(f"   Códigos únicos: {df_codigos['codigo'].nunique():,}")
    print(f"   Pacientes únicos: {df_codigos['case_id'].nunique():,}")

    # 2. Identificar umbral outlier
    umbral_outlier = identificar_outliers_iqr(df_codigos)
    n_outliers = len(df_codigos[df_codigos['los_dias'] >= umbral_outlier])
    print(f"\n📊 Umbral outlier (Q3 + 1.5*IQR): {umbral_outlier:.1f} días")
    print(f"   Registros outlier: {n_outliers:,} ({(n_outliers/len(df_codigos))*100:.1f}%)")

    # 3. Analizar códigos outliers
    print(f"\n🔴 Analizando códigos asociados a outliers...")
    stats_outliers = analizar_codigos_outliers(df_codigos, umbral_outlier, top_n=20)
    print(f"   Top código outlier: {stats_outliers.iloc[0]['codigo']} ({stats_outliers.iloc[0]['prob_outlier']*100:.1f}% prob.)")

    # 4. Análisis de frecuencia
    print(f"\n📈 Analizando códigos frecuentes...")
    freq_top = df_codigos['codigo'].value_counts().head(20)
    print(f"   Top código frecuente: {freq_top.index[0]} ({freq_top.values[0]:,} pacientes)")

    # 5. Generar gráficos
    print(f"\n🎨 Generando gráficos...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Gráfico 1: Códigos outliers
    fig1 = graficar_top_codigos_outliers(stats_outliers, tipo, umbral_outlier)
    fig1.savefig(f"{output_dir}01_codigos_outliers.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"   ✅ 01_codigos_outliers.png")

    # Gráfico 2: Boxplot outliers
    fig2 = graficar_boxplot_codigos(df_codigos, stats_outliers, tipo)
    fig2.savefig(f"{output_dir}02_boxplot_outliers.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"   ✅ 02_boxplot_outliers.png")

    # Gráfico 3: Códigos frecuentes
    fig3 = graficar_densidad_frecuencia(df_codigos, tipo, top_n=20)
    fig3.savefig(f"{output_dir}03_codigos_frecuentes.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"   ✅ 03_codigos_frecuentes.png")

    # Gráfico 4: Violin plot frecuentes
    fig4 = graficar_violin_top_frecuentes(df_codigos, tipo, top_n=10)
    fig4.savefig(f"{output_dir}04_violin_frecuentes.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"   ✅ 04_violin_frecuentes.png")

    # 6. Exportar estadísticas
    print(f"\n💾 Exportando estadísticas...")
    stats_outliers.to_csv(f"{output_dir}estadisticas_outliers.csv", index=False, sep=';', encoding='utf-8')
    print(f"   ✅ estadisticas_outliers.csv")

    freq_df = pd.DataFrame({
        'codigo': freq_top.index,
        'n_pacientes': freq_top.values
    })
    freq_df.to_csv(f"{output_dir}estadisticas_frecuencia.csv", index=False, sep=';', encoding='utf-8')
    print(f"   ✅ estadisticas_frecuencia.csv")

    # 7. Generar conclusiones .md
    print(f"\n📝 Generando conclusiones...")
    generar_conclusiones_md(
        stats_outliers,
        freq_top,
        df_codigos,
        tipo,
        umbral_outlier,
        f"{output_dir}CONCLUSIONES.md"
    )

    print(f"\n✨ Análisis de {tipo} completado!\n")

# ========================
# EJECUCIÓN PRINCIPAL
# ========================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 ANÁLISIS DE CÓDIGOS ICD-10 Y OUTLIERS DE LOS")
    print("="*60)

    # Cargar datos
    print("\n📂 Cargando dataset maestro...")
    df = cargar_datos()
    print(f"   ✅ {len(df):,} pacientes cargados")
    print(f"   LOS promedio: {df['los_dias'].mean():.2f} días")
    print(f"   LOS mediana: {df['los_dias'].median():.1f} días")
    print(f"   LOS máximo: {df['los_dias'].max():.0f} días")

    # Análisis de DIAGNÓSTICOS
    analizar_tipo(
        df,
        columnas=['diagnosticos_primarios', 'diagnosticos_secundarios'],
        tipo='diagnóstico',
        output_dir=OUTPUT_DIAG
    )

    # Análisis de PROCEDIMIENTOS
    analizar_tipo(
        df,
        columnas=['procedimientos'],
        tipo='procedimiento',
        output_dir=OUTPUT_PROC
    )

    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*60)
    print(f"\n📁 Resultados guardados en:")
    print(f"   - {OUTPUT_DIAG}")
    print(f"   - {OUTPUT_PROC}")
    print("\n🎉 ¡Revisa los archivos CONCLUSIONES.md para insights clave!\n")
