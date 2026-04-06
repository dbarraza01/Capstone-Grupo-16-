"""
Script de Análisis Estadístico para Datasets de Predicción LOS Hospitalario
============================================================================

Este script genera reportes estadísticos detallados (estilo summary de R) para:
- dataset_maestro.csv
- caso_diagnostico.csv
- pacientes_rechazados.csv

Los reportes incluyen estadísticas descriptivas, distribuciones y métricas clave.

Autor: Sistema de Análisis LOS
Fecha: 2026-03-30
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

# Archivos de entrada
ARCHIVO_DATASET_MAESTRO = "data/processed/dataset_maestro.csv"
ARCHIVO_CASO_DIAGNOSTICO = "data/processed/caso_diagnostico.csv"
ARCHIVO_PACIENTES_RECHAZADOS = "data/processed/pacientes_rechazados.csv"

# Archivos de salida (reportes)
ARCHIVO_REPORTE_MAESTRO = "data/reports/reporte_estadistico_maestro.csv"
ARCHIVO_REPORTE_DIAGNOSTICO = "data/reports/reporte_estadistico_diagnostico.csv"
ARCHIVO_REPORTE_RECHAZADOS = "data/reports/reporte_estadistico_rechazados.csv"


# ============================================================================
# FUNCIÓN 1: ANÁLISIS ESTADÍSTICO PARA DATASET_MAESTRO
# ============================================================================

def analizar_dataset_maestro():
    """
    Genera reporte estadístico completo para dataset_maestro.csv

    Incluye:
    - Estadísticas descriptivas numéricas (count, mean, std, min, max, percentiles)
    - Distribución de urgencias
    - Distribución de LOS
    - Estadísticas de procedimientos y diagnósticos

    Returns:
        DataFrame con reporte estadístico
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS ESTADÍSTICO: dataset_maestro.csv")
    print("=" * 70)

    # Cargar dataset
    df = pd.read_csv(ARCHIVO_DATASET_MAESTRO, sep=';', dtype=str)

    # Convertir columnas numéricas
    df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')
    df['es_urgencia'] = pd.to_numeric(df['es_urgencia'], errors='coerce')
    df['n_procedimientos'] = pd.to_numeric(df['n_procedimientos'], errors='coerce')
    df['n_diag_primarios'] = pd.to_numeric(df['n_diag_primarios'], errors='coerce')
    df['n_diag_secundarios'] = pd.to_numeric(df['n_diag_secundarios'], errors='coerce')
    df['n_diag_total'] = pd.to_numeric(df['n_diag_total'], errors='coerce')

    print(f"Total de registros: {len(df):,}")

    # Crear lista de estadísticas
    estadisticas = []

    # -------------------------------------------------------------------------
    # SECCIÓN 1: Información General
    # -------------------------------------------------------------------------
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'numero_pacientes', 'valor': len(df)})
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'periodo_inicio', 'valor': df['fecha_ingreso'].min()})
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'periodo_fin', 'valor': df['fecha_egreso'].max()})

    # -------------------------------------------------------------------------
    # SECCIÓN 2: LOS (Length of Stay)
    # -------------------------------------------------------------------------
    los_stats = df['los_dias'].describe()
    estadisticas.append({'seccion': 'LOS', 'variable': 'count', 'valor': los_stats['count']})
    estadisticas.append({'seccion': 'LOS', 'variable': 'mean', 'valor': round(los_stats['mean'], 2)})
    estadisticas.append({'seccion': 'LOS', 'variable': 'std', 'valor': round(los_stats['std'], 2)})
    estadisticas.append({'seccion': 'LOS', 'variable': 'min', 'valor': los_stats['min']})
    estadisticas.append({'seccion': 'LOS', 'variable': 'percentil_25', 'valor': los_stats['25%']})
    estadisticas.append({'seccion': 'LOS', 'variable': 'median', 'valor': los_stats['50%']})
    estadisticas.append({'seccion': 'LOS', 'variable': 'percentil_75', 'valor': los_stats['75%']})
    estadisticas.append({'seccion': 'LOS', 'variable': 'max', 'valor': los_stats['max']})

    # Cuartiles adicionales
    estadisticas.append({'seccion': 'LOS', 'variable': 'percentil_90', 'valor': df['los_dias'].quantile(0.90)})
    estadisticas.append({'seccion': 'LOS', 'variable': 'percentil_95', 'valor': df['los_dias'].quantile(0.95)})
    estadisticas.append({'seccion': 'LOS', 'variable': 'percentil_99', 'valor': df['los_dias'].quantile(0.99)})

    # Rangos de LOS
    estadisticas.append({'seccion': 'LOS_RANGOS', 'variable': 'los_0', 'valor': (df['los_dias'] == 0).sum()})
    estadisticas.append({'seccion': 'LOS_RANGOS', 'variable': 'los_1_3', 'valor': ((df['los_dias'] >= 1) & (df['los_dias'] <= 3)).sum()})
    estadisticas.append({'seccion': 'LOS_RANGOS', 'variable': 'los_4_7', 'valor': ((df['los_dias'] >= 4) & (df['los_dias'] <= 7)).sum()})
    estadisticas.append({'seccion': 'LOS_RANGOS', 'variable': 'los_8_14', 'valor': ((df['los_dias'] >= 8) & (df['los_dias'] <= 14)).sum()})
    estadisticas.append({'seccion': 'LOS_RANGOS', 'variable': 'los_15_30', 'valor': ((df['los_dias'] >= 15) & (df['los_dias'] <= 30)).sum()})
    estadisticas.append({'seccion': 'LOS_RANGOS', 'variable': 'los_mas_30', 'valor': (df['los_dias'] > 30).sum()})

    # -------------------------------------------------------------------------
    # SECCIÓN 3: Urgencias
    # -------------------------------------------------------------------------
    urgencias_count = df['es_urgencia'].value_counts()
    estadisticas.append({'seccion': 'URGENCIA', 'variable': 'pacientes_urgencia', 'valor': urgencias_count.get(1, 0)})
    estadisticas.append({'seccion': 'URGENCIA', 'variable': 'pacientes_electivos', 'valor': urgencias_count.get(0, 0)})
    estadisticas.append({'seccion': 'URGENCIA', 'variable': 'porcentaje_urgencias', 'valor': round((urgencias_count.get(1, 0) / len(df)) * 100, 2)})

    # LOS promedio por tipo de ingreso
    los_urgencia = df[df['es_urgencia'] == 1]['los_dias'].mean()
    los_electivo = df[df['es_urgencia'] == 0]['los_dias'].mean()
    estadisticas.append({'seccion': 'URGENCIA', 'variable': 'los_promedio_urgencia', 'valor': round(los_urgencia, 2)})
    estadisticas.append({'seccion': 'URGENCIA', 'variable': 'los_promedio_electivo', 'valor': round(los_electivo, 2)})

    # -------------------------------------------------------------------------
    # SECCIÓN 4: Procedimientos
    # -------------------------------------------------------------------------
    proc_stats = df['n_procedimientos'].describe()
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'mean', 'valor': round(proc_stats['mean'], 2)})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'std', 'valor': round(proc_stats['std'], 2)})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'min', 'valor': proc_stats['min']})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'median', 'valor': proc_stats['50%']})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'max', 'valor': proc_stats['max']})

    # Distribución de número de procedimientos
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'sin_procedimientos', 'valor': (df['n_procedimientos'] == 0).sum()})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'proc_1_2', 'valor': ((df['n_procedimientos'] >= 1) & (df['n_procedimientos'] <= 2)).sum()})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'proc_3_5', 'valor': ((df['n_procedimientos'] >= 3) & (df['n_procedimientos'] <= 5)).sum()})
    estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'proc_mas_5', 'valor': (df['n_procedimientos'] > 5).sum()})

    # -------------------------------------------------------------------------
    # SECCIÓN 5: Diagnósticos
    # -------------------------------------------------------------------------
    diag_stats = df['n_diag_total'].describe()
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'mean_total', 'valor': round(diag_stats['mean'], 2)})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'std_total', 'valor': round(diag_stats['std'], 2)})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'min_total', 'valor': diag_stats['min']})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'median_total', 'valor': diag_stats['50%']})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'max_total', 'valor': diag_stats['max']})

    # Diagnósticos primarios vs secundarios
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'mean_primarios', 'valor': round(df['n_diag_primarios'].mean(), 2)})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'mean_secundarios', 'valor': round(df['n_diag_secundarios'].mean(), 2)})

    # Distribución de número de diagnósticos
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'diag_1_5', 'valor': ((df['n_diag_total'] >= 1) & (df['n_diag_total'] <= 5)).sum()})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'diag_6_10', 'valor': ((df['n_diag_total'] >= 6) & (df['n_diag_total'] <= 10)).sum()})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'diag_11_15', 'valor': ((df['n_diag_total'] >= 11) & (df['n_diag_total'] <= 15)).sum()})
    estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'diag_mas_15', 'valor': (df['n_diag_total'] > 15).sum()})

    # -------------------------------------------------------------------------
    # SECCIÓN 6: Correlaciones (LOS vs otras variables)
    # -------------------------------------------------------------------------
    corr_procedimientos = df[['los_dias', 'n_procedimientos']].corr().iloc[0, 1]
    corr_diagnosticos = df[['los_dias', 'n_diag_total']].corr().iloc[0, 1]

    estadisticas.append({'seccion': 'CORRELACIONES', 'variable': 'los_vs_procedimientos', 'valor': round(corr_procedimientos, 3)})
    estadisticas.append({'seccion': 'CORRELACIONES', 'variable': 'los_vs_diagnosticos', 'valor': round(corr_diagnosticos, 3)})

    # Crear DataFrame de reporte
    df_reporte = pd.DataFrame(estadisticas)

    print(f"✓ Generadas {len(estadisticas)} estadísticas para dataset maestro")

    return df_reporte


# ============================================================================
# FUNCIÓN 2: ANÁLISIS ESTADÍSTICO PARA CASO_DIAGNOSTICO
# ============================================================================

def analizar_caso_diagnostico():
    """
    Genera reporte estadístico completo para caso_diagnostico.csv

    Incluye:
    - Total de diagnósticos y pacientes únicos
    - Distribución de diagnósticos primarios vs secundarios
    - Top 20 códigos más frecuentes
    - Top 20 capítulos (caract_1) más frecuentes
    - Top 20 categorías (caract_3) más frecuentes

    Returns:
        DataFrame con reporte estadístico
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS ESTADÍSTICO: caso_diagnostico.csv")
    print("=" * 70)

    # Cargar dataset
    df = pd.read_csv(ARCHIVO_CASO_DIAGNOSTICO, sep=';', dtype=str)

    print(f"Total de registros: {len(df):,}")

    # Crear lista de estadísticas
    estadisticas = []

    # -------------------------------------------------------------------------
    # SECCIÓN 1: Información General
    # -------------------------------------------------------------------------
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'numero_diagnosticos', 'valor': len(df)})
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'pacientes_unicos', 'valor': df['case_id'].nunique()})
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'codigos_unicos', 'valor': df['d_code'].nunique()})
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'diagnosticos_por_paciente_promedio',
                        'valor': round(len(df) / df['case_id'].nunique(), 2)})

    # -------------------------------------------------------------------------
    # SECCIÓN 2: Tipo de Diagnóstico (P/S)
    # -------------------------------------------------------------------------
    tipo_counts = df['tipo_d'].value_counts()
    estadisticas.append({'seccion': 'TIPO', 'variable': 'primarios', 'valor': tipo_counts.get('P', 0)})
    estadisticas.append({'seccion': 'TIPO', 'variable': 'secundarios', 'valor': tipo_counts.get('S', 0)})
    estadisticas.append({'seccion': 'TIPO', 'variable': 'porcentaje_primarios',
                        'valor': round((tipo_counts.get('P', 0) / len(df)) * 100, 2)})

    # -------------------------------------------------------------------------
    # SECCIÓN 3: Top 20 Códigos más frecuentes
    # -------------------------------------------------------------------------
    top_codigos = df['d_code'].value_counts().head(20)
    for i, (codigo, freq) in enumerate(top_codigos.items(), 1):
        estadisticas.append({'seccion': 'TOP_CODIGOS', 'variable': f'top_{i:02d}', 'valor': f'{codigo} ({freq})'})

    # -------------------------------------------------------------------------
    # SECCIÓN 4: Top 20 Capítulos (primera letra)
    # -------------------------------------------------------------------------
    top_capitulos = df['d_caract_1'].value_counts().head(20)
    for i, (capitulo, freq) in enumerate(top_capitulos.items(), 1):
        estadisticas.append({'seccion': 'TOP_CAPITULOS', 'variable': f'top_{i:02d}', 'valor': f'{capitulo} ({freq})'})

    # -------------------------------------------------------------------------
    # SECCIÓN 5: Top 20 Categorías (primeros 3 caracteres)
    # -------------------------------------------------------------------------
    top_categorias = df['d_caract_3'].value_counts().head(20)
    for i, (categoria, freq) in enumerate(top_categorias.items(), 1):
        estadisticas.append({'seccion': 'TOP_CATEGORIAS', 'variable': f'top_{i:02d}', 'valor': f'{categoria} ({freq})'})

    # -------------------------------------------------------------------------
    # SECCIÓN 6: Distribución por paciente
    # -------------------------------------------------------------------------
    diag_por_paciente = df.groupby('case_id').size()
    estadisticas.append({'seccion': 'DISTRIBUCION', 'variable': 'min_diag_por_paciente', 'valor': diag_por_paciente.min()})
    estadisticas.append({'seccion': 'DISTRIBUCION', 'variable': 'max_diag_por_paciente', 'valor': diag_por_paciente.max()})
    estadisticas.append({'seccion': 'DISTRIBUCION', 'variable': 'median_diag_por_paciente', 'valor': diag_por_paciente.median()})

    # Crear DataFrame de reporte
    df_reporte = pd.DataFrame(estadisticas)

    print(f"✓ Generadas {len(estadisticas)} estadísticas para caso diagnóstico")

    return df_reporte


# ============================================================================
# FUNCIÓN 3: ANÁLISIS ESTADÍSTICO PARA PACIENTES_RECHAZADOS
# ============================================================================

def analizar_pacientes_rechazados():
    """
    Genera reporte estadístico completo para pacientes_rechazados.csv

    Incluye:
    - Total de pacientes rechazados
    - Distribución por motivo de rechazo
    - Estadísticas de LOS para rechazados
    - Códigos diagnósticos inválidos más frecuentes

    Returns:
        DataFrame con reporte estadístico
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS ESTADÍSTICO: pacientes_rechazados.csv")
    print("=" * 70)

    # Cargar dataset
    df = pd.read_csv(ARCHIVO_PACIENTES_RECHAZADOS, sep=';', dtype=str)

    # Convertir columnas numéricas
    df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')
    df['n_procedimientos'] = pd.to_numeric(df['n_procedimientos'], errors='coerce')
    df['n_diag_total'] = pd.to_numeric(df['n_diag_total'], errors='coerce')

    print(f"Total de registros: {len(df):,}")

    # Crear lista de estadísticas
    estadisticas = []

    # -------------------------------------------------------------------------
    # SECCIÓN 1: Información General
    # -------------------------------------------------------------------------
    estadisticas.append({'seccion': 'GENERAL', 'variable': 'pacientes_rechazados', 'valor': len(df)})

    # -------------------------------------------------------------------------
    # SECCIÓN 2: Motivos de Rechazo
    # -------------------------------------------------------------------------
    motivos_count = df['motivo_rechazo'].value_counts()
    for i, (motivo, count) in enumerate(motivos_count.items(), 1):
        estadisticas.append({'seccion': 'MOTIVOS', 'variable': f'motivo_{i:02d}', 'valor': f'{motivo} ({count})'})

    # -------------------------------------------------------------------------
    # SECCIÓN 3: Códigos Diagnósticos Inválidos
    # -------------------------------------------------------------------------
    # Contar códigos inválidos únicos
    codigos_invalidos = df['codigos_diagnostico_invalidos'].dropna()
    codigos_invalidos = codigos_invalidos[codigos_invalidos != '']

    if len(codigos_invalidos) > 0:
        # Separar por comas y contar
        todos_codigos = []
        for codigos in codigos_invalidos:
            todos_codigos.extend(codigos.split(','))

        codigos_unicos = pd.Series(todos_codigos).value_counts()
        estadisticas.append({'seccion': 'CODIGOS_INVALIDOS', 'variable': 'total_codigos_invalidos', 'valor': len(todos_codigos)})
        estadisticas.append({'seccion': 'CODIGOS_INVALIDOS', 'variable': 'codigos_unicos', 'valor': len(codigos_unicos)})

        # Top códigos inválidos
        for i, (codigo, freq) in enumerate(codigos_unicos.items(), 1):
            estadisticas.append({'seccion': 'CODIGOS_INVALIDOS', 'variable': f'codigo_{i:02d}', 'valor': f'{codigo} ({freq})'})

    # -------------------------------------------------------------------------
    # SECCIÓN 4: Estadísticas de LOS (si existen)
    # -------------------------------------------------------------------------
    if not df['los_dias'].isna().all():
        los_validos = df['los_dias'].dropna()
        if len(los_validos) > 0:
            estadisticas.append({'seccion': 'LOS', 'variable': 'count', 'valor': len(los_validos)})
            estadisticas.append({'seccion': 'LOS', 'variable': 'mean', 'valor': round(los_validos.mean(), 2)})
            estadisticas.append({'seccion': 'LOS', 'variable': 'median', 'valor': los_validos.median()})
            estadisticas.append({'seccion': 'LOS', 'variable': 'min', 'valor': los_validos.min()})
            estadisticas.append({'seccion': 'LOS', 'variable': 'max', 'valor': los_validos.max()})

    # -------------------------------------------------------------------------
    # SECCIÓN 5: Estadísticas de Procedimientos y Diagnósticos
    # -------------------------------------------------------------------------
    if not df['n_procedimientos'].isna().all():
        proc_validos = df['n_procedimientos'].dropna()
        if len(proc_validos) > 0:
            estadisticas.append({'seccion': 'PROCEDIMIENTOS', 'variable': 'mean', 'valor': round(proc_validos.mean(), 2)})

    if not df['n_diag_total'].isna().all():
        diag_validos = df['n_diag_total'].dropna()
        if len(diag_validos) > 0:
            estadisticas.append({'seccion': 'DIAGNOSTICOS', 'variable': 'mean', 'valor': round(diag_validos.mean(), 2)})

    # Crear DataFrame de reporte
    df_reporte = pd.DataFrame(estadisticas)

    print(f"✓ Generadas {len(estadisticas)} estadísticas para pacientes rechazados")

    return df_reporte


# ============================================================================
# FUNCIÓN 4: GUARDAR REPORTES
# ============================================================================

def guardar_reportes(df_reporte_maestro, df_reporte_diagnostico, df_reporte_rechazados):
    """
    Guarda todos los reportes estadísticos en archivos CSV.

    Args:
        df_reporte_maestro: Reporte estadístico del dataset maestro
        df_reporte_diagnostico: Reporte estadístico de caso_diagnostico
        df_reporte_rechazados: Reporte estadístico de pacientes_rechazados
    """
    print("\n" + "=" * 70)
    print("GUARDANDO REPORTES ESTADÍSTICOS...")
    print("=" * 70)

    # Crear directorio si no existe
    import os
    os.makedirs('data/reports', exist_ok=True)

    # Guardar reportes
    df_reporte_maestro.to_csv(ARCHIVO_REPORTE_MAESTRO, index=False, sep=';')
    print(f"✓ Reporte maestro guardado: {ARCHIVO_REPORTE_MAESTRO}")
    print(f"  Estadísticas: {len(df_reporte_maestro):,}")

    df_reporte_diagnostico.to_csv(ARCHIVO_REPORTE_DIAGNOSTICO, index=False, sep=';')
    print(f"✓ Reporte diagnóstico guardado: {ARCHIVO_REPORTE_DIAGNOSTICO}")
    print(f"  Estadísticas: {len(df_reporte_diagnostico):,}")

    df_reporte_rechazados.to_csv(ARCHIVO_REPORTE_RECHAZADOS, index=False, sep=';')
    print(f"✓ Reporte rechazados guardado: {ARCHIVO_REPORTE_RECHAZADOS}")
    print(f"  Estadísticas: {len(df_reporte_rechazados):,}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todos los análisis estadísticos.
    """
    print("\n")
    print("█" * 70)
    print("  ANÁLISIS ESTADÍSTICO - PREDICCIÓN LOS HOSPITALARIO")
    print("█" * 70)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Análisis 1: Dataset Maestro
    df_reporte_maestro = analizar_dataset_maestro()

    # Análisis 2: Caso Diagnóstico
    df_reporte_diagnostico = analizar_caso_diagnostico()

    # Análisis 3: Pacientes Rechazados
    df_reporte_rechazados = analizar_pacientes_rechazados()

    # Guardar reportes
    guardar_reportes(df_reporte_maestro, df_reporte_diagnostico, df_reporte_rechazados)

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\n>>> Todos los reportes estadísticos han sido generados <<<\n")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
