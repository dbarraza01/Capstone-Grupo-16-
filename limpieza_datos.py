"""
Script de Limpieza de Datos para Predicción de LOS (Length of Stay) Hospitalario
==================================================================================

Este script procesa datos de diagnósticos y procedimientos de pacientes,
los limpia, valida y genera un dataset maestro para modelado predictivo.

Autor: Sistema de Limpieza LOS
Fecha: 2026-03-18
"""

import pandas as pd
import re
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE PARÁMETROS
# ============================================================================

# Rutas de archivos de entrada (CSV con separador ;)
ARCHIVO_DIAGNOSTICOS = "datos_diagnostico.csv"
ARCHIVO_PROCEDIMIENTOS = "procedimiento_pacientes.csv"

# Rutas de archivos de salida
ARCHIVO_SALIDA_MAESTRO = "data/processed/dataset_maestro.csv"
ARCHIVO_SALIDA_RECHAZADOS = "data/processed/pacientes_rechazados.csv"
ARCHIVO_SALIDA_REPORTE = "data/reports/reporte_limpieza.csv"

# Patrón REGEX para códigos ICD-10 válidos
# ICD-10: Letra (A-Z excepto U) + 2-6 caracteres alfanuméricos
# Ejemplos válidos: E6601, S72302E, Z6841, I10
# Ejemplos inválidos: UUUUUU, 123, ABC
ICD10_REGEX = r'^[A-TV-Z][0-9]{1,2}[A-Z0-9]{0,5}$'

# Valores permitidos para PrincSec (Principal/Secundario)
VALORES_PRINCSEC_VALIDOS = {'P', 'S'}


# ============================================================================
# FUNCIÓN 1: CARGAR DATOS CRUDOS
# ============================================================================

def cargar_datos_crudos():
    """
    Carga los archivos CSV de diagnósticos y procedimientos.

    Returns:
        tuple: (df_diagnosticos, df_procedimientos)

    Nota: Todos los datos se cargan como string (dtype=str) para evitar
            conversiones automáticas que puedan causar pérdida de información.
    """
    print("=" * 70)
    print("PASO 1: Cargando archivos CSV...")
    print("=" * 70)

    # Cargar archivo de diagnósticos
    # Columnas esperadas: CASE, PrincSec, Diagnosis
    df_diagnosticos = pd.read_csv(ARCHIVO_DIAGNOSTICOS, sep=';', dtype=str)
    print(f"✓ Diagnósticos cargados: {len(df_diagnosticos):,} registros")
    print(f"  Columnas: {list(df_diagnosticos.columns)}")

    # Cargar archivo de procedimientos
    # Columnas esperadas: Case, Procedure, Date, Release
    df_procedimientos = pd.read_csv(ARCHIVO_PROCEDIMIENTOS, sep=';', dtype=str)
    print(f"✓ Procedimientos cargados: {len(df_procedimientos):,} registros")
    print(f"  Columnas: {list(df_procedimientos.columns)}")

    return df_diagnosticos, df_procedimientos


# ============================================================================
# FUNCIÓN 2: LIMPIAR Y VALIDAR DIAGNÓSTICOS
# ============================================================================

def limpiar_diagnosticos(df):
    """
    Limpia y valida los datos de diagnósticos.

    Proceso:
    1. Estandariza nombres de columnas
    2. Identifica códigos ICD-10 inválidos (pero NO elimina pacientes)
    3. Filtra solo códigos válidos para agregación
    4. Mantiene registro de códigos problemáticos para auditoría

    Args:
        df: DataFrame con columnas CASE, PrincSec, Diagnosis

    Returns:
        DataFrame procesado a nivel paciente con listas de diagnósticos
    """
    print("\n" + "=" * 70)
    print("PASO 2: Limpiando y validando diagnósticos...")
    print("=" * 70)

    # Crear copia para no modificar original
    df = df.copy()

    # Estandarizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()

    # Renombrar 'case' a 'case_id' para claridad
    df = df.rename(columns={'case': 'case_id'})

    # Total de registros originales
    total_registros = len(df)
    print(f"Registros totales: {total_registros:,}")

    # -------------------------------------------------------------------------
    # Validación 1: PrincSec debe ser 'P' (Principal) o 'S' (Secundario)
    # -------------------------------------------------------------------------
    df['princsec_valido'] = df['princsec'].isin(VALORES_PRINCSEC_VALIDOS)
    invalidos_princsec = (~df['princsec_valido']).sum()
    print(f"Registros con PrincSec inválido: {invalidos_princsec:,}")

    # -------------------------------------------------------------------------
    # Validación 2: Diagnosis debe cumplir patrón ICD-10
    # -------------------------------------------------------------------------
    df['diagnosis_valido'] = df['diagnosis'].str.match(ICD10_REGEX, na=False)
    invalidos_diagnosis = (~df['diagnosis_valido']).sum()
    print(f"Registros con código ICD-10 inválido: {invalidos_diagnosis:,}")

    # Mostrar ejemplos de códigos inválidos (para auditoría)
    if invalidos_diagnosis > 0:
        codigos_invalidos_unicos = df[~df['diagnosis_valido']]['diagnosis'].unique()
        print(f"  Ejemplos de códigos inválidos: {codigos_invalidos_unicos[:10]}")

    # -------------------------------------------------------------------------
    # DECISIÓN CRÍTICA: Solo filtrar REGISTROS con códigos inválidos,
    # NO eliminar pacientes completos
    # -------------------------------------------------------------------------
    # Filtrar solo registros que cumplan AMBAS validaciones
    df_valido = df[df['princsec_valido'] & df['diagnosis_valido']].copy()

    registros_eliminados = len(df) - len(df_valido)
    print(f"\n✓ Registros eliminados por validación: {registros_eliminados:,}")
    print(f"✓ Registros válidos conservados: {len(df_valido):,}")

    # -------------------------------------------------------------------------
    # Agregación a nivel PACIENTE
    # -------------------------------------------------------------------------
    print("\nAgregando diagnósticos por paciente...")

    # Separar diagnósticos primarios (P) y secundarios (S)
    df_primarios = df_valido[df_valido['princsec'] == 'P']
    df_secundarios = df_valido[df_valido['princsec'] == 'S']

    # Crear listas de diagnósticos por paciente
    # groupby + apply(list) convierte múltiples filas en una lista
    diag_primarios = (
        df_primarios
        .groupby('case_id')['diagnosis']
        .apply(list)
        .rename('diagnosticos_primarios')
    )

    diag_secundarios = (
        df_secundarios
        .groupby('case_id')['diagnosis']
        .apply(list)
        .rename('diagnosticos_secundarios')
    )

    # Unir diagnósticos primarios y secundarios
    # outer join para conservar pacientes que solo tienen uno u otro
    df_paciente = (
        diag_primarios
        .to_frame()
        .join(diag_secundarios.to_frame(), how='outer')
        .reset_index()
    )

    # Reemplazar NaN con listas vacías
    df_paciente['diagnosticos_primarios'] = (
        df_paciente['diagnosticos_primarios']
        .apply(lambda x: x if isinstance(x, list) else [])
    )
    df_paciente['diagnosticos_secundarios'] = (
        df_paciente['diagnosticos_secundarios']
        .apply(lambda x: x if isinstance(x, list) else [])
    )

    # Crear contadores
    df_paciente['n_diag_primarios'] = (
        df_paciente['diagnosticos_primarios'].apply(len)
    )
    df_paciente['n_diag_secundarios'] = (
        df_paciente['diagnosticos_secundarios'].apply(len)
    )
    df_paciente['n_diag_total'] = (
        df_paciente['n_diag_primarios'] + df_paciente['n_diag_secundarios']
    )

    # Flag de si tiene al menos un diagnóstico primario
    df_paciente['tiene_diag_primario'] = df_paciente['n_diag_primarios'] > 0

    print(f"✓ Pacientes únicos con diagnósticos: {len(df_paciente):,}")
    print(f"  - Con diagnóstico primario: {df_paciente['tiene_diag_primario'].sum():,}")
    print(f"  - Solo diagnósticos secundarios: {(~df_paciente['tiene_diag_primario']).sum():,}")

    return df_paciente


# ============================================================================
# FUNCIÓN 3: LIMPIAR Y VALIDAR PROCEDIMIENTOS
# ============================================================================

def limpiar_procedimientos(df):
    """
    Limpia y valida los datos de procedimientos.

    Proceso:
    1. Estandariza nombres de columnas
    2. Convierte fechas a formato datetime
    3. Calcula LOS (Length of Stay en días)
    4. Valida fechas y LOS
    5. Agrega a nivel paciente

    Args:
        df: DataFrame con columnas Case, Procedure, Date, Release

    Returns:
        DataFrame procesado a nivel paciente con métricas de procedimientos y LOS
    """
    print("\n" + "=" * 70)
    print("PASO 3: Limpiando y validando procedimientos...")
    print("=" * 70)

    # Crear copia
    df = df.copy()

    # Estandarizar nombres de columnas
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'case': 'case_id',
        'procedure': 'procedimiento',
        'date': 'fecha_ingreso',
        'release': 'fecha_egreso'
    })

    total_registros = len(df)
    print(f"Registros totales: {total_registros:,}")

    # -------------------------------------------------------------------------
    # Conversión de fechas (formato DD-MM-YY)
    # -------------------------------------------------------------------------
    print("\nConvirtiendo fechas...")

    # Convertir strings a datetime
    # dayfirst=True porque el formato es DD-MM-YY
    df['fecha_ingreso'] = pd.to_datetime(
        df['fecha_ingreso'],
        format='%d-%m-%y',
        errors='coerce'
    )
    df['fecha_egreso'] = pd.to_datetime(
        df['fecha_egreso'],
        format='%d-%m-%y',
        errors='coerce'
    )

    # Identificar fechas inválidas (NaT = Not a Time)
    fechas_ingreso_invalidas = df['fecha_ingreso'].isna().sum()
    fechas_egreso_invalidas = df['fecha_egreso'].isna().sum()

    print(f"  Fechas de ingreso inválidas: {fechas_ingreso_invalidas:,}")
    print(f"  Fechas de egreso inválidas: {fechas_egreso_invalidas:,}")

    # -------------------------------------------------------------------------
    # Agregación por PACIENTE (antes de calcular LOS)
    # -------------------------------------------------------------------------
    # Necesitamos agrupar por paciente para tener:
    # - Fecha de ingreso más temprana (primer procedimiento)
    # - Fecha de egreso más tardía (último procedimiento)
    # - Lista de todos los procedimientos

    print("\nAgregando por paciente...")

    # Crear lista de procedimientos no vacíos
    df_con_proc = df[df['procedimiento'].notna() & (df['procedimiento'] != '')]

    procedimientos_por_paciente = (
        df_con_proc
        .groupby('case_id')['procedimiento']
        .apply(list)
        .rename('procedimientos')
    )

    # Obtener fechas mínimas y máximas por paciente
    fechas_por_paciente = (
        df.groupby('case_id')
        .agg({
            'fecha_ingreso': 'min',  # Primera fecha de ingreso
            'fecha_egreso': 'max'     # Última fecha de egreso
        })
    )

    # Unir procedimientos y fechas
    df_paciente = (
        fechas_por_paciente
        .join(procedimientos_por_paciente, how='left')
        .reset_index()
    )

    # Reemplazar NaN en procedimientos con lista vacía
    df_paciente['procedimientos'] = (
        df_paciente['procedimientos']
        .apply(lambda x: x if isinstance(x, list) else [])
    )

    # Contar procedimientos
    df_paciente['n_procedimientos'] = df_paciente['procedimientos'].apply(len)

    # -------------------------------------------------------------------------
    # Cálculo de LOS (Length of Stay)
    # -------------------------------------------------------------------------
    print("\nCalculando LOS (Length of Stay)...")

    # LOS = diferencia en días entre egreso e ingreso
    df_paciente['los_dias'] = (
        (df_paciente['fecha_egreso'] - df_paciente['fecha_ingreso']).dt.days
    )

    # Flags de validación
    df_paciente['fechas_invalidas'] = (
        df_paciente['fecha_ingreso'].isna() |
        df_paciente['fecha_egreso'].isna()
    )

    df_paciente['los_negativo'] = df_paciente['los_dias'] < 0
    df_paciente['los_cero'] = df_paciente['los_dias'] == 0

    # Estadísticas de LOS
    print(f"\n✓ Pacientes únicos: {len(df_paciente):,}")
    print(f"  - Con fechas inválidas: {df_paciente['fechas_invalidas'].sum():,}")
    print(f"  - Con LOS negativo: {df_paciente['los_negativo'].sum():,}")
    print(f"  - Con LOS = 0: {df_paciente['los_cero'].sum():,} (CONSERVADOS como dato válido)")

    # Estadísticas de LOS válido (no negativo y no nulo)
    los_valido = df_paciente[
        (~df_paciente['fechas_invalidas']) &
        (~df_paciente['los_negativo'])
    ]['los_dias']

    if len(los_valido) > 0:
        print(f"\nEstadísticas de LOS válido:")
        print(f"  - Mínimo: {los_valido.min():.0f} días")
        print(f"  - Máximo: {los_valido.max():.0f} días")
        print(f"  - Promedio: {los_valido.mean():.1f} días")
        print(f"  - Mediana: {los_valido.median():.0f} días")

    return df_paciente


# ============================================================================
# FUNCIÓN 4: INTEGRAR DIAGNÓSTICOS Y PROCEDIMIENTOS
# ============================================================================

def integrar_datos(df_diagnosticos_paciente, df_procedimientos_paciente):
    """
    Integra los datos de diagnósticos y procedimientos en un dataset maestro.

    Regla de rechazo:
    - SOLO se rechazan pacientes que están en un archivo pero NO en el otro
    - NO se rechazan pacientes por LOS=0 (son datos válidos)
    - NO se rechazan pacientes por tener códigos inválidos filtrados
    - SÍ se rechazan pacientes con LOS negativo o fechas inválidas

    Args:
        df_diagnosticos_paciente: DataFrame de diagnósticos agregados
        df_procedimientos_paciente: DataFrame de procedimientos agregados

    Returns:
        tuple: (df_maestro, df_rechazados, df_completo)
    """
    print("\n" + "=" * 70)
    print("PASO 4: Integrando diagnósticos y procedimientos...")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Merge OUTER para identificar pacientes en solo uno de los archivos
    # -------------------------------------------------------------------------
    df = df_procedimientos_paciente.merge(
        df_diagnosticos_paciente,
        on='case_id',
        how='outer',
        indicator=True  # Crea columna '_merge' indicando origen
    )

    # Estadísticas de merge
    solo_procedimientos = (df['_merge'] == 'left_only').sum()
    solo_diagnosticos = (df['_merge'] == 'right_only').sum()
    en_ambos = (df['_merge'] == 'both').sum()

    print(f"\nResultados del merge:")
    print(f"  - Pacientes en ambos archivos: {en_ambos:,}")
    print(f"  - Solo en procedimientos: {solo_procedimientos:,}")
    print(f"  - Solo en diagnósticos: {solo_diagnosticos:,}")
    print(f"  - Total único de pacientes: {len(df):,}")

    # -------------------------------------------------------------------------
    # Rellenar listas vacías en columnas de listas
    # -------------------------------------------------------------------------
    columnas_lista = [
        'procedimientos',
        'diagnosticos_primarios',
        'diagnosticos_secundarios'
    ]

    for col in columnas_lista:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # Recalcular contadores después del merge (pueden tener NaN)
    df['n_procedimientos'] = df['procedimientos'].apply(len)
    df['n_diag_primarios'] = df['diagnosticos_primarios'].apply(len)
    df['n_diag_secundarios'] = df['diagnosticos_secundarios'].apply(len)
    df['n_diag_total'] = df['n_diag_primarios'] + df['n_diag_secundarios']
    df['tiene_diag_primario'] = df['n_diag_primarios'] > 0

    # -------------------------------------------------------------------------
    # Criterios de RECHAZO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Aplicando criterios de rechazo...")
    print("-" * 70)

    # Lista de razones de rechazo para cada paciente
    df['razones_rechazo'] = [[] for _ in range(len(df))]

    # CRITERIO 1: Paciente solo en procedimientos (no en diagnósticos)
    mask_solo_proc = df['_merge'] == 'left_only'
    df.loc[mask_solo_proc, 'razones_rechazo'] = df.loc[mask_solo_proc, 'razones_rechazo'].apply(
        lambda x: x + ['falta_en_diagnosticos']
    )
    print(f"  ✗ Falta en diagnósticos: {mask_solo_proc.sum():,}")

    # CRITERIO 2: Paciente solo en diagnósticos (no en procedimientos)
    mask_solo_diag = df['_merge'] == 'right_only'
    df.loc[mask_solo_diag, 'razones_rechazo'] = df.loc[mask_solo_diag, 'razones_rechazo'].apply(
        lambda x: x + ['falta_en_procedimientos']
    )
    print(f"  ✗ Falta en procedimientos: {mask_solo_diag.sum():,}")

    # CRITERIO 3: Fechas inválidas (NaT)
    mask_fechas_invalidas = df['fechas_invalidas'].fillna(False)
    df.loc[mask_fechas_invalidas, 'razones_rechazo'] = df.loc[mask_fechas_invalidas, 'razones_rechazo'].apply(
        lambda x: x + ['fechas_invalidas']
    )
    print(f"  ✗ Fechas inválidas: {mask_fechas_invalidas.sum():,}")

    # CRITERIO 4: LOS negativo (fecha egreso < fecha ingreso)
    mask_los_negativo = df['los_negativo'].fillna(False)
    df.loc[mask_los_negativo, 'razones_rechazo'] = df.loc[mask_los_negativo, 'razones_rechazo'].apply(
        lambda x: x + ['los_negativo']
    )
    print(f"  ✗ LOS negativo: {mask_los_negativo.sum():,}")

    # CRITERIOS EXPLÍCITAMENTE NO APLICADOS:
    print("\n  ✓ LOS = 0: NO rechazados (son datos válidos)")
    print("  ✓ Códigos diagnóstico inválidos: filtrados pero paciente conservado")

    # -------------------------------------------------------------------------
    # Separar dataset maestro y rechazados
    # -------------------------------------------------------------------------
    df['es_rechazado'] = df['razones_rechazo'].apply(len) > 0
    df['motivo_rechazo'] = df['razones_rechazo'].apply(lambda x: ' | '.join(x) if len(x) > 0 else '')

    df_maestro = df[~df['es_rechazado']].copy()
    df_rechazados = df[df['es_rechazado']].copy()

    # Limpiar columnas auxiliares del maestro
    columnas_eliminar = ['_merge', 'razones_rechazo', 'es_rechazado', 'motivo_rechazo']
    df_maestro = df_maestro.drop(columns=[c for c in columnas_eliminar if c in df_maestro.columns])

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    print(f"✓ Pacientes en dataset MAESTRO: {len(df_maestro):,}")
    print(f"✗ Pacientes RECHAZADOS: {len(df_rechazados):,}")
    print(f"  Tasa de aceptación: {len(df_maestro) / len(df) * 100:.1f}%")

    # Estadísticas del dataset maestro
    if len(df_maestro) > 0:
        print(f"\nEstadísticas del dataset maestro:")
        print(f"  - Pacientes con LOS = 0: {(df_maestro['los_cero'] == True).sum():,}")
        print(f"  - LOS promedio: {df_maestro['los_dias'].mean():.1f} días")
        print(f"  - Procedimientos promedio: {df_maestro['n_procedimientos'].mean():.1f}")
        print(f"  - Diagnósticos promedio: {df_maestro['n_diag_total'].mean():.1f}")

    return df_maestro, df_rechazados, df


# ============================================================================
# FUNCIÓN 5: GUARDAR RESULTADOS
# ============================================================================

def guardar_resultados(df_maestro, df_rechazados, df_completo):
    """
    Guarda los resultados en archivos CSV.

    Args:
        df_maestro: Dataset maestro limpio
        df_rechazados: Dataset de pacientes rechazados
        df_completo: Dataset completo con todos los pacientes
    """
    print("\n" + "=" * 70)
    print("PASO 5: Guardando resultados...")
    print("=" * 70)

    # Crear directorios si no existen
    import os
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/reports', exist_ok=True)

    # -------------------------------------------------------------------------
    # Guardar dataset maestro
    # -------------------------------------------------------------------------
    # Convertir listas a strings para exportar a CSV
    # (CSV no soporta listas nativamente)
    df_maestro_export = df_maestro.copy()

    columnas_lista = ['procedimientos', 'diagnosticos_primarios', 'diagnosticos_secundarios']
    for col in columnas_lista:
        if col in df_maestro_export.columns:
            # Convertir lista a string separado por comas
            df_maestro_export[col] = df_maestro_export[col].apply(
                lambda x: ','.join(x) if isinstance(x, list) else ''
            )

    df_maestro_export.to_csv(ARCHIVO_SALIDA_MAESTRO, index=False, sep=';')
    print(f"✓ Dataset maestro guardado: {ARCHIVO_SALIDA_MAESTRO}")
    print(f"  Registros: {len(df_maestro):,}")

    # -------------------------------------------------------------------------
    # Guardar pacientes rechazados
    # -------------------------------------------------------------------------
    if len(df_rechazados) > 0:
        df_rechazados_export = df_rechazados.copy()

        for col in columnas_lista:
            if col in df_rechazados_export.columns:
                df_rechazados_export[col] = df_rechazados_export[col].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else ''
                )

        # Seleccionar columnas relevantes para auditoría
        cols_rechazados = [
            'case_id', 'motivo_rechazo', 'los_dias',
            'n_procedimientos', 'n_diag_total',
            'fecha_ingreso', 'fecha_egreso'
        ]
        cols_rechazados = [c for c in cols_rechazados if c in df_rechazados_export.columns]

        df_rechazados_export[cols_rechazados].to_csv(
            ARCHIVO_SALIDA_RECHAZADOS,
            index=False,
            sep=';'
        )
        print(f"✓ Pacientes rechazados guardados: {ARCHIVO_SALIDA_RECHAZADOS}")
        print(f"  Registros: {len(df_rechazados):,}")

    # -------------------------------------------------------------------------
    # Guardar reporte de calidad
    # -------------------------------------------------------------------------
    # Crear un resumen ejecutivo de la limpieza
    reporte = {
        'metrica': [
            'pacientes_totales',
            'pacientes_maestro',
            'pacientes_rechazados',
            'tasa_aceptacion_pct',
            'pacientes_con_los_cero',
            'pacientes_falta_diagnosticos',
            'pacientes_falta_procedimientos',
            'pacientes_fechas_invalidas',
            'pacientes_los_negativo',
            'los_promedio_dias',
            'procedimientos_promedio',
            'diagnosticos_promedio'
        ],
        'valor': [
            len(df_completo),
            len(df_maestro),
            len(df_rechazados),
            round(len(df_maestro) / len(df_completo) * 100, 2) if len(df_completo) > 0 else 0,
            (df_maestro['los_cero'] == True).sum() if len(df_maestro) > 0 else 0,
            (df_completo['_merge'] == 'right_only').sum(),
            (df_completo['_merge'] == 'left_only').sum(),
            df_completo['fechas_invalidas'].fillna(False).sum(),
            df_completo['los_negativo'].fillna(False).sum(),
            round(df_maestro['los_dias'].mean(), 2) if len(df_maestro) > 0 else 0,
            round(df_maestro['n_procedimientos'].mean(), 2) if len(df_maestro) > 0 else 0,
            round(df_maestro['n_diag_total'].mean(), 2) if len(df_maestro) > 0 else 0
        ]
    }

    df_reporte = pd.DataFrame(reporte)
    df_reporte.to_csv(ARCHIVO_SALIDA_REPORTE, index=False, sep=';')
    print(f"✓ Reporte de calidad guardado: {ARCHIVO_SALIDA_REPORTE}")

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline de limpieza.
    """
    print("\n")
    print("█" * 70)
    print("  PIPELINE DE LIMPIEZA DE DATOS - PREDICCIÓN LOS HOSPITALARIO")
    print("█" * 70)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Paso 1: Cargar datos crudos
    df_diag_raw, df_proc_raw = cargar_datos_crudos()

    # Paso 2: Limpiar diagnósticos
    df_diag_paciente = limpiar_diagnosticos(df_diag_raw)

    # Paso 3: Limpiar procedimientos
    df_proc_paciente = limpiar_procedimientos(df_proc_raw)

    # Paso 4: Integrar datos
    df_maestro, df_rechazados, df_completo = integrar_datos(
        df_diag_paciente,
        df_proc_paciente
    )

    # Paso 5: Guardar resultados
    guardar_resultados(df_maestro, df_rechazados, df_completo)

    print("\n>>> Todos los archivos han sido generados correctamente <<<\n")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
