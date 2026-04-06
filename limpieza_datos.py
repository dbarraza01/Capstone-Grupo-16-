"""
Script de Limpieza de Datos para Predicción de LOS (Length of Stay) Hospitalario
==================================================================================

Este script procesa datos de diagnósticos y procedimientos de pacientes,
los limpia, valida y genera un dataset maestro para modelado predictivo.

Autor: Sistema de Limpieza LOS
Fecha: 2026-03-18
"""

import pandas as pd
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
ARCHIVO_SALIDA_CASO_DIAGNOSTICO = "data/processed/caso_diagnostico.csv"
ARCHIVO_SALIDA_CASO_PROCEDIMIENTO = "data/processed/caso_procedimiento.csv"
ARCHIVO_SALIDA_REPORTE = "data/reports/reporte_limpieza.csv"

# Patrones REGEX para códigos ICD-10-CM válidos
# ICD-10-CM (Clinical Modification) para DIAGNÓSTICOS - VALIDACIÓN DE FORMA (sin punto decimal)
# Longitud total permitida: 3 a 7 caracteres
# Primer carácter: letra A-Z
# Segundo carácter: dígito obligatorio (0-9)
# Tercer carácter: dígito o letra (0-9 o A-Z)
# Hasta 4 caracteres alfanuméricos adicionales opcionales
# Ejemplos válidos: E6601, S72302E, Z6841, I10, U071, A1234B
# Ejemplos inválidos: DDDDDD, AAAAAA, E6 (muy corto), 123 (no comienza con letra)
ICD10_CM_REGEX = r'^[A-Z][0-9][0-9A-Z][0-9A-Z]{0,4}$'

# ICD-10-PCS (Procedure Coding System) para PROCEDIMIENTOS - VALIDACIÓN DE FORMA SINTÁCTICA
# Longitud exacta: 7 caracteres
# Caracteres permitidos: números (0-9) y letras alfanuméricas excepto I y O
# Nota: esta validación es de FORMA estructural, no valida existencia en tablas oficiales PCS
# Ejemplos válidos: 0DB64Z3, 0BB64ZZ, 02H60JZ
# Ejemplos inválidos: 0DB64Z (6 chars), 0DB64Z3X (8 chars), OIIOIIO (contiene I/O prohibidas)
ICD10_PCS_REGEX = r'^[0-9A-HJ-NP-Z]{7}$'

# Valores permitidos para PrincSec (Principal/Secundario)
VALORES_PRINCSEC_VALIDOS = {'P', 'S'}

# Token especial del dataset para marcar admisiones por urgencia
TOKEN_URGENCIA = 'UUUUUU'


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
    2. Identifica códigos ICD-10-CM inválidos
    3. FILTRA REGISTROS (no pacientes): solo elimina registros diagnósticos inválidos
    4. Mantiene pacientes que tienen AL MENOS UN diagnóstico válido
    5. Captura códigos inválidos para auditoría (no para rechazo de paciente)
    6. Devuelve datos limpios a nivel paciente

    Args:
        df: DataFrame con columnas CASE, PrincSec, Diagnosis

    Returns:
        tuple: (df_paciente_aceptados, set_case_id_rechazados_vacio, dict_codigos_invalidos)
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

    # Limpiar espacios en blanco en códigos de diagnóstico y case_id
    df['diagnosis'] = df['diagnosis'].str.strip()
    df['case_id'] = df['case_id'].str.strip()
    df['princsec'] = df['princsec'].str.strip()

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
    # Validación 2: Diagnosis debe cumplir patrón ICD-10-CM o ser token de urgencia
    # -------------------------------------------------------------------------
    # Acepta códigos ICD-10-CM válidos formales O el token especial TOKEN_URGENCIA
    df['diagnosis_valido'] = (df['diagnosis'].str.match(ICD10_CM_REGEX, na=False)) | (df['diagnosis'] == TOKEN_URGENCIA)
    invalidos_diagnosis = (~df['diagnosis_valido']).sum()
    print(f"Registros con código ICD-10-CM inválido: {invalidos_diagnosis:,}")

    # Mostrar ejemplos de códigos inválidos (para auditoría)
    if invalidos_diagnosis > 0:
        codigos_invalidos_unicos = df[~df['diagnosis_valido']]['diagnosis'].unique()
        print(f"  Ejemplos de códigos inválidos: {codigos_invalidos_unicos[:10]}")

    # -------------------------------------------------------------------------
    # PARA AUDITORÍA: Capturar TODOS los códigos inválidos encontrados
    # -------------------------------------------------------------------------
    # Nota: En Opción A, filtramos REGISTROS (no pacientes)
    # Entonces registramos inválidos solo para auditoría/reporte, no para rechazo de paciente
    df_invalidos = df[~df['diagnosis_valido']]

    # Crear diccionario: case_id -> lista de códigos inválidos (para auditoría)
    dict_codigos_invalidos = {}
    for case_id in df_invalidos['case_id'].unique():
        codigos = df_invalidos[df_invalidos['case_id'] == case_id]['diagnosis'].unique()
        dict_codigos_invalidos[case_id] = list(codigos)

    # En Opción A: NO rechazamos pacientes por códigos inválidos
    # Solo registramos los inválidos encontrados
    set_case_id_rechazados = set()  # VACÍO para Opción A

    if len(dict_codigos_invalidos) > 0:
        print(f"\n⚠ Códigos diagnósticos inválidos encontrados en {len(dict_codigos_invalidos):,} pacientes")
        print(f"  (Estos registros serán FILTRADOS pero los pacientes NO serán rechazados)")
        print(f"  (Total de registros inválidos a eliminar: {len(df_invalidos):,})")

    # -------------------------------------------------------------------------
    # Filtrar solo registros que cumplan AMBAS validaciones
    # -------------------------------------------------------------------------
    df_valido = df[df['princsec_valido'] & df['diagnosis_valido']].copy()

    registros_eliminados = len(df) - len(df_valido)
    print(f"\n✓ Registros eliminados por validación: {registros_eliminados:,}")
    print(f"✓ Registros válidos conservados: {len(df_valido):,}")

    # -------------------------------------------------------------------------
    # Agregación a nivel PACIENTE (solo para validados)
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

    print(f"✓ Pacientes únicos con ALL diagnósticos válidos: {len(df_paciente):,}")
    print(f"  - Con diagnóstico primario: {df_paciente['tiene_diag_primario'].sum():,}")
    print(f"  - Solo diagnósticos secundarios: {(~df_paciente['tiene_diag_primario']).sum():,}")

    return df_paciente, set_case_id_rechazados, dict_codigos_invalidos


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

    # Limpiar espacios en blanco
    df['case_id'] = df['case_id'].str.strip()
    df['procedimiento'] = df['procedimiento'].str.strip()
    df['fecha_ingreso'] = df['fecha_ingreso'].str.strip()
    df['fecha_egreso'] = df['fecha_egreso'].str.strip()

    total_registros = len(df)
    print(f"Registros totales: {total_registros:,}")

    # -------------------------------------------------------------------------
    # Validación de códigos ICD-10-PCS
    # -------------------------------------------------------------------------
    print("\nValidando códigos ICD-10-PCS...")

    # Validar solo códigos no vacíos
    df['procedimiento_valido'] = (
        (df['procedimiento'].notna()) &
        (df['procedimiento'] != '') &
        (df['procedimiento'].str.match(ICD10_PCS_REGEX, na=False))
    )

    invalidos_procedimiento = (~df['procedimiento_valido']).sum()
    print(f"Registros con código ICD-10-PCS inválido: {invalidos_procedimiento:,}")

    # Mostrar ejemplos de códigos inválidos (para auditoría)
    if invalidos_procedimiento > 0:
        codigos_invalidos_unicos = df[~df['procedimiento_valido']]['procedimiento'].unique()
        print(f"  Ejemplos de códigos inválidos: {codigos_invalidos_unicos[:10]}")

    # Filtrar solo registros con códigos válidos
    df_valido = df[df['procedimiento_valido']].copy()
    registros_eliminados = len(df) - len(df_valido)
    print(f"\n✓ Registros eliminados por validación: {registros_eliminados:,}")
    print(f"✓ Registros válidos conservados: {len(df_valido):,}")

    # Actualizar df para trabajar solo con válidos
    df = df_valido

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

    # Crear lista de procedimientos válidos
    # Nota: ya filtramos por procedimiento_valido arriba, así que todos son válidos
    procedimientos_por_paciente = (
        df.groupby('case_id')['procedimiento']
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
    print(f"\n Pacientes únicos: {len(df_paciente):,}")
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

def integrar_datos(df_diagnosticos_paciente, df_procedimientos_paciente, set_case_id_rechazados, dict_codigos_invalidos):
    """
    Integra los datos de diagnósticos y procedimientos en un dataset maestro.

    Regla de rechazo (Opción A - Filtrado de registros, no de pacientes):
    - Registros diagnósticos inválidos se FILTRARON previamente
    - Pacientes con solo registros inválidos fueron excluidos automáticamente
    - Criterios de rechazo a nivel PACIENTE (integración):
      * Pacientes que están en un archivo pero NO en el otro
      * Pacientes con fechas inválidas (NaT)
      * Pacientes con LOS negativo (fecha egreso < fecha ingreso)
    - NO se rechazan pacientes por LOS=0 (son datos válidos)

    Args:
        df_diagnosticos_paciente: DataFrame de diagnósticos agregados (ya filtrados)
        df_procedimientos_paciente: DataFrame de procedimientos agregados
        set_case_id_rechazados: Set vacío (Opción A no rechaza por códigos inválidos)
        dict_codigos_invalidos: Dict mapeando case_id -> lista de códigos inválidos (auditoría)

    Returns:
        tuple: (df_maestro, df_rechazados, df_completo, dict_codigos_invalidos)
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

    # Crear bandera de urgencia (es_urgencia = 1 si contiene TOKEN_URGENCIA, 0 si no)
    def tiene_codigo_urgencia(lista_diag_primarios, lista_diag_secundarios):
        """Verifica si el paciente tiene token TOKEN_URGENCIA en cualquiera de sus diagnósticos"""
        todos_diags = lista_diag_primarios + lista_diag_secundarios
        return 1 if TOKEN_URGENCIA in todos_diags else 0

    df['es_urgencia'] = df.apply(
        lambda row: tiene_codigo_urgencia(
            row['diagnosticos_primarios'],
            row['diagnosticos_secundarios']
        ),
        axis=1
    )

    # -------------------------------------------------------------------------
    # Criterios de RECHAZO (Opción A: sin rechazo por códigos inválidos)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Aplicando criterios de rechazo...")
    print("-" * 70)

    # Lista de razones de rechazo para cada paciente
    df['razones_rechazo'] = [[] for _ in range(len(df))]

    # NOTA: En Opción A, NO rechazamos por "contiene_codigo_diagnostico_invalido"
    # Los registros inválidos fueron filtrados en limpiar_diagnosticos()

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

    # CRITERIOS EXPLÍCITAMENTE NO APLICADOS (Opción A):
    print(f"\n  ✓ Registros con códigos diagnósticos inválidos: FILTRADOS (no rechazan paciente)")
    print(f"  ✓ LOS = 0: NO rechazados (son datos válidos)")

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

    return df_maestro, df_rechazados, df, dict_codigos_invalidos



# ============================================================================
# FUNCIÓN 5: GENERAR CASO_DIAGNOSTICO GRANULAR
# ============================================================================

def generar_caso_diagnostico(df_diag_raw, set_case_id_rechazados):
    """
    Genera archivo granular de diagnósticos (una fila por diagnóstico).

    Proceso:
    1. Limpia datos (strip espacios)
    2. Valida códigos ICD-10-CM
    3. Excluye pacientes rechazados (con AAAAAA u otros inválidos)
    4. Extrae características: primera letra (d_caract_1) y primeros 3 chars (d_caract_3)

    Args:
        df_diag_raw: DataFrame crudo de diagnósticos (datos_diagnostico.csv)
        set_case_id_rechazados: Set de case_id que fueron rechazados

    Returns:
        DataFrame con columnas: case_id, d_code, tipo_d, d_caract_1, d_caract_3
    """
    print("\n" + "=" * 70)
    print("GENERANDO caso_diagnostico.csv (diagnósticos granulares)...")
    print("=" * 70)

    # Crear copia
    df = df_diag_raw.copy()

    # Estandarizar columnas
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'case': 'case_id'})

    # Limpiar espacios
    df['case_id'] = df['case_id'].str.strip()
    df['diagnosis'] = df['diagnosis'].str.strip()
    df['princsec'] = df['princsec'].str.strip()

    print(f"Registros totales: {len(df):,}")

    # Validar códigos ICD-10-CM (incluida aceptación de token de urgencia)
    df['diagnosis_valido'] = (df['diagnosis'].str.match(ICD10_CM_REGEX, na=False)) | (df['diagnosis'] == TOKEN_URGENCIA)
    df['princsec_valido'] = df['princsec'].isin(VALORES_PRINCSEC_VALIDOS)

    # Filtrar solo registros válidos
    df_valido = df[df['diagnosis_valido'] & df['princsec_valido']].copy()
    print(f"Registros con códigos válidos: {len(df_valido):,}")

    # Excluir pacientes rechazados
    df_valido = df_valido[~df_valido['case_id'].isin(set_case_id_rechazados)].copy()
    print(f"Registros después de excluir pacientes rechazados: {len(df_valido):,}")

    # Renombrar columnas para output
    df_valido = df_valido.rename(columns={
        'diagnosis': 'd_code',
        'princsec': 'tipo_d'
    })

    # Extraer características del código diagnóstico
    df_valido['d_caract_1'] = df_valido['d_code'].str[0]  # Primera letra
    df_valido['d_caract_3'] = df_valido['d_code'].str[:3]  # Primeros 3 caracteres

    # Seleccionar columnas finales
    df_output = df_valido[['case_id', 'd_code', 'tipo_d', 'd_caract_1', 'd_caract_3']].copy()

    print(f"✓ Diagnósticos únicos generados: {len(df_output):,}")
    print(f"  - Pacientes únicos: {df_output['case_id'].nunique():,}")
    print(f"  - Diagnósticos primarios: {(df_output['tipo_d'] == 'P').sum():,}")
    print(f"  - Diagnósticos secundarios: {(df_output['tipo_d'] == 'S').sum():,}")

    return df_output


# ============================================================================
# FUNCIÓN 6: GENERAR CASO_PROCEDIMIENTO GRANULAR
# ============================================================================

def generar_caso_procedimiento(df_proc_raw):
    """
    Genera archivo granular de procedimientos (una fila por procedimiento).

    Proceso:
    1. Limpia datos (strip espacios)
    2. Valida códigos ICD-10-PCS
    3. Extrae características: primera letra (p_caract_1) y primeros 3 chars (p_caract_3)

    Args:
        df_proc_raw: DataFrame crudo de procedimientos (procedimiento_pacientes.csv)

    Returns:
        DataFrame con columnas: case_id, p_code, p_caract_1, p_caract_3
    """
    print("\n" + "=" * 70)
    print("GENERANDO caso_procedimiento.csv (procedimientos granulares)...")
    print("=" * 70)

    # Crear copia
    df = df_proc_raw.copy()

    # Estandarizar columnas
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'case': 'case_id',
        'procedure': 'procedimiento'
    })

    # Limpiar espacios
    df['case_id'] = df['case_id'].str.strip()
    df['procedimiento'] = df['procedimiento'].str.strip()

    print(f"Registros totales: {len(df):,}")

    # Validar códigos ICD-10-PCS
    df['procedimiento_valido'] = (
        (df['procedimiento'].notna()) &
        (df['procedimiento'] != '') &
        (df['procedimiento'].str.match(ICD10_PCS_REGEX, na=False))
    )

    # Filtrar solo registros válidos
    df_valido = df[df['procedimiento_valido']].copy()
    print(f"Registros con códigos válidos: {len(df_valido):,}")

    # Renombrar columnas para output
    df_valido = df_valido.rename(columns={'procedimiento': 'p_code'})

    # Extraer características del código de procedimiento
    df_valido['p_caract_1'] = df_valido['p_code'].str[0]  # Primera letra (tipo de procedimiento)
    df_valido['p_caract_3'] = df_valido['p_code'].str[:3]  # Primeros 3 caracteres

    # Seleccionar columnas finales
    df_output = df_valido[['case_id', 'p_code', 'p_caract_1', 'p_caract_3']].copy()

    print(f"✓ Procedimientos únicos generados: {len(df_output):,}")
    print(f"  - Pacientes únicos: {df_output['case_id'].nunique():,}")

    return df_output


# ============================================================================
# FUNCIÓN 7: GUARDAR RESULTADOS
# ============================================================================

def guardar_resultados(df_maestro, df_rechazados, df_completo, dict_codigos_invalidos,
                       df_caso_diagnostico, df_caso_procedimiento):
    """
    Guarda los resultados en archivos CSV.

    Args:
        df_maestro: Dataset maestro limpio
        df_rechazados: Dataset de pacientes rechazados
        df_completo: Dataset completo con todos los pacientes
        dict_codigos_invalidos: Dict mapeando case_id -> lista de códigos diagnósticos inválidos
        df_caso_diagnostico: DataFrame granular de diagnósticos
        df_caso_procedimiento: DataFrame granular de procedimientos
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

    # Ordenar columnas para que los_dias esté después de fecha_egreso y es_urgencia después
    columnas_ordenadas = [
        'case_id',
        'fecha_ingreso',
        'fecha_egreso',
        'los_dias',
        'es_urgencia',
        'procedimientos',
        'n_procedimientos',
        'diagnosticos_primarios',
        'diagnosticos_secundarios',
        'n_diag_primarios',
        'n_diag_secundarios',
        'n_diag_total',
        'tiene_diag_primario',
        'fechas_invalidas',
        'los_negativo',
        'los_cero'
    ]

    # Filtrar solo las columnas que existen en el DataFrame
    columnas_finales = [col for col in columnas_ordenadas if col in df_maestro_export.columns]

    df_maestro_export[columnas_finales].to_csv(ARCHIVO_SALIDA_MAESTRO, index=False, sep=';')
    print(f"✓ Dataset maestro guardado: {ARCHIVO_SALIDA_MAESTRO}")
    print(f"  Registros: {len(df_maestro):,}")
    print(f"  Columnas: {columnas_finales}")

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

        # Agregar códigos diagnósticos inválidos para pacientes rechazados
        def obtener_codigos_invalidos(case_id):
            if case_id in dict_codigos_invalidos:
                return ','.join(dict_codigos_invalidos[case_id])
            return ''

        df_rechazados_export['codigos_diagnostico_invalidos'] = (
            df_rechazados_export['case_id'].apply(obtener_codigos_invalidos)
        )

        # Seleccionar columnas relevantes para auditoría
        cols_rechazados = [
            'case_id', 'motivo_rechazo', 'codigos_diagnostico_invalidos',
            'los_dias', 'n_procedimientos', 'n_diag_total',
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
    # Guardar caso_diagnostico (diagnósticos granulares)
    # -------------------------------------------------------------------------
    df_caso_diagnostico.to_csv(ARCHIVO_SALIDA_CASO_DIAGNOSTICO, index=False, sep=';')
    print(f"✓ Caso diagnóstico guardado: {ARCHIVO_SALIDA_CASO_DIAGNOSTICO}")
    print(f"  Registros: {len(df_caso_diagnostico):,}")

    # -------------------------------------------------------------------------
    # Guardar caso_procedimiento (procedimientos granulares)
    # -------------------------------------------------------------------------
    df_caso_procedimiento.to_csv(ARCHIVO_SALIDA_CASO_PROCEDIMIENTO, index=False, sep=';')
    print(f"✓ Caso procedimiento guardado: {ARCHIVO_SALIDA_CASO_PROCEDIMIENTO}")
    print(f"  Registros: {len(df_caso_procedimiento):,}")

    # -------------------------------------------------------------------------
    # Guardar reporte de calidad
    # -------------------------------------------------------------------------
    # Crear un resumen ejecutivo de la limpieza

    # Contar rechazados por motivo
    n_invalidos_diag = 0  # Opción A: no rechaza por códigos inválidos
    n_falta_diag = (df_rechazados['motivo_rechazo'].str.contains('falta_en_diagnosticos', na=False)).sum() if len(df_rechazados) > 0 else 0
    n_falta_proc = (df_rechazados['motivo_rechazo'].str.contains('falta_en_procedimientos', na=False)).sum() if len(df_rechazados) > 0 else 0
    n_fechas_invalidas = (df_rechazados['motivo_rechazo'].str.contains('fechas_invalidas', na=False)).sum() if len(df_rechazados) > 0 else 0
    n_los_negativo = (df_rechazados['motivo_rechazo'].str.contains('los_negativo', na=False)).sum() if len(df_rechazados) > 0 else 0

    reporte = {
        'metrica': [
            'pacientes_totales',
            'pacientes_maestro',
            'pacientes_rechazados',
            'tasa_aceptacion_pct',
            'pacientes_con_los_cero',
            'registros_diagnostico_filtrados_por_codigo_invalido',
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
            len(dict_codigos_invalidos),  # Auditoría: número de pacientes con registros inválidos
            n_falta_diag,
            n_falta_proc,
            n_fechas_invalidas,
            n_los_negativo,
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
    df_diag_paciente, set_case_id_rechazados, dict_codigos_invalidos = limpiar_diagnosticos(df_diag_raw)

    # Paso 3: Limpiar procedimientos
    df_proc_paciente = limpiar_procedimientos(df_proc_raw)

    # Paso 4: Integrar datos (ahora incluye rechazo de códigos inválidos y es_urgencia)
    df_maestro, df_rechazados, df_completo, dict_codigos_invalidos = integrar_datos(
        df_diag_paciente,
        df_proc_paciente,
        set_case_id_rechazados,
        dict_codigos_invalidos
    )

    # Paso 5: Generar archivos granulares
    df_caso_diagnostico = generar_caso_diagnostico(df_diag_raw, set_case_id_rechazados)
    df_caso_procedimiento = generar_caso_procedimiento(df_proc_raw)

    # Paso 6: Guardar resultados
    guardar_resultados(
        df_maestro,
        df_rechazados,
        df_completo,
        dict_codigos_invalidos,
        df_caso_diagnostico,
        df_caso_procedimiento
    )

    print("\n>>> Todos los archivos han sido generados correctamente <<<\n")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
