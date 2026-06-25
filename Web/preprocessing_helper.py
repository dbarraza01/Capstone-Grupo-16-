import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configurar rutas relativas
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
DIAG_CSV = BASE_DIR / "data" / "processed" / "caso_diagnostico.csv"
PROC_CSV = BASE_DIR / "data" / "processed" / "caso_procedimiento.csv"

# Parámetros y constantes
UMBRAL_DIAG = 20
UMBRAL_PROC_CODE = 10
UMBRAL_PROC_CAT3 = 20
CODIGO_URGENCIA = "UUUUUU"

# Inicializar diccionarios de frecuencia vacíos
freq_d_code = {}
freq_d_3 = {}
freq_d_1 = {}
freq_p_code = {}
freq_p_3 = {}
freq_p_1 = {}

def inicializar_frecuencias():
    """Carga los CSVs históricos y calcula las frecuencias para el mapeo jerárquico."""
    global freq_d_code, freq_d_3, freq_d_1
    global freq_p_code, freq_p_3, freq_p_1
    
    print("Precomputando frecuencias de códigos para el mapeo jerárquico...")
    try:
        # Cargar datos diagnósticos
        df_diag = pd.read_csv(DIAG_CSV, sep=";", dtype=str)
        for col in ['d_code', 'd_caract_3', 'd_caract_1']:
            if col in df_diag.columns:
                df_diag[col] = df_diag[col].astype(str).str.strip().str.upper()
        
        # Excluir UUUUUU
        df_diag_clean = df_diag[df_diag['d_code'] != CODIGO_URGENCIA].copy()
        
        freq_d_code = df_diag_clean.groupby('d_code')['case_id'].nunique().to_dict()
        freq_d_3 = df_diag_clean.groupby('d_caract_3')['case_id'].nunique().to_dict()
        freq_d_1 = df_diag_clean.groupby('d_caract_1')['case_id'].nunique().to_dict()
        
        # Cargar datos procedimientos
        df_proc = pd.read_csv(PROC_CSV, sep=";", dtype=str)
        for col in ['p_code', 'p_caract_3', 'p_caract_1']:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].astype(str).str.strip().str.upper()
                
        freq_p_code = df_proc.groupby('p_code')['case_id'].nunique().to_dict()
        freq_p_3 = df_proc.groupby('p_caract_3')['case_id'].nunique().to_dict()
        freq_p_1 = df_proc.groupby('p_caract_1')['case_id'].nunique().to_dict()
        
        print(f"Frecuencias cargadas correctamente. Diagnósticos únicos: {len(freq_d_code)}, Procedimientos únicos: {len(freq_p_code)}")
    except Exception as e:
        print(f"ERROR al inicializar frecuencias: {e}")

# Mapeo de códigos individuales
def mapear_codigo_diagnostico(code):
    code = str(code).strip().upper().replace(".", "")
    if not code or code == CODIGO_URGENCIA:
        return None
    cat3 = code[:3]
    cap1 = code[0]
    
    if freq_d_code.get(code, 0) >= UMBRAL_DIAG:
        return f"diag_{code}"
    elif freq_d_3.get(cat3, 0) >= UMBRAL_DIAG:
        return f"diag_{cat3}"
    else:
        return f"diag_rare_cap_{cap1}"

def mapear_codigo_procedimiento(code):
    code = str(code).strip().upper().replace(".", "")
    if not code:
        return None
    cat3 = code[:3]
    sec1 = code[0]
    
    if freq_p_code.get(code, 0) >= UMBRAL_PROC_CODE:
        return f"proc_{code}"
    elif freq_p_3.get(cat3, 0) >= UMBRAL_PROC_CAT3:
        return f"proc_{cat3}"
    else:
        return f"proc_rare_sec_{sec1}"

# Inicializar frecuencias de inmediato al cargar el módulo
inicializar_frecuencias()

def calcular_charlson(codigos_diag):
    """Calcula el indice de Charlson usando comorbidipy.

    comorbidipy 0.8+ usa Polars y requiere Python 3.13. Para desarrollo local
    con Python 3.11 usamos la rama 0.5.x, que trabaja con pandas. La funcion
    soporta ambos formatos para que la app no dependa de una version unica.
    """
    if not codigos_diag:
        return 0
    try:
        # Compatibilidad: comorbidipy 0.5.x importa SettingWithCopyWarning desde
        # pandas.core.common, ruta eliminada en pandas 3.
        try:
            import pandas.core.common as pandas_common

            if not hasattr(pandas_common, "SettingWithCopyWarning"):
                class SettingWithCopyWarning(Warning):
                    pass

                pandas_common.SettingWithCopyWarning = SettingWithCopyWarning
        except Exception:
            pass

        from comorbidipy import comorbidity
        
        # Limpiar códigos
        codigos_limpios = list(set([str(c).strip().upper().replace(".", "") for c in codigos_diag if str(c).strip()]))
        if not codigos_limpios:
            return 0

        df_codes = pd.DataFrame({
            "id": ["P_TEMP"] * len(codigos_limpios),
            "code": codigos_limpios,
        })

        try:
            df_res = comorbidity(
                df_codes,
                id="id",
                code="code",
                age=None,
                score="charlson",
                icd="icd10",
                variant="quan",
                weighting="quan",
            )
        except TypeError:
            import polars as pl

            df_res = comorbidity(
                pl.from_pandas(df_codes),
                id_col="id",
                code_col="code",
                age_col=None,
                score="charlson",
                icd="icd10",
                variant="quan",
            )

        if hasattr(df_res, "height"):
            if df_res.height > 0 and "comorbidity_score" in df_res.columns:
                return int(df_res["comorbidity_score"][0])
        elif not df_res.empty and "comorbidity_score" in df_res.columns:
            return int(df_res["comorbidity_score"].iloc[0])
        return 0
    except Exception as e:
        print(f"Error al calcular Charlson index: {e}. Usando fallback a 0.")
        return 0

def construir_vector_paciente(diag_primario, diags_secundarios, procedimientos, es_urgencia, fecha_ingreso=None, columnas_modelo=None):
    """
    Construye el DataFrame con las variables y transformaciones requeridas para los modelos de Machine Learning (XGBoost y Random Forest).
    Espera una lista de nombres de columnas (1,651 columnas) para ordenar y rellenar.
    """
    if columnas_modelo is None:
        raise ValueError("Se debe proporcionar la lista de columnas esperadas del modelo.")
        
    # Limpieza
    diag_primario = str(diag_primario).strip().upper() if pd.notna(diag_primario) else ""
    diags_secundarios = [str(d).strip().upper() for d in diags_secundarios if pd.notna(d) and str(d).strip()]
    procedimientos = [str(p).strip().upper() for p in procedimientos if pd.notna(p) and str(p).strip()]
    
    # Lista consolidada de diagnósticos
    todos_diagnosticos = []
    if diag_primario and diag_primario != CODIGO_URGENCIA:
        todos_diagnosticos.append(diag_primario)
    todos_diagnosticos.extend([d for d in diags_secundarios if d != CODIGO_URGENCIA])
    
    # 1. Variables Base
    n_procedimientos = len(procedimientos)
    n_diag_primarios = 1 if (diag_primario and diag_primario != CODIGO_URGENCIA) else 0
    n_diag_secundarios = len([d for d in diags_secundarios if d != CODIGO_URGENCIA])
    n_diag_total = n_diag_primarios + n_diag_secundarios
    tiene_diag_primario = 1 if n_diag_primarios > 0 else 0
    
    # Fechas
    if fecha_ingreso is None:
        fecha_ingreso = pd.Timestamp.now()
    else:
        fecha_ingreso = pd.to_datetime(fecha_ingreso, errors='coerce')
        if pd.isna(fecha_ingreso):
            fecha_ingreso = pd.Timestamp.now()
            
    mes_ingreso = int(fecha_ingreso.month)
    dia_semana_ingreso = int(fecha_ingreso.dayofweek)
    
    # 2. Features de repetición / carga
    # Diagnósticos
    total_registros_diag = len(todos_diagnosticos)
    codigos_unicos_diag = len(set(todos_diagnosticos))
    n_diag_codigos_repetidos = total_registros_diag - codigos_unicos_diag
    
    # Mapeo a grupos diagnósticos
    grupos_diag_mapped = [mapear_codigo_diagnostico(d) for d in todos_diagnosticos]
    grupos_diag_mapped = [g for g in grupos_diag_mapped if g is not None]
    
    grupos_unicos_diag = len(set(grupos_diag_mapped))
    
    if grupos_diag_mapped:
        # Frecuencia máxima de un grupo
        max_repeticion_diag_grupo = max([grupos_diag_mapped.count(g) for g in set(grupos_diag_mapped)])
    else:
        max_repeticion_diag_grupo = 0
        
    # Procedimientos
    total_registros_proc = len(procedimientos)
    codigos_unicos_proc = len(set(procedimientos))
    n_proc_codigos_repetidos = total_registros_proc - codigos_unicos_proc
    
    # Mapeo a grupos de procedimiento
    grupos_proc_mapped = [mapear_codigo_procedimiento(p) for p in procedimientos]
    grupos_proc_mapped = [g for g in grupos_proc_mapped if g is not None]
    
    grupos_unicos_proc = len(set(grupos_proc_mapped))
    
    if grupos_proc_mapped:
        max_repeticion_proc_grupo = max([grupos_proc_mapped.count(g) for g in set(grupos_proc_mapped)])
    else:
        max_repeticion_proc_grupo = 0
        
    # 3. Índice de Charlson
    charlson_index = calcular_charlson(todos_diagnosticos)
    
    # 4. Crear diccionario de fila inicializado en 0
    row_dict = {col: 0 for col in columnas_modelo}
    
    # Asignar variables base
    row_dict['es_urgencia'] = int(es_urgencia)
    row_dict['n_procedimientos'] = n_procedimientos
    row_dict['n_diag_primarios'] = n_diag_primarios
    row_dict['n_diag_secundarios'] = n_diag_secundarios
    row_dict['n_diag_total'] = n_diag_total
    row_dict['tiene_diag_primario'] = tiene_diag_primario
    row_dict['mes_ingreso'] = mes_ingreso
    row_dict['dia_semana_ingreso'] = dia_semana_ingreso
    
    row_dict['n_diag_codigos_repetidos'] = n_diag_codigos_repetidos
    row_dict['grupos_unicos_diag'] = grupos_unicos_diag
    row_dict['max_repeticion_diag_grupo'] = max_repeticion_diag_grupo
    row_dict['n_proc_codigos_repetidos'] = n_proc_codigos_repetidos
    row_dict['grupos_unicos_proc'] = grupos_unicos_proc
    row_dict['max_repeticion_proc_grupo'] = max_repeticion_proc_grupo
    row_dict['charlson_index'] = charlson_index
    
    # Activar columnas binarias mapeadas que estén presentes en las columnas del modelo
    for col_diag in set(grupos_diag_mapped):
        if col_diag in row_dict:
            row_dict[col_diag] = 1
            
    for col_proc in set(grupos_proc_mapped):
        if col_proc in row_dict:
            row_dict[col_proc] = 1
            
    # Crear DataFrame con una sola fila ordenada por las columnas del modelo
    df_vector = pd.DataFrame([row_dict], columns=columnas_modelo)
    return df_vector

def construir_vector_paciente_lr(diag_primario, diags_secundarios, procedimientos, es_urgencia, fecha_ingreso=None, columnas_modelo=None, es_urgente_modelo=True):
    """
    Construye el DataFrame con las variables requeridas para el modelo base de regresion lineal.
    Si un bundle antiguo pidiera interacciones, tambien puede calcularlas por compatibilidad.
    """
    if columnas_modelo is None:
        raise ValueError("Se debe proporcionar la lista de columnas esperadas del modelo base.")
        
    # Obtener el vector base (del mismo tamaño que el modelo de ML, para reutilizar la lógica)
    # Usamos una lista de columnas base construida a partir de la unión de columnas comunes
    # Creamos un vector base con todas las variables diagnósticas y procedimentales.
    # Para ello, cargamos la lista de columnas del modelo de ML para tener la plantilla
    with open(WEB_DIR / "columnas_modelo_final.pkl", "rb") as f:
        xgb_cols = pickle.load(f)
        
    df_base = construir_vector_paciente(
        diag_primario=diag_primario,
        diags_secundarios=diags_secundarios,
        procedimientos=procedimientos,
        es_urgencia=es_urgencia,
        fecha_ingreso=fecha_ingreso,
        columnas_modelo=xgb_cols
    )
    
    # Extraer valores numéricos para las interacciones
    charlson_val = int(df_base['charlson_index'].iloc[0])
    diag_total_val = int(df_base['n_diag_total'].iloc[0])
    proc_val = int(df_base['n_procedimientos'].iloc[0])
    
    # Calcular interacciones
    int_charlson_diag = charlson_val * diag_total_val
    int_proc_diag = proc_val * diag_total_val
    int_charlson_proc = charlson_val * proc_val
    
    # Crear diccionario para el vector del modelo lineal
    row_dict_lr = {}
    for col in columnas_modelo:
        if col == 'int_charlson_diag':
            row_dict_lr[col] = int_charlson_diag
        elif col == 'int_proc_diag':
            row_dict_lr[col] = int_proc_diag
        elif col == 'int_charlson_proc':
            row_dict_lr[col] = int_charlson_proc
        elif col in df_base.columns:
            row_dict_lr[col] = df_base[col].iloc[0]
        else:
            row_dict_lr[col] = 0
            
    df_lr = pd.DataFrame([row_dict_lr], columns=columnas_modelo)
    return df_lr
