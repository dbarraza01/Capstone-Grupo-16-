import os
import sys
import uuid
import pickle
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path

# Configurar rutas y directorios
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR = BASE_DIR / "ml_operacional_entrega3" / "reports"

# Importar helper de preprocesamiento
sys.path.append(str(WEB_DIR))
import preprocessing_helper

app = Flask(__name__)

# Almacenamiento en memoria para resultados de lotes (bulk predictions)
batch_store = {}


def _percent(value):
    return round(float(value) * 100, 1)


def _metric_row(df, model):
    row = df[df["modelo"].astype(str).str.upper() == model.upper()]
    if row.empty:
        raise ValueError(f"No se encontro modelo {model} en metricas")
    return row.iloc[0]


def _segment_row(df, model, segment):
    row = df[
        (df["modelo"].astype(str).str.upper() == model.upper())
        & (df["segmento"].astype(str) == segment)
    ]
    if row.empty:
        raise ValueError(f"No se encontro modelo {model} segmento {segment}")
    return row.iloc[0]


def load_analytics_context():
    """Carga metricas actuales desde los reportes operacionales."""
    comparison = pd.read_csv(REPORTS_DIR / "comparacion_final_modelos.csv")
    by_segment = pd.read_csv(REPORTS_DIR / "comparacion_final_por_segmento.csv")
    xgb_tramos = pd.read_csv(REPORTS_DIR / "metricas_por_tramo_holdout_xgb.csv")
    rf_tramos = pd.read_csv(REPORTS_DIR / "metricas_por_tramo_holdout_rf.csv")

    xgb = _metric_row(comparison, "XGB")
    rf = _metric_row(comparison, "RF")
    lr_programado = _segment_row(by_segment, "LR", "programado")
    lr_urgente = _segment_row(by_segment, "LR", "urgente")

    error_bands = [
        {
            "label": "Error Abs <= 1 Dia",
            "xgb": _percent(xgb["pct_error_abs_le_1d"]),
            "rf": _percent(rf["pct_error_abs_le_1d"]),
        },
        {
            "label": "Error Abs <= 3 Dias",
            "xgb": _percent(xgb["pct_error_abs_le_3d"]),
            "rf": _percent(rf["pct_error_abs_le_3d"]),
        },
        {
            "label": "Error Abs <= 7 Dias",
            "xgb": _percent(xgb["pct_error_abs_le_7d"]),
            "rf": _percent(rf["pct_error_abs_le_7d"]),
        },
    ]

    tramo_labels = {
        "0-2": "Tramo 0-2 dias (Corto)",
        "3-6": "Tramo 3-6 dias (Medio)",
        "7-13": "Tramo 7-13 dias (Pre-PLOS)",
        "14+ (PLOS)": "Tramo 14+ dias (PLOS)",
    }
    subestimacion = []
    for _, row in xgb_tramos.iterrows():
        tramo = str(row["tramo"])
        subestimacion.append(
            {
                "label": tramo_labels.get(tramo, tramo),
                "value": _percent(row["pup"]),
                "bar_class": (
                    "bg-[#2e7d32]"
                    if tramo == "0-2"
                    else "bg-primary/50"
                    if tramo in {"3-6", "7-13"}
                    else "bg-error"
                ),
                "text_class": (
                    "text-[#2e7d32]"
                    if tramo == "0-2"
                    else "text-outline-variant"
                    if tramo in {"3-6", "7-13"}
                    else "text-error"
                ),
            }
        )

    return {
        "kpis": {
            "xgb_mae": round(float(xgb["mae"]), 2),
            "xgb_f1_plos": _percent(xgb["f1_plos_14"]),
            "lr_programado_mae": round(float(lr_programado["mae"]), 2),
            "lr_urgente_mae": round(float(lr_urgente["mae"]), 2),
        },
        "error_bands": error_bands,
        "subestimacion": subestimacion,
        "table": {
            "xgb": {
                "mae": round(float(xgb["mae"]), 3),
                "rmse": round(float(xgb["rmse"]), 3),
                "medae": round(float(xgb["medae"]), 3),
                "precision": _percent(xgb["precision_plos_14"]),
                "recall": _percent(xgb["recall_plos_14"]),
            },
            "rf": {
                "mae": round(float(rf["mae"]), 3),
                "rmse": round(float(rf["rmse"]), 3),
                "medae": round(float(rf["medae"]), 3),
                "precision": _percent(rf["precision_plos_14"]),
                "recall": _percent(rf["recall_plos_14"]),
            },
        },
    }

# Carga de Modelos y Columnas en memoria al iniciar el servidor
models = {}

def cargar_modelos():
    global models
    print("Cargando modelos clínicos finales y segmentados en memoria...")
    
    MODELS_DIR = BASE_DIR / "ml_operacional_entrega3" / "modelos_guardados"
    
    try:
        for segment in ["urgente", "programado"]:
            models[f"clf_{segment}"] = joblib.load(MODELS_DIR / f"clf_xgb_{segment}.joblib")
            models[f"reg_xgb_{segment}"] = joblib.load(MODELS_DIR / f"reg_xgb_{segment}.joblib")
            models[f"reg_rf_{segment}"] = joblib.load(MODELS_DIR / f"reg_rf_{segment}.joblib")
            models[f"reg_lr_{segment}"] = joblib.load(MODELS_DIR / f"reg_lr_{segment}.joblib")
        print("Modelos clínicos finales inicializados correctamente en memoria.")
    except Exception as e:
        print(f"ERROR CRÍTICO al cargar modelos: {e}")

# Ejecutar carga de modelos
cargar_modelos()

# ============================================================================
# Rutas de Vistas
# ============================================================================

@app.route('/')
def route_individual():
    """Ruta principal: Predicción individual."""
    return render_template('individual.html')

@app.route('/bulk')
def route_bulk():
    """Ruta: Predicción masiva."""
    return render_template('bulk.html')

@app.route('/analytics')
def route_analytics():
    """Ruta: Dashboard de Analytics."""
    return render_template('analytics.html', analytics=load_analytics_context())

# ============================================================================
# Rutas de API
# ============================================================================

@app.route('/charlson', methods=['POST'])
def api_charlson():
    """Calcula dinámicamente el índice de Charlson para los diagnósticos provistos."""
    data = request.get_json() or {}
    diagnosticos = data.get('diagnosticos', [])
    score = preprocessing_helper.calcular_charlson(diagnosticos)
    return jsonify({'charlson_index': score})

@app.route('/predict', methods=['POST'])
def api_predict():
    """Realiza la predicción de LOS para un paciente individual utilizando el modelo seleccionado."""
    data = request.get_json() or {}
    
    diagnosticos = data.get('diagnosticos', [])
    procedimientos = data.get('procedimientos', [])
    es_urgencia = int(data.get('es_urgencia', 1))
    fecha_ingreso = data.get('fecha_ingreso', None)
    model_name = data.get('model_name', 'xgboost')
    
    # Separar primer diagnóstico como primario, y los demás como secundarios
    diag_primario = diagnosticos[0] if len(diagnosticos) > 0 else ""
    diags_secundarios = diagnosticos[1:] if len(diagnosticos) > 1 else []
    
    segment = "urgente" if es_urgencia == 1 else "programado"
    
    try:
        # 1. Obtener features y bundles
        clf_bundle = models[f"clf_{segment}"]
        reg_xgb_bundle = models[f"reg_xgb_{segment}"]
        reg_rf_bundle = models[f"reg_rf_{segment}"]
        reg_lr_bundle = models[f"reg_lr_{segment}"]
        
        # 2. Construir vector de entrada para la Etapa 1
        df_vector_ml = preprocessing_helper.construir_vector_paciente(
            diag_primario=diag_primario,
            diags_secundarios=diags_secundarios,
            procedimientos=procedimientos,
            es_urgencia=es_urgencia,
            fecha_ingreso=fecha_ingreso,
            columnas_modelo=clf_bundle["features"]
        )
        
        # Asegurar tipo de datos int para booleanos
        bool_cols = df_vector_ml.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            df_vector_ml[bool_cols] = df_vector_ml[bool_cols].astype(int)
            
        # 3. Predicción Etapa 1 (Clasificador de Riesgo)
        clf_model = clf_bundle["model"]
        prob_plos_14 = float(clf_model.predict_proba(df_vector_ml)[:, 1][0])
        
        # 4. Preparar vectores para Regresores de Etapa 2
        # XGBoost
        df_vector_reg_xgb = df_vector_ml.copy()
        df_vector_reg_xgb["prob_los_14"] = prob_plos_14
        df_vector_reg_xgb = df_vector_reg_xgb[reg_xgb_bundle["features"]]
        
        # Random Forest
        df_vector_reg_rf = df_vector_ml.copy()
        df_vector_reg_rf["prob_los_14"] = prob_plos_14
        df_vector_reg_rf = df_vector_reg_rf[reg_rf_bundle["features"]]
        
        # Regresion lineal base
        df_vector_lr = preprocessing_helper.construir_vector_paciente_lr(
            diag_primario=diag_primario,
            diags_secundarios=diags_secundarios,
            procedimientos=procedimientos,
            es_urgencia=es_urgencia,
            fecha_ingreso=fecha_ingreso,
            columnas_modelo=reg_lr_bundle["features"],
            es_urgente_modelo=(es_urgencia == 1)
        )
        
        # 5. Ejecutar inferencia de días (TransformedTargetRegressor revierte automáticamente el log1p)
        pred_xgb = float(reg_xgb_bundle["model"].predict(df_vector_reg_xgb)[0])
        pred_rf = float(reg_rf_bundle["model"].predict(df_vector_reg_rf)[0])
        pred_ridge = float(reg_lr_bundle["model"].predict(df_vector_lr)[0])
        
        pred_xgb = max(0.0, pred_xgb)
        pred_rf = max(0.0, pred_rf)
        pred_ridge = max(0.0, pred_ridge)
        
        # Seleccionar la predicción del modelo activo
        selected_pred = pred_xgb
        if model_name == 'random_forest':
            selected_pred = pred_rf
        elif model_name == 'ridge':
            selected_pred = pred_ridge
            
        # 6. Determinar riesgo y labels de negocio a partir de la probabilidad de Etapa 1
        risk_level = 'low'
        risk_label = f"Riesgo Bajo (Mediana: {selected_pred:.1f} días)"
        if prob_plos_14 >= 0.50:
            risk_level = 'high'
            risk_label = f"Riesgo Elevado / Estancia Prolongada ({selected_pred:.1f} días)"
        elif prob_plos_14 >= 0.35:
            risk_level = 'medium'
            risk_label = f"Riesgo Moderado ({selected_pred:.1f} días)"
            
        # 7. Calcular factores de decision dinamicos usando coeficientes lineales
        driving_factors = calcular_driving_factors(df_vector_lr, es_urgencia == 1)
        charlson_calc = int(df_vector_ml['charlson_index'].iloc[0])
        
        return jsonify({
            'predicted_los': selected_pred,
            'risk_level': risk_level,
            'risk_label': risk_label,
            'charlson_index': charlson_calc,
            'driving_factors': driving_factors,
            'comparison': {
                'xgb': pred_xgb,
                'rf': pred_rf,
                'ridge': pred_ridge
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def calcular_driving_factors(df_vector, es_urgente):
    """Calcula los aportes de cada variable clinica activa a partir de los coeficientes lineales."""
    segment = "urgente" if es_urgente else "programado"
    reg_lr_bundle = models.get(f"reg_lr_{segment}")
    if reg_lr_bundle is None:
        return []
        
    transformed_model = reg_lr_bundle["model"]
    model = transformed_model.regressor_  # LinearRegression dentro del TransformedTargetRegressor
    cols = reg_lr_bundle["features"]
    
    if model is None or not hasattr(model, 'coef_'):
        return []
        
    coefs = model.coef_
    
    contributions = []
    for col, coef in zip(cols, coefs):
        val = df_vector[col].iloc[0]
        if val != 0:
            contrib = val * coef
            contributions.append((col, val, contrib))
            
    # Ordenar por magnitud de aporte absoluto
    contributions = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)
    
    factors = []
    for col, val, contrib in contributions[:4]: # Mostrar máximo 4 factores
        direction = 'up' if contrib >= 0 else 'down'
        
        # Traducir columnas de base a nombres médicos
        factor_name = col
        desc = "Impacto en la estancia esperada del paciente"
        
        if col == 'charlson_index':
            factor_name = f"Índice de Charlson ({val})"
            desc = "Carga de comorbilidad crónica"
        elif col == 'n_diag_total':
            factor_name = f"Total de Diagnósticos ({val})"
            desc = "Complejidad por múltiples diagnósticos secundarios"
        elif col == 'n_procedimientos':
            factor_name = f"Total de Procedimientos ({val})"
            desc = "Severidad quirúrgica y cantidad de intervenciones"
        elif col == 'int_charlson_diag':
            factor_name = "Interacción Comorbilidad × Diagnósticos"
            desc = "Efecto combinado de alta gravedad basal y patología múltiple"
        elif col == 'int_charlson_proc':
            factor_name = "Interacción Comorbilidad × Procedimientos"
            desc = "Riesgo potenciado por enfermedades de base sometidas a intervención"
        elif col == 'int_proc_diag':
            factor_name = "Interacción Procedimientos × Diagnósticos"
            desc = "Asociación combinada de múltiples procedimientos y diagnósticos"
        elif col.startswith('diag_'):
            diag_code = col.replace('diag_', '')
            if 'rare_cap_' in diag_code:
                cap_letter = diag_code.replace('rare_cap_', '')
                factor_name = f"Capítulo Diagnóstico {cap_letter}"
                desc = "Grupo patológico de menor frecuencia"
            else:
                factor_name = f"Diagnóstico {diag_code}"
                desc = f"Presencia de la condición clínica: {diag_code}"
        elif col.startswith('proc_'):
            proc_code = col.replace('proc_', '')
            if 'rare_sec_' in proc_code:
                sec_letter = proc_code.replace('rare_sec_', '')
                factor_name = f"Sección Procedimiento {sec_letter}"
                desc = "Grupo quirúrgico de menor frecuencia"
            else:
                factor_name = f"Procedimiento {proc_code}"
                desc = f"Intervención procedimental codificada: {proc_code}"
                
        # Estimar impacto en días aproximados
        impact_days = np.expm1(abs(contrib))
        impact_sign = '+' if contrib >= 0 else '-'
        impact_str = f"{impact_sign}{impact_days:.1f} días"
        
        factors.append({
            'factor': factor_name,
            'description': desc,
            'impact': impact_str,
            'direction': direction
        })
        
    return factors

# ============================================================================
# Rutas de Predicción Masiva (Bulk)
# ============================================================================

@app.route('/predict-bulk', methods=['POST'])
def api_predict_bulk():
    """Recibe un archivo CSV de entrada, realiza predicciones por lotes y almacena los resultados."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se cargó ningún archivo.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío.'}), 400
        
    model_name = request.form.get('model_name', 'xgboost')
    threshold = float(request.form.get('threshold', 14.0))
    
    try:
        # Guardar archivo temporal
        file_id = str(uuid.uuid4())
        filepath = UPLOAD_DIR / f"{file_id}_input.csv"
        file.save(filepath)
        
        # Leer archivo
        df_input = pd.read_csv(filepath, sep=';')
        
        # Validar columnas mínimas requeridas
        req_cols = ['diagnosticos_primarios', 'diagnosticos_secundarios', 'procedimientos', 'es_urgencia']
        missing_cols = [c for c in req_cols if c not in df_input.columns]
        if missing_cols:
            # Reintentar leyendo con separador coma en vez de punto y coma
            df_input = pd.read_csv(filepath, sep=',')
            missing_cols = [c for c in req_cols if c not in df_input.columns]
            if missing_cols:
                return jsonify({
                    'error': f"El archivo subido no contiene la estructura requerida del entrenamiento. Columnas faltantes: {missing_cols}"
                }), 400
                
        # Asegurar columna case_id
        if 'case_id' not in df_input.columns:
            df_input['case_id'] = [f"B-{i+1:04d}" for i in range(len(df_input))]
            
        predictions_output = []
        df_results = df_input.copy()
        pred_los_list = []
        charlson_list = []
        risk_list = []
        
        # Preprocesar e inferir fila por fila para garantizar robustez
        for idx, row in df_input.iterrows():
            case_id = str(row['case_id'])
            es_urg = int(row['es_urgencia'])
            fecha_ing = row.get('fecha_ingreso', None)
            
            # Limpiar listas
            diag_prim = str(row['diagnosticos_primarios']).strip().upper() if pd.notna(row['diagnosticos_primarios']) else ""
            
            diag_sec_raw = str(row['diagnosticos_secundarios']).strip() if pd.notna(row['diagnosticos_secundarios']) else ""
            diag_sec = [d.strip().upper() for d in diag_sec_raw.split(',') if d.strip()] if diag_sec_raw else []
            
            proc_raw = str(row['procedimientos']).strip() if pd.notna(row['procedimientos']) else ""
            procs = [p.strip().upper() for p in proc_raw.split(',') if p.strip()] if proc_raw else []
            
            # 1. Calcular Charlson
            todos_diags = []
            if diag_prim:
                todos_diags.append(diag_prim)
            todos_diags.extend(diag_sec)
            c_index = preprocessing_helper.calcular_charlson(todos_diags)
            charlson_list.append(c_index)
            
            # 2. Inferencia en dos etapas
            pred_days = 0.0
            segment = "urgente" if es_urg == 1 else "programado"
            
            clf_bundle = models[f"clf_{segment}"]
            clf_features = clf_bundle["features"]
            
            # Vector ML base para Etapa 1
            df_vec = preprocessing_helper.construir_vector_paciente(
                diag_primario=diag_prim,
                diags_secundarios=diag_sec,
                procedimientos=procs,
                es_urgencia=es_urg,
                fecha_ingreso=fecha_ing,
                columnas_modelo=clf_features
            )
            
            # Asegurar tipo de datos int para booleanos
            bool_cols = df_vec.select_dtypes(include="bool").columns
            if len(bool_cols) > 0:
                df_vec[bool_cols] = df_vec[bool_cols].astype(int)
                
            # Clasificador Etapa 1
            clf_model = clf_bundle["model"]
            prob_plos_14 = float(clf_model.predict_proba(df_vec)[:, 1][0])
            
            if model_name == 'ridge':
                reg_lr_bundle = models[f"reg_lr_{segment}"]
                df_vec_lr = preprocessing_helper.construir_vector_paciente_lr(
                    diag_primario=diag_prim,
                    diags_secundarios=diag_sec,
                    procedimientos=procs,
                    es_urgencia=es_urg,
                    fecha_ingreso=fecha_ing,
                    columnas_modelo=reg_lr_bundle["features"],
                    es_urgente_modelo=(es_urg == 1)
                )
                active_model = reg_lr_bundle["model"]
                if active_model is not None:
                    pred_days = float(active_model.predict(df_vec_lr)[0])
            elif model_name == 'random_forest':
                reg_rf_bundle = models[f"reg_rf_{segment}"]
                df_vec_reg_rf = df_vec.copy()
                df_vec_reg_rf["prob_los_14"] = prob_plos_14
                df_vec_reg_rf = df_vec_reg_rf[reg_rf_bundle["features"]]
                active_model = reg_rf_bundle["model"]
                if active_model is not None:
                    pred_days = float(active_model.predict(df_vec_reg_rf)[0])
            else: # xgboost
                reg_xgb_bundle = models[f"reg_xgb_{segment}"]
                df_vec_reg_xgb = df_vec.copy()
                df_vec_reg_xgb["prob_los_14"] = prob_plos_14
                df_vec_reg_xgb = df_vec_reg_xgb[reg_xgb_bundle["features"]]
                active_model = reg_xgb_bundle["model"]
                if active_model is not None:
                    pred_days = float(active_model.predict(df_vec_reg_xgb)[0])
                    
            pred_days = max(0.0, float(pred_days))
            pred_los_list.append(pred_days)
            
            # 3. Risk level
            risk = 'low'
            if pred_days >= threshold:
                risk = 'high'
            elif pred_days >= 6.0:
                risk = 'medium'
            risk_list.append(risk)
            
            # Para visualización resumida en la UI (límite de 50 registros en respuesta JSON)
            if len(predictions_output) < 50:
                predictions_output.append({
                    'case_id': case_id,
                    'predicted_los': pred_days,
                    'risk_level': risk,
                    'diagnostico_primario': diag_prim if diag_prim else "Sin especificar",
                    'charlson_index': c_index
                })
                
        # Guardar las predicciones en el dataframe de salida
        df_results['los_predicho_dias'] = pred_los_list
        df_results['charlson_calculado'] = charlson_list
        df_results['nivel_riesgo'] = risk_list
        
        # Exportar CSV de salida
        out_filepath = UPLOAD_DIR / f"{file_id}_output.csv"
        df_results.to_csv(out_filepath, sep=';', index=False)
        
        # Guardar en memoria
        batch_store[file_id] = {
            'input_path': str(filepath),
            'output_path': str(out_filepath),
            'filename': file.filename
        }
        
        return jsonify({
            'batch_id': file_id,
            'predictions': predictions_output,
            'total_rows': len(df_input)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error procesando archivo CSV: {str(e)}"}), 500

@app.route('/export-bulk/<batch_id>', methods=['GET'])
def export_bulk(batch_id):
    """Permite al usuario descargar el archivo CSV de salida con predicciones anexadas."""
    if batch_id not in batch_store:
        return "ID de lote no encontrado.", 404
        
    data = batch_store[batch_id]
    out_path = data['output_path']
    orig_filename = data['filename']
    
    # Agregar prefijo predicho al nombre original
    base_name = orig_filename.rsplit('.', 1)[0]
    out_filename = f"{base_name}_predicciones_stay_intel.csv"
    
    return send_file(out_path, as_attachment=True, download_name=out_filename)

@app.route('/download-template', methods=['GET'])
def download_template():
    """Genera y descarga una plantilla CSV con la estructura correcta del dataset maestro."""
    template_data = {
        'case_id': ['14035188', '14085514', '14102910'],
        'fecha_ingreso': ['2018-01-11', '2018-02-07', '2018-03-12'],
        'es_urgencia': [1, 0, 1],
        'diagnosticos_primarios': ['J984', 'S72302E', 'I219'],
        'diagnosticos_secundarios': ['I119,E669,Z6830', 'V299XXD', 'E785,I10'],
        'procedimientos': ['0BB64ZZ', '0QP934Z', '0210093,02500ZZ']
    }
    df_template = pd.DataFrame(template_data)
    
    template_path = UPLOAD_DIR / "plantilla_pacientes.csv"
    df_template.to_csv(template_path, sep=';', index=False)
    
    return send_file(template_path, as_attachment=True, download_name="plantilla_pacientes_estancia.csv")

if __name__ == '__main__':
    # Ejecutar en puerto local estándar 5000 para desarrollo
    app.run(host='0.0.0.0', port=5000, debug=True)
