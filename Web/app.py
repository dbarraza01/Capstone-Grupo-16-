import os
import sys
import uuid
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path

# Configurar rutas y directorios
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Importar helper de preprocesamiento
sys.path.append(str(WEB_DIR))
import preprocessing_helper

app = Flask(__name__)

# Almacenamiento en memoria para resultados de lotes (bulk predictions)
batch_store = {}

# Carga de Modelos y Columnas en memoria al iniciar el servidor
xgb_model = None
rf_model = None
lr_urg_model = None
lr_nurg_model = None

xgb_cols = []
lr_urg_cols = []
lr_nurg_cols = []

def cargar_modelos():
    global xgb_model, rf_model, lr_urg_model, lr_nurg_model
    global xgb_cols, lr_urg_cols, lr_nurg_cols
    
    print("Cargando modelos clínicos en memoria...")
    
    # 1. Rutas de archivos
    xgb_path = BASE_DIR / "ml" / "modelos" / "XGB" / "final" / "xgboost_final.pkl"
    rf_path = BASE_DIR / "ml" / "modelos" / "RF" / "final" / "random_forest_final.pkl"
    lr_urg_path = BASE_DIR / "Modelo_Base_Ultima entrega" / "lr_base_Urgencias.pkl"
    lr_nurg_path = BASE_DIR / "Modelo_Base_Ultima entrega" / "lr_base_No_Urgencias.pkl"
    
    xgb_cols_path = WEB_DIR / "columnas_modelo_final.pkl"
    lr_urg_cols_path = BASE_DIR / "Modelo_Base_Ultima entrega" / "columnas_modelo_lr_Urgencias.pkl"
    lr_nurg_cols_path = BASE_DIR / "Modelo_Base_Ultima entrega" / "columnas_modelo_lr_No_Urgencias.pkl"
    
    try:
        # Carga de XGBoost
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)
            print("  [OK] XGBoost cargado con éxito.")
        else:
            print(f"  [ERROR]: No se encontró el modelo XGBoost en {xgb_path}")
            
        # Carga de Random Forest
        if rf_path.exists():
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)
            print("  [OK] Random Forest cargado con éxito.")
        else:
            print(f"  [ERROR]: No se encontró el modelo Random Forest en {rf_path}")
            
        # Carga de Regresiones Ridge
        if lr_urg_path.exists():
            with open(lr_urg_path, "rb") as f:
                lr_urg_model = pickle.load(f)
            print("  [OK] Regresión Ridge Urgencias cargada.")
        if lr_nurg_path.exists():
            with open(lr_nurg_path, "rb") as f:
                lr_nurg_model = pickle.load(f)
            print("  [OK] Regresión Ridge No Urgencias cargada.")
            
        # Carga de columnas de features
        if xgb_cols_path.exists():
            with open(xgb_cols_path, "rb") as f:
                xgb_cols = pickle.load(f)
        if lr_urg_cols_path.exists():
            with open(lr_urg_cols_path, "rb") as f:
                lr_urg_cols = pickle.load(f)
        if lr_nurg_cols_path.exists():
            with open(lr_nurg_cols_path, "rb") as f:
                lr_nurg_cols = pickle.load(f)
                
        print("Modelos inicializados correctamente en memoria.")
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
    return render_template('analytics.html')

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
    
    try:
        # 1. Construir vectores de entrada
        # Vector para XGBoost y Random Forest
        df_vector_ml = preprocessing_helper.construir_vector_paciente(
            diag_primario=diag_primario,
            diags_secundarios=diags_secundarios,
            procedimientos=procedimientos,
            es_urgencia=es_urgencia,
            fecha_ingreso=fecha_ingreso,
            columnas_modelo=xgb_cols
        )
        
        # Vector para Regresión Ridge correspondiente
        lr_cols = lr_urg_cols if es_urgencia == 1 else lr_nurg_cols
        df_vector_lr = preprocessing_helper.construir_vector_paciente_lr(
            diag_primario=diag_primario,
            diags_secundarios=diags_secundarios,
            procedimientos=procedimientos,
            es_urgencia=es_urgencia,
            fecha_ingreso=fecha_ingreso,
            columnas_modelo=lr_cols,
            es_urgente_modelo=(es_urgencia == 1)
        )
        
        # 2. Ejecutar predicciones (aplicando la exponencial expm1 a la predicción logarítmica)
        pred_xgb = 0.0
        pred_rf = 0.0
        pred_ridge = 0.0
        
        # Inferencia XGBoost
        if xgb_model is not None:
            pred_xgb_log = xgb_model.predict(df_vector_ml)[0]
            pred_xgb = np.expm1(pred_xgb_log)
            pred_xgb = max(0.0, float(pred_xgb))
            
        # Inferencia Random Forest
        if rf_model is not None:
            pred_rf_log = rf_model.predict(df_vector_ml)[0]
            pred_rf = np.expm1(pred_rf_log)
            pred_rf = max(0.0, float(pred_rf))
            
        # Inferencia Ridge (según urgencia)
        lr_active_model = lr_urg_model if es_urgencia == 1 else lr_nurg_model
        if lr_active_model is not None:
            pred_ridge_log = lr_active_model.predict(df_vector_lr)[0]
            pred_ridge = np.expm1(pred_ridge_log)
            pred_ridge = max(0.0, float(pred_ridge))
            
        # Seleccionar la predicción del modelo activo
        selected_pred = pred_xgb
        if model_name == 'random_forest':
            selected_pred = pred_rf
        elif model_name == 'ridge':
            selected_pred = pred_ridge
            
        # 3. Determinar riesgo y labels de negocio
        risk_level = 'low'
        risk_label = f"Riesgo Bajo (Mediana: {selected_pred:.1f} días)"
        if selected_pred >= 14.0:
            risk_level = 'high'
            risk_label = f"Riesgo Elevado / Estancia Prolongada ({selected_pred:.1f} días)"
        elif selected_pred >= 6.0:
            risk_level = 'medium'
            risk_label = f"Riesgo Moderado ({selected_pred:.1f} días)"
            
        # 4. Calcular factores de decisión dinámicos usando coeficientes Ridge
        driving_factors = []
        if lr_active_model is not None:
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
    """Calcula los aportes lineales de cada variable clínica activa a partir de los coeficientes de Ridge."""
    model = lr_urg_model if es_urgente else lr_nurg_model
    cols = lr_urg_cols if es_urgente else lr_nurg_cols
    
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
    threshold = float(request.form.get('threshold', 27.0))
    
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
            
            # 2. Inferencia
            pred_days = 0.0
            if model_name == 'ridge':
                lr_cols = lr_urg_cols if es_urg == 1 else lr_nurg_cols
                df_vec = preprocessing_helper.construir_vector_paciente_lr(
                    diag_primario=diag_prim,
                    diags_secundarios=diag_sec,
                    procedimientos=procs,
                    es_urgencia=es_urg,
                    fecha_ingreso=fecha_ing,
                    columnas_modelo=lr_cols,
                    es_urgente_modelo=(es_urg == 1)
                )
                active_model = lr_urg_model if es_urg == 1 else lr_nurg_model
                if active_model is not None:
                    pred_days = np.expm1(active_model.predict(df_vec)[0])
            else:
                df_vec = preprocessing_helper.construir_vector_paciente(
                    diag_primario=diag_prim,
                    diags_secundarios=diag_sec,
                    procedimientos=procs,
                    es_urgencia=es_urg,
                    fecha_ingreso=fecha_ing,
                    columnas_modelo=xgb_cols
                )
                active_model = xgb_model if model_name == 'xgboost' else rf_model
                if active_model is not None:
                    pred_days = np.expm1(active_model.predict(df_vec)[0])
                    
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
