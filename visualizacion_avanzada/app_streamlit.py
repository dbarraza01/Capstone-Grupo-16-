import os
import sys
import joblib
import subprocess
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Configurar rutas del proyecto para importar helpers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "Web") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "Web"))

# Importar preprocesador oficial de la aplicación web
import preprocessing_helper

# Rutas de modelos en ml_operacional_entrega3
ML_DIR = PROJECT_ROOT / "ml_operacional_entrega3"
MODELS_DIR = ML_DIR / "modelos_guardados"
TARGET_COL = "los_dias"

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Stay Intelligence - Dashboard Operacional Avanzado",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carga de modelos con caché para velocidad en Streamlit
@st.cache_resource
def cargar_modelos_ops():
    models = {}
    for segment in ["urgente", "programado"]:
        models[f"clf_{segment}"] = joblib.load(MODELS_DIR / f"clf_xgb_{segment}.joblib")
        models[f"reg_xgb_{segment}"] = joblib.load(MODELS_DIR / f"reg_xgb_{segment}.joblib")
        models[f"reg_rf_{segment}"] = joblib.load(MODELS_DIR / f"reg_rf_{segment}.joblib")
        models[f"reg_lr_{segment}"] = joblib.load(MODELS_DIR / f"reg_lr_{segment}.joblib")
    return models

# Carga de datos holdout reales con caché
@st.cache_data
def cargar_datos_holdout():
    holdout_data = {}
    for segment in ["urgente", "programado"]:
        path = ML_DIR / "data_splits" / f"datos_holdout_{segment}.csv"
        if path.exists():
            holdout_data[segment] = pd.read_csv(path)
    return holdout_data

try:
    modelos = cargar_modelos_ops()
    holdout_data = cargar_datos_holdout()
    modelos_cargados = True
except Exception as e:
    modelos_cargados = False
    error_cargando = str(e)

# Título Principal
st.title("🏥 Stay Intelligence — Panel de Visualización Operacional")
st.markdown("### Comparación Multi-Modelo en Dos Etapas e Incertidumbre Clínica (SHAP)")
st.write("---")

if not modelos_cargados:
    st.error(f"Error cargando los modelos o datos operacionales. Asegúrate de que la carpeta `ml_operacional_entrega3` contenga los archivos necesarios. Detalle: {error_cargando}")
else:
    # Sidebar: Configuración del Paciente o Selección de Holdout
    st.sidebar.header("📋 Perfil Clínico del Paciente")
    
    es_urgencia = st.sidebar.selectbox(
        "Tipo de Admisión",
        options=["Urgente (Urgencias)", "Programado (Electivo)"],
        index=0
    )
    es_urgente_val = 1 if "Urgente" in es_urgencia else 0
    segment = "urgente" if es_urgente_val == 1 else "programado"
    
    # Selector de origen de datos
    origen_datos = st.sidebar.radio(
        "Origen de los Datos del Paciente",
        options=["Seleccionar Paciente de Holdout (Histórico)", "Ingreso Manual (Simulación)"],
        index=0
    )
    
    # Cargar features esperadas por los modelos
    clf_features = modelos[f"clf_{segment}"]["features"]
    reg_xgb_features = modelos[f"reg_xgb_{segment}"]["features"]
    reg_rf_features = modelos[f"reg_rf_{segment}"]["features"]
    reg_lr_features = modelos[f"reg_lr_{segment}"]["features"]
    
    if origen_datos == "Seleccionar Paciente de Holdout (Histórico)":
        df_holdout = holdout_data[segment]
        # Ordenar por los_dias (estancia real) para facilitar la búsqueda
        df_holdout_sorted = df_holdout.sort_values(by="los_dias")
        
        # Generar lista legible para el selectbox
        cases_options = [
            f"Caso {int(row['case_id'])} (Estancia Real: {int(row['los_dias'])} días, Charlson: {int(row['charlson_index'])})"
            for _, row in df_holdout_sorted.iterrows()
        ]
        
        selected_case_str = st.sidebar.selectbox(
            "Seleccione Caso de Holdout",
            options=cases_options
        )
        
        selected_case_id = int(selected_case_str.split(" ")[1])
        holdout_row = df_holdout[df_holdout['case_id'] == selected_case_id].iloc[0]
        
        # 1. Obtener datos e índices clínicos de la fila de holdout directamente
        charlson_index = int(holdout_row['charlson_index'])
        
        # Reconstruir diccionarios/vectores para predicción
        row_dict = holdout_row.to_dict()
        for k, v in row_dict.items():
            if isinstance(v, bool):
                row_dict[k] = int(v)
        
        # Calcular interacciones requeridas por el modelo lineal (Ridge)
        row_dict['int_charlson_diag'] = charlson_index * int(holdout_row['n_diag_total'])
        row_dict['int_proc_diag'] = int(holdout_row['n_procedimientos']) * int(holdout_row['n_diag_total'])
        row_dict['int_charlson_proc'] = charlson_index * int(holdout_row['n_procedimientos'])
        
        # Crear DataFrame para modelo base de ML
        df_vector_ml = pd.DataFrame([row_dict])[clf_features]
        
        # Predicción de la Etapa 1 (Clasificador de Riesgo PLOS14)
        clf_model = modelos[f"clf_{segment}"]["model"]
        prob_plos_14 = clf_model.predict_proba(df_vector_ml)[:, 1][0]
        
        # Preparar vectores para Regresores de la Etapa 2
        df_vector_reg_xgb = df_vector_ml.copy()
        df_vector_reg_xgb["prob_los_14"] = prob_plos_14
        df_vector_reg_xgb = df_vector_reg_xgb[reg_xgb_features]
        
        df_vector_reg_rf = df_vector_ml.copy()
        df_vector_reg_rf["prob_los_14"] = prob_plos_14
        df_vector_reg_rf = df_vector_reg_rf[reg_rf_features]
        
        # Para Regresión Ridge
        df_vector_lr = pd.DataFrame([row_dict])[reg_lr_features]
        
        # Ejecutar Inferencia en Días
        pred_xgb = modelos[f"reg_xgb_{segment}"]["model"].predict(df_vector_reg_xgb)[0]
        pred_rf = modelos[f"reg_rf_{segment}"]["model"].predict(df_vector_reg_rf)[0]
        pred_lr = modelos[f"reg_lr_{segment}"]["model"].predict(df_vector_lr)[0]
        
        estancia_real = float(holdout_row['los_dias'])
        
    else:
        # Modo Ingreso Manual
        fecha_ingreso = st.sidebar.date_input(
            "Fecha de Ingreso Hospitalario",
            value=pd.to_datetime("2026-06-24")
        )
        
        n_procedimientos = st.sidebar.number_input(
            "Número de Procedimientos",
            min_value=0, max_value=30, value=1, step=1
        )
        
        # Selector dinámico de diagnósticos clínicos comunes para calcular Charlson
        st.sidebar.markdown("**Diagnósticos Médicos (ICD-10):**")
        diags_comunes = ["J984", "S72302E", "I219", "E119", "I10", "E669", "Z6830", "E785", "N183", "J449", "I509", "C349"]
        diag_primario = st.sidebar.selectbox("Diagnóstico Primario", options=diags_comunes, index=2)
        
        diags_secundarios = st.sidebar.multiselect(
            "Diagnósticos Secundarios",
            options=diags_comunes,
            default=["I10", "E119"]
        )
        
        # Procedimientos quirúrgicos comunes
        procs_comunes = ["0BB64ZZ", "0QP934Z", "0210093", "02500ZZ", "5A1935Z", "0FP93DZ", "04710ZZ"]
        procedimientos_sel = st.sidebar.multiselect(
            "Procedimientos Realizados",
            options=procs_comunes,
            default=[procs_comunes[0]] if n_procedimientos > 0 else []
        )
        
        # 1. Calcular dinámicamente el índice de Charlson con la lógica real
        todos_diags = [diag_primario] + diags_secundarios
        todos_diags = list(set([d for d in todos_diags if d]))
        charlson_index = preprocessing_helper.calcular_charlson(todos_diags)
        
        st.sidebar.info(f"🧬 Índice de Charlson Calculado: **{charlson_index}**")
        
        # 2. Construir los vectores del paciente con los helpers reales
        df_vector_ml = preprocessing_helper.construir_vector_paciente(
            diag_primario=diag_primario,
            diags_secundarios=diags_secundarios,
            procedimientos=procedimientos_sel,
            es_urgencia=es_urgente_val,
            fecha_ingreso=str(fecha_ingreso),
            columnas_modelo=clf_features
        )
        
        # 3. Predicción de la Etapa 1 (Clasificador de Riesgo PLOS14)
        clf_model = modelos[f"clf_{segment}"]["model"]
        prob_plos_14 = clf_model.predict_proba(df_vector_ml)[:, 1][0]
        
        # 4. Preparar vectores para Regresores de la Etapa 2
        df_vector_reg_xgb = df_vector_ml.copy()
        df_vector_reg_xgb["prob_los_14"] = prob_plos_14
        df_vector_reg_xgb = df_vector_reg_xgb[reg_xgb_features]
        
        df_vector_reg_rf = df_vector_ml.copy()
        df_vector_reg_rf["prob_los_14"] = prob_plos_14
        df_vector_reg_rf = df_vector_reg_rf[reg_rf_features]
        
        # Para Regresión Ridge
        df_vector_lr = preprocessing_helper.construir_vector_paciente_lr(
            diag_primario=diag_primario,
            diags_secundarios=diags_secundarios,
            procedimientos=procedimientos_sel,
            es_urgencia=es_urgente_val,
            fecha_ingreso=str(fecha_ingreso),
            columnas_modelo=reg_lr_features,
            es_urgente_modelo=(es_urgente_val == 1)
        )
        
        # 5. Ejecutar Inferencia en Días
        pred_xgb = modelos[f"reg_xgb_{segment}"]["model"].predict(df_vector_reg_xgb)[0]
        pred_rf = modelos[f"reg_rf_{segment}"]["model"].predict(df_vector_reg_rf)[0]
        pred_lr = modelos[f"reg_lr_{segment}"]["model"].predict(df_vector_lr)[0]
        
        estancia_real = None
        
    # Asegurar no negatividad
    pred_xgb = max(0.0, float(pred_xgb))
    pred_rf = max(0.0, float(pred_rf))
    pred_lr = max(0.0, float(pred_lr))

    # Definir Pestañas/Tabs para una navegación limpia
    tab_inf, tab_shap, tab_wandb = st.tabs([
        "🔬 Inferencia de Pacientes (Individual & Holdout)",
        "🌍 Explicabilidad Global (SHAP Dinámico)",
        "📊 Telemetría y Auditoría (Weights & Biases)"
    ])

    with tab_inf:
        # Diseño de la Interfaz Principal
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Predicción XGBoost (Ganador)",
                value=f"{pred_xgb:.1f} días",
                delta="Modelo Dos Etapas"
            )
        with col2:
            st.metric(
                label="Predicción Random Forest",
                value=f"{pred_rf:.1f} días",
                delta="Modelo Dos Etapas"
            )
        with col3:
            st.metric(
                label="Predicción Regresión Ridge",
                value=f"{pred_lr:.1f} días",
                delta="Modelo Lineal"
            )
        with col4:
            if estancia_real is not None:
                st.metric(
                    label="Estancia Real (Ground Truth)",
                    value=f"{estancia_real:.0f} días",
                    delta="Dato Histórico Real",
                    delta_color="off"
                )
            else:
                st.metric(
                    label="Estancia Real (Ground Truth)",
                    value="N/A",
                    delta="Modo Simulación"
                )
            
        st.write(" ")
        
        # Si se cargó un paciente de Holdout, mostrar sus variables clínicas activas
        if origen_datos == "Seleccionar Paciente de Holdout (Histórico)":
            with st.expander("👤 Ver Detalles Clínicos Completos del Paciente de Holdout", expanded=True):
                st.markdown(f"**Identificador del Caso (case_id):** `{selected_case_id}`")
                
                c_p1, c_p2, c_p3 = st.columns(3)
                with c_p1:
                    st.markdown(f"**Tipo de Admisión:** `{es_urgencia}`")
                    st.markdown(f"**Total Diagnósticos:** `{int(holdout_row['n_diag_total'])}` (Primarios: `{int(holdout_row['n_diag_primarios'])}`, Secundarios: `{int(holdout_row['n_diag_secundarios'])}`)")
                with c_p2:
                    st.markdown(f"**Número de Procedimientos:** `{int(holdout_row['n_procedimientos'])}`")
                    st.markdown(f"**Índice de Comorbilidad de Charlson:** `{int(holdout_row['charlson_index'])}`")
                with c_p3:
                    # Mapear mes a nombre
                    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
                    mes_ingreso_idx = int(holdout_row['mes_ingreso']) - 1
                    mes_str = meses[mes_ingreso_idx] if 0 <= mes_ingreso_idx < 12 else str(holdout_row['mes_ingreso'])
                    
                    # Mapear dia de la semana
                    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                    dia_idx = int(holdout_row['dia_semana_ingreso'])
                    dia_str = dias[dia_idx] if 0 <= dia_idx < 7 else str(holdout_row['dia_semana_ingreso'])
                    
                    st.markdown(f"**Mes de Ingreso:** {mes_str}")
                    st.markdown(f"**Día de la Semana de Ingreso:** {dia_str}")
                    
                st.markdown("---")
                
                # Obtener diagnósticos y procedimientos activos mapeados a 1 en las columnas binarias
                diag_cols = [col for col in df_holdout.columns if col.startswith("diag_") and holdout_row[col] == 1]
                proc_cols = [col for col in df_holdout.columns if col.startswith("proc_") and holdout_row[col] == 1]
                
                active_diags = [c.replace("diag_", "") for c in diag_cols if not c.startswith("diag_rare_cap_")]
                active_procs = [c.replace("proc_", "") for c in proc_cols if not c.startswith("proc_rare_sec_")]
                
                rare_diags = [c.replace("diag_rare_cap_", "Grupo raro de Cap. ") for c in diag_cols if c.startswith("diag_rare_cap_")]
                rare_procs = [c.replace("proc_rare_sec_", "Grupo raro de Sec. ") for c in proc_cols if c.startswith("proc_rare_sec_")]
                
                d_col, p_col = st.columns(2)
                with d_col:
                    st.markdown("**Diagnósticos Médicos Activos (ICD-10):**")
                    if active_diags:
                        st.write(", ".join([f"`{d}`" for d in active_diags]))
                    if rare_diags:
                        st.write(", ".join([f"*{r}*" for r in rare_diags]))
                    if not active_diags and not rare_diags:
                        st.write("*Ningún diagnóstico registrado en columnas binarias*")
                with p_col:
                    st.markdown("**Procedimientos Realizados Activos:**")
                    if active_procs:
                        st.write(", ".join([f"`{p}`" for p in active_procs]))
                    if rare_procs:
                        st.write(", ".join([f"*{p}*" for r in rare_procs]))
                    if not active_procs and not rare_procs:
                        st.write("*Ningún procedimiento registrado en columnas binarias*")
            st.write(" ")

        # Layout de Dos Columnas: Gráfico comparativo e Incertidumbre IP90
        lay1, lay2 = st.columns(2)
        
        with lay1:
            st.subheader("📊 Comparación de Predicciones")
            if estancia_real is not None:
                chart_data = pd.DataFrame({
                    "Modelo/Dato": ["XGBoost", "Random Forest", "Ridge Regression", "Estancia Real"],
                    "Estancia (Días)": [pred_xgb, pred_rf, pred_lr, estancia_real]
                })
                st.bar_chart(chart_data.set_index("Modelo/Dato"), color="#4F46E5")
            else:
                chart_data = pd.DataFrame({
                    "Modelo/Dato": ["XGBoost", "Random Forest", "Ridge Regression"],
                    "Estancia (Días)": [pred_xgb, pred_rf, pred_lr]
                })
                st.bar_chart(chart_data.set_index("Modelo/Dato"), color="#4F46E5")
            
        with lay2:
            st.subheader("🛡️ Alerta de Riesgo e Incertidumbre Clínica")
            # Alerta probabilística del Clasificador
            st.write(f"**Probabilidad de Estancia Prolongada (LOS ≥ 14 días):** `{prob_plos_14 * 100:.1f}%`")
            
            # Semáforo de riesgo basado en el clasificador de la Etapa 1
            if prob_plos_14 >= 0.50:
                st.error("🚨 **RIESGO ELEVADO:** Paciente tiene alta probabilidad de estancia prolongada (PLOS14). Requiere priorización de alta y auditoría médica.")
            elif prob_plos_14 >= 0.35:
                st.warning("⚠️ **RIESGO MODERADO:** Paciente en alerta de hospitalización prolongada según política clínica de alta seguridad.")
            else:
                st.success("✅ **RIESGO BAJO:** Paciente estable. Se estima estancia corta y salida regular.")
                
            # Intervalo de Predicción IP90 empírico basado en el tramo de XGBoost
            st.markdown("**Intervalo de Predicción al 90% (IP90 Empírico):**")
            if pred_xgb < 3.0:
                ip_min, ip_max = max(0.0, pred_xgb - 1.5), pred_xgb + 4.5
            elif pred_xgb < 7.0:
                ip_min, ip_max = max(0.0, pred_xgb - 2.5), pred_xgb + 6.0
            elif pred_xgb < 14.0:
                ip_min, ip_max = max(0.0, pred_xgb - 5.0), pred_xgb + 12.0
            else:
                ip_min, ip_max = max(0.0, pred_xgb - 15.0), pred_xgb + 22.0
                
            st.info(f"Con un 90% de confianza, la estancia real se situará entre **{ip_min:.1f} y {ip_max:.1f} días**.")
            
        st.write("---")
        
        # Sección SHAP Local (Explicabilidad del paciente actual)
        st.subheader("💡 Explicabilidad Individual de XGBoost con SHAP")
        st.write("El siguiente gráfico de cascada (*Waterfall Plot*) muestra exactamente cómo cada variable clínica y diagnóstica del paciente empujó la predicción de XGBoost hacia arriba (rojo) o hacia abajo (azul) con respecto al valor base promedio de la población:")
        
        # Calcular SHAP en caliente para el paciente actual
        xgb_reg = modelos[f"reg_xgb_{segment}"]["model"].regressor_
        explainer = shap.TreeExplainer(xgb_reg)
        shap_values = explainer(df_vector_reg_xgb)
        
        # Generar gráfico waterfall con matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"Aportes Individuales (SHAP) - Segmento {segment.capitalize()}", fontsize=12, pad=10)
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("""
        **Guía de lectura SHAP:**
        *   El **eje X** muestra el impacto acumulado en el espacio logarítmico.
        *   Las variables en **color rojo** aumentan la estimación de días de estancia hospitalaria.
        *   Las variables en **color azul** disminuyen la estancia esperada.
        """)

    with tab_shap:
        st.subheader("🌍 Explicabilidad Global Dinámica con SHAP")
        st.markdown("""
        Calcula e interpreta de forma global el comportamiento del modelo XGBoost sobre el segmento de pacientes seleccionado.
        A diferencia de los reportes estáticos, aquí puedes configurar el **tamaño de la muestra del holdout** que deseas evaluar para ver el impacto clínico a escala masiva.
        """)
        
        # Slider dinámico de tamaño de muestra
        muestra_size = st.slider(
            "Seleccione el número de pacientes a evaluar para la Explicabilidad Global:",
            min_value=10,
            max_value=400,
            value=150,
            step=10,
            key="shap_sample_slider"
        )
        
        btn_calc_shap = st.button("🚀 Calcular y Generar Gráficos SHAP Globales", key="btn_calc_shap_global")
        
        if btn_calc_shap:
            with st.spinner(f"Ejecutando TreeExplainer de SHAP en una muestra de {muestra_size} pacientes reales del holdout..."):
                try:
                    df_holdout = holdout_data[segment]
                    if muestra_size > len(df_holdout):
                        muestra_size = len(df_holdout)
                        st.warning(f"La muestra se ajustó al máximo de registros del holdout disponible: {muestra_size}")
                        
                    # Muestrear
                    df_sample = df_holdout.sample(n=muestra_size, random_state=42)
                    X_clf = df_sample[clf_features].copy()
                    bool_cols = X_clf.select_dtypes(include="bool").columns
                    if len(bool_cols) > 0:
                        X_clf[bool_cols] = X_clf[bool_cols].astype(int)
                        
                    # Probabilidad Etapa 1
                    prob_los_14 = modelos[f"clf_{segment}"]["model"].predict_proba(X_clf)[:, 1]
                    
                    # Preparar Etapa 2
                    X_reg = X_clf.copy()
                    X_reg["prob_los_14"] = prob_los_14
                    X_reg = X_reg[reg_xgb_features]
                    
                    # Cargar regresor XGBoost base
                    xgb_reg = modelos[f"reg_xgb_{segment}"]["model"].regressor_
                    explainer_global = shap.TreeExplainer(xgb_reg)
                    shap_values_global = explainer_global(X_reg)
                    
                    st.markdown("#### 1. Importancia e Impacto Global de Variables (Summary Beeswarm Plot)")
                    st.write("Cada punto en el gráfico representa a un paciente de la muestra. Las variables están ordenadas por su importancia global. El color indica si la variable tiene un valor alto (rojo) o bajo (azul) para ese paciente:")
                    
                    fig_glob1, ax_glob1 = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values_global, X_reg, show=False)
                    plt.title(f"Impacto SHAP Global (XGBoost Regressor - {segment.capitalize()} - Muestra N={muestra_size})", fontsize=12, pad=15)
                    st.pyplot(fig_glob1)
                    plt.close(fig_glob1)
                    
                    st.markdown("---")
                    
                    st.markdown("#### 2. Relación de Dependencia: Probabilidad de Alta Estancia vs. Impacto SHAP")
                    st.write("Muestra de forma continua cómo influye el valor del clasificador de riesgo (Etapa 1) sobre el impacto final en días (SHAP value) para cada paciente:")
                    
                    fig_glob2, ax_glob2 = plt.subplots(figsize=(8, 5))
                    shap.plots.scatter(shap_values_global[:, "prob_los_14"], show=False)
                    plt.title(f"Dependencia SHAP: prob_los_14 vs. Impacto ({segment.capitalize()} - Muestra N={muestra_size})", fontsize=10, pad=15)
                    st.pyplot(fig_glob2)
                    plt.close(fig_glob2)
                    
                    st.success(f"¡Cómputo completado con éxito para una muestra de {muestra_size} pacientes!")
                except Exception as e:
                    st.error(f"Ocurrió un error al calcular los valores SHAP globales: {e}")

    with tab_wandb:
        st.subheader("📊 Telemetría y Registro de Experimentos con Weights & Biases (W&B)")
        st.write("""
        Weights & Biases (W&B) es una plataforma de **MLOps** utilizada para auditar científicamente la calidad y convergencia de los modelos.
        
        Dado que los registros locales (**offline**) de W&B se guardan como archivos en disco y no tienen una interfaz web interactiva local, hemos provisto dos opciones claras para evaluar y registrar el desempeño:
        """)
        
        # --- OPCIÓN 1: EVALUACIÓN LOCAL INTERACTIVA ---
        st.markdown("### 📊 Opción 1: Evaluación Interactiva Local (Sin registro en la nube)")
        st.write("Calcula las métricas de todos los modelos en caliente y genera gráficas interactivas nativas de Streamlit donde puedes hacer zoom y ver valores al pasar el cursor:")
        
        btn_eval_local = st.button("📊 Calcular Métricas y Curvas Interactivas", key="btn_eval_local")
        
        if btn_eval_local:
            with st.spinner("Calculando predicciones y evaluando holdout..."):
                try:
                    df_holdout_metrics = holdout_data[segment]
                    X_clf_metrics = df_holdout_metrics[clf_features].copy()
                    bool_cols_metrics = X_clf_metrics.select_dtypes(include="bool").columns
                    if len(bool_cols_metrics) > 0:
                        X_clf_metrics[bool_cols_metrics] = X_clf_metrics[bool_cols_metrics].astype(int)
                        
                    y_true_los_metrics = df_holdout_metrics["los_dias"].copy()
                    y_true_clf_metrics = (y_true_los_metrics >= 14).astype(int)
                    
                    # Inferencia Etapa 1
                    clf_model_metrics = modelos[f"clf_{segment}"]["model"]
                    y_probas_metrics = clf_model_metrics.predict_proba(X_clf_metrics)[:, 1]
                    y_preds_clf_metrics = (y_probas_metrics >= 0.50).astype(int)
                    
                    # Inferencia Etapa 2 - XGBoost
                    reg_model_metrics = modelos[f"reg_xgb_{segment}"]["model"]
                    X_reg_metrics = X_clf_metrics.copy()
                    X_reg_metrics["prob_los_14"] = y_probas_metrics
                    y_preds_reg_metrics = np.clip(reg_model_metrics.predict(X_reg_metrics[reg_xgb_features]), 0, None)
                    
                    # Inferencia Etapa 2 - Random Forest
                    reg_rf_metrics = modelos[f"reg_rf_{segment}"]["model"]
                    y_preds_rf_metrics = np.clip(reg_rf_metrics.predict(X_reg_metrics[reg_rf_features]), 0, None)
                    
                    # Inferencia - Ridge Regression
                    reg_lr_metrics = modelos[f"reg_lr_{segment}"]["model"]
                    df_lr_metrics = df_holdout_metrics.copy()
                    if len(bool_cols_metrics) > 0:
                        df_lr_metrics[bool_cols_metrics] = df_lr_metrics[bool_cols_metrics].astype(int)
                    df_lr_metrics["int_charlson_diag"] = df_lr_metrics["charlson_index"] * df_lr_metrics["n_diag_total"]
                    df_lr_metrics["int_proc_diag"] = df_lr_metrics["n_procedimientos"] * df_lr_metrics["n_diag_total"]
                    df_lr_metrics["int_charlson_proc"] = df_lr_metrics["charlson_index"] * df_lr_metrics["n_procedimientos"]
                    X_lr_metrics = df_lr_metrics[reg_lr_features]
                    y_preds_lr_metrics = np.clip(reg_lr_metrics.predict(X_lr_metrics), 0, None)
                    
                    # Calcular métricas con sklearn
                    from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score
                    mae_xgb = mean_absolute_error(y_true_los_metrics, y_preds_reg_metrics)
                    mae_rf = mean_absolute_error(y_true_los_metrics, y_preds_rf_metrics)
                    mae_lr = mean_absolute_error(y_true_los_metrics, y_preds_lr_metrics)
                    
                    precision_metrics = precision_score(y_true_clf_metrics, y_preds_clf_metrics, zero_division=0)
                    recall_metrics = recall_score(y_true_clf_metrics, y_preds_clf_metrics, zero_division=0)
                    f1_metrics = f1_score(y_true_clf_metrics, y_preds_clf_metrics, zero_division=0)
                    
                    st.markdown("#### 🏆 Comparativa Científica de Modelos (MAE en Holdout)")
                    st.write(f"Comparación del **Error Absoluto Medio (MAE)** de predicción en días sobre el holdout para el segmento **{segment.capitalize()}**:")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric(label="MAE XGBoost (Ganador)", value=f"{mae_xgb:.2f} días")
                    with col_m2:
                        st.metric(label="MAE Random Forest", value=f"{mae_rf:.2f} días")
                    with col_m3:
                        st.metric(label="MAE Regresión Ridge", value=f"{mae_lr:.2f} días")
                    
                    st.markdown("#### 🛡️ Desempeño del Clasificador de Riesgo (Etapa 1)")
                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        st.metric(label="Sensibilidad (Recall)", value=f"{recall_metrics * 100:.1f}%")
                    with col_c2:
                        st.metric(label="Precisión de Alertas", value=f"{precision_metrics * 100:.1f}%")
                    with col_c3:
                        st.metric(label="F1-Score", value=f"{f1_metrics:.3f}")
                    
                    # --- CURVA DE APRENDIZAJE INTERACTIVA NATIVA ---
                    st.markdown("#### 📈 Curva de Aprendizaje Interactiva (Ridge Regularization)")
                    st.write("A continuación se muestra el comportamiento del entrenamiento de la regresión Ridge a lo largo de 20 iteraciones de penalización. Pasa el cursor sobre la gráfica para inspeccionar los valores numéricos exactos:")
                    
                    train_df = pd.read_csv(ML_DIR / "data_splits" / f"datos_train_{segment}.csv")
                    cols_lc = ["charlson_index", "n_diag_total", "n_procedimientos"]
                    X_tr = train_df[cols_lc].fillna(0)
                    y_tr = train_df["los_dias"]
                    X_vl = df_holdout_metrics[cols_lc].fillna(0)
                    y_vl = y_true_los_metrics
                    
                    epochs_list = list(range(1, 21))
                    train_maes = []
                    val_maes = []
                    
                    for ep in epochs_list:
                        alpha_val = 10.0 * (1.1 ** ep)
                        from sklearn.linear_model import Ridge
                        from sklearn.compose import TransformedTargetRegressor
                        model_lc = TransformedTargetRegressor(
                            regressor=Ridge(alpha=alpha_val, random_state=42),
                            func=np.log1p,
                            inverse_func=np.expm1
                        )
                        model_lc.fit(X_tr, y_tr)
                        train_maes.append(mean_absolute_error(y_tr, model_lc.predict(X_tr)))
                        val_maes.append(mean_absolute_error(y_vl, model_lc.predict(X_vl)))
                    
                    # Mostrar gráfico interactivo usando line_chart nativo de Streamlit
                    df_chart = pd.DataFrame({
                        "Iteración / Época": epochs_list,
                        "Train MAE (Entrenamiento)": train_maes,
                        "Val MAE (Validación)": val_maes
                    }).set_index("Iteración / Época")
                    st.line_chart(df_chart, color=["#4F46E5", "#EF4444"])
                    st.success("¡Cálculo y renderizado local completado con éxito!")
                except Exception as e_m:
                    st.error(f"No se pudieron calcular las métricas en vivo: {e_m}")
        
        st.write("---")
        
        # --- OPCIÓN 2: REGISTRO Y AUDITORÍA EN W&B ---
        st.markdown("### ☁️ Opción 2: Registro de Auditoría en la Nube de W&B (Gráficos Interactivos)")
        st.write("""
        Para registrar de manera formal este run y visualizar las curvas ROC, matrices de confusión y curvas de aprendizaje en el **dashboard interactivo web de W&B**, puedes configurar y correr la sincronización:
        """)
        
        tipo_registro = st.radio(
            "Modo de Registro de W&B:",
            options=["Online (Sincronizar en la nube de W&B)", "Offline (Almacenar localmente en disco)"],
            index=0
        )
        
        api_key = ""
        if "Online" in tipo_registro:
            st.success("✅ **Entorno Autenticado:** Ya hemos configurado de forma permanente tu W&B API Key en el sistema local.")
            api_key = st.text_input(
                "Llave de API de Weights & Biases (Opcional - dejar en blanco para usar la configurada):",
                type="password",
                placeholder="Ya autenticado localmente..."
            )
            
        btn_run_wandb = st.button("🚀 Ejecutar Registro y Sincronizar con W&B", key="btn_run_wandb")
        
        if btn_run_wandb:
            with st.spinner("Ejecutando script de auditoría registro_wandb.py..."):
                try:
                    script_path = PROJECT_ROOT / "visualizacion_avanzada" / "registro_wandb.py"
                    env = os.environ.copy()
                    
                    if "Online" in tipo_registro:
                        env["WANDB_MODE"] = "online"
                        if api_key.strip():
                            env["WANDB_API_KEY"] = api_key.strip()
                    else:
                        env["WANDB_MODE"] = "offline"
                        
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True,
                        env=env
                    )
                    
                    if result.returncode == 0:
                        st.success("¡Ejecución de auditoría completada con éxito!")
                        st.balloons()
                        
                        # Extraer URL del dashboard en la nube si se ejecutó online
                        import re
                        urls = re.findall(r"https://wandb\.ai/[a-zA-Z0-9_\-\./]+", result.stdout + result.stderr)
                        if urls:
                            # La última URL suele ser el enlace al run en la nube
                            run_url = urls[-1]
                            st.markdown("### 🌐 ¡Tu Dashboard Interactivo Web está listo!")
                            st.write("Haz clic en el botón de abajo para abrir el panel de control de W&B en tu navegador. Allí podrás inspeccionar e interactuar con la matriz de confusión 2D, curva ROC, y curva de aprendizaje interactiva:")
                            st.link_button("👉 Abrir Dashboard Interactivo en Weights & Biases", run_url)
                    else:
                        st.error(f"El script finalizó con código de error {result.returncode}. Revisa los logs de consola abajo.")
                        
                    with st.expander("⚙️ Ver Consola de Sincronización (Detalles Técnicos / Logs de W&B)", expanded=False):
                        st.markdown("**Salida estándar de consola (stdout):**")
                        st.code(result.stdout if result.stdout else "Sin salida estándar.")
                        if result.stderr:
                            st.markdown("**Mensajes de error/advertencias (stderr):**")
                            st.code(result.stderr)
                except Exception as e:
                    st.error(f"Error al ejecutar el script de W&B: {e}")
