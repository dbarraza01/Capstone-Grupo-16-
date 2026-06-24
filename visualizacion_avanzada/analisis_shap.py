import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Configurar rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Rutas de datos y modelos en ml_operacional_entrega3
ML_DIR = PROJECT_ROOT / "ml_operacional_entrega3"
MODELS_DIR = ML_DIR / "modelos_guardados"
DATA_SPLITS_DIR = ML_DIR / "data_splits"
OUTPUT_DIR = PROJECT_ROOT / "visualizacion_avanzada"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "los_dias"
ID_COL = "case_id"

def load_data_and_predict(segment):
    """Carga los modelos reales de dos etapas y calcula los datos con probabilidad."""
    # 1. Cargar splits de holdout reales
    holdout_df = pd.read_csv(DATA_SPLITS_DIR / f"datos_holdout_{segment}.csv")
    
    # 2. Cargar clasificadores y regresores reales de la entrega
    clf_bundle = joblib.load(MODELS_DIR / f"clf_xgb_{segment}.joblib")
    reg_bundle = joblib.load(MODELS_DIR / f"reg_xgb_{segment}.joblib")
    
    clf_model = clf_bundle["model"]
    reg_model = reg_bundle["model"]
    
    clf_features = clf_bundle["features"]
    reg_features = reg_bundle["features"]
    
    # 3. Preparar X para el clasificador
    X_clf = holdout_df[clf_features].copy()
    # Asegurar que los booleanos sean enteros
    bool_cols = X_clf.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X_clf[bool_cols] = X_clf[bool_cols].astype(int)
        
    # 4. Obtener probabilidades estimadas por el clasificador oficial
    prob = clf_model.predict_proba(X_clf)[:, 1]
    
    # 5. Preparar X para el regresor (añadiendo la probabilidad real de la Etapa 1)
    X_reg = X_clf.copy()
    X_reg["prob_los_14"] = prob
    X_reg = X_reg[reg_features]
    
    return X_reg, reg_model

def main():
    print("======================================================================")
    print("Iniciando Cálculo y Generación de Gráficos SHAP (Modelos Reales)")
    print("======================================================================")
    
    for segment in ["urgente", "programado"]:
        print(f"\nProcesando segmento: {segment.upper()}...")
        X, reg_trans = load_data_and_predict(segment)
        
        # Como reg_trans es un TransformedTargetRegressor, el modelo XGBoost real
        # se encuentra en el atributo regressor_
        xgb_reg = reg_trans.regressor_
        
        # Inicializar el explicador SHAP con el modelo XGBoost consolidado
        print("  [INFO] Inicializando TreeExplainer de SHAP...")
        explainer = shap.TreeExplainer(xgb_reg)
        
        # Calcular los valores SHAP sobre un subconjunto de holdout representativo (150 casos)
        # para agilizar el cómputo y mejorar la visualización del summary plot
        sample_size = min(150, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        print(f"  [INFO] Calculando SHAP values para {sample_size} pacientes...")
        shap_values = explainer(X_sample)
        
        # 1. Summary Plot (Beeswarm)
        print("  [INFO] Generando SHAP Summary Plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"Impacto SHAP Global (XGBoost Regressor - {segment.capitalize()})", fontsize=14, pad=15)
        summary_path = OUTPUT_DIR / f"shap_summary_{segment}.png"
        plt.savefig(summary_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  [OK] Guardado: {summary_path.name}")
        
        # 2. Waterfall Plot (Caso Individual de Alto Riesgo)
        # Buscar un caso interesante que tenga alta probabilidad de larga estancia (prob_los_14 alto)
        high_risk_idx = X_sample["prob_los_14"].idxmax()
        sample_idx = X_sample.index.get_loc(high_risk_idx)
        
        print(f"  [INFO] Generando SHAP Waterfall Plot para el paciente ID {high_risk_idx} (alto riesgo)...")
        plt.figure(figsize=(10, 6))
        # En las versiones más nuevas de SHAP, el waterfall espera un elemento de Explanation
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        plt.title(f"Explicabilidad Individual (Paciente Alto Riesgo - {segment.capitalize()})", fontsize=14, pad=15)
        waterfall_path = OUTPUT_DIR / f"shap_waterfall_{segment}.png"
        plt.savefig(waterfall_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  [OK] Guardado: {waterfall_path.name}")
        
        # 3. Dependence Plot (prob_los_14 vs SHAP)
        print("  [INFO] Generando SHAP Dependence Plot para prob_los_14...")
        plt.figure(figsize=(8, 5))
        # Buscamos la posición de la columna prob_los_14
        shap.plots.scatter(shap_values[:, "prob_los_14"], show=False)
        plt.title(f"Dependencia SHAP: prob_los_14 vs. Impacto ({segment.capitalize()})", fontsize=12, pad=15)
        dependence_path = OUTPUT_DIR / f"shap_dependence_{segment}.png"
        plt.savefig(dependence_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  [OK] Guardado: {dependence_path.name}")
        
    print("\n[ÉXITO] Todos los gráficos SHAP han sido generados y guardados correctamente en:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    main()
