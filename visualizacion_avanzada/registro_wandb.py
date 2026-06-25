import os
import sys
import joblib
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score, precision_score

# Configurar rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Rutas de datos y modelos en ml_operacional_entrega3
ML_DIR = PROJECT_ROOT / "ml_operacional_entrega3"
MODELS_DIR = ML_DIR / "modelos_guardados"
DATA_SPLITS_DIR = ML_DIR / "data_splits"

TARGET_COL = "los_dias"
ID_COL = "case_id"

def evaluar_modelos_reales(run):
    """Carga los modelos consolidados de la entrega y los evalúa en holdout, logueando a W&B."""
    print("\n--- Evaluando Modelos Reales Consolidados y Reportando a W&B ---")
    
    global_rows = []
    
    for segment in ["urgente", "programado"]:
        print(f"Evaluando segmento: {segment.upper()}...")
        # 1. Cargar splits de holdout reales
        holdout_df = pd.read_csv(DATA_SPLITS_DIR / f"datos_holdout_{segment}.csv")
        
        # Cargar bundles reales
        clf_bundle = joblib.load(MODELS_DIR / f"clf_xgb_{segment}.joblib")
        reg_bundle = joblib.load(MODELS_DIR / f"reg_xgb_{segment}.joblib")
        
        clf = clf_bundle["model"]
        reg = reg_bundle["model"]
        clf_features = clf_bundle["features"]
        reg_features = reg_bundle["features"]
        
        # Preparar features
        X = holdout_df[clf_features].copy()
        bool_cols = X.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)
            
        y = holdout_df[TARGET_COL].copy()
        
        # Inferencia Etapa 1 (Clasificador)
        y_probas = clf.predict_proba(X)[:, 1]
        y_preds_clf = (y_probas >= 0.50).astype(int)
        y_true_clf = (y >= 14).astype(int)
        
        # Inferencia Etapa 2 (XGBoost Regressor)
        X_reg = X.copy()
        X_reg["prob_los_14"] = y_probas
        y_preds_reg = np.clip(reg.predict(X_reg[reg_features]), 0, None)
        
        # Inferencia Random Forest Regressor
        reg_rf_bundle = joblib.load(MODELS_DIR / f"reg_rf_{segment}.joblib")
        reg_rf = reg_rf_bundle["model"]
        reg_rf_features = reg_rf_bundle["features"]
        y_preds_rf = np.clip(reg_rf.predict(X_reg[reg_rf_features]), 0, None)
        
        # Inferencia Regresion Lineal Base
        reg_lr_bundle = joblib.load(MODELS_DIR / f"reg_lr_{segment}.joblib")
        reg_lr = reg_lr_bundle["model"]
        reg_lr_features = reg_lr_bundle["features"]
        
        df_lr = holdout_df.copy()
        if len(bool_cols) > 0:
            df_lr[bool_cols] = df_lr[bool_cols].astype(int)
        df_lr["int_charlson_diag"] = df_lr["charlson_index"] * df_lr["n_diag_total"]
        df_lr["int_proc_diag"] = df_lr["n_procedimientos"] * df_lr["n_diag_total"]
        df_lr["int_charlson_proc"] = df_lr["charlson_index"] * df_lr["n_procedimientos"]
        X_lr = df_lr[reg_lr_features]
        y_preds_lr = np.clip(reg_lr.predict(X_lr), 0, None)
        
        # Métricas de Regresión XGBoost
        mae = mean_absolute_error(y, y_preds_reg)
        rmse = np.sqrt(mean_squared_error(y, y_preds_reg))
        medae = np.median(np.abs(y_preds_reg - y))
        
        # Métricas de Regresión RF
        mae_rf = mean_absolute_error(y, y_preds_rf)
        rmse_rf = np.sqrt(mean_squared_error(y, y_preds_rf))
        
        # Metricas de Regresion Lineal Base
        mae_lr = mean_absolute_error(y, y_preds_lr)
        rmse_lr = np.sqrt(mean_squared_error(y, y_preds_lr))
        
        # Métricas del Clasificador de Riesgo (Etapa 1)
        precision = precision_score(y_true_clf, y_preds_clf, zero_division=0)
        recall = recall_score(y_true_clf, y_preds_clf, zero_division=0)
        f1 = f1_score(y_true_clf, y_preds_clf, zero_division=0)
        
        # 1. Crear tabla comparativa de regresores (MAE y RMSE)
        tabla_regresores = wandb.Table(columns=["Modelo", "MAE (Días)", "RMSE (Días)"])
        tabla_regresores.add_data("XGBoost (Ganador)", mae, rmse)
        tabla_regresores.add_data("Random Forest", mae_rf, rmse_rf)
        tabla_regresores.add_data("Regresión Lineal Base", mae_lr, rmse_lr)
        
        # 2. Crear tabla de desempeño del clasificador de riesgo (Etapa 1)
        tabla_clasificador = wandb.Table(columns=["Métrica", "Valor"])
        tabla_clasificador.add_data("Sensibilidad (Recall)", recall)
        tabla_clasificador.add_data("Precisión de Alertas", precision)
        tabla_clasificador.add_data("F1-Score", f1)
        
        # Loguear las tablas comparativas interactivas
        wandb.log({
            f"{segment}/comparativa_regresores": tabla_regresores,
            f"{segment}/metricas_clasificador": tabla_clasificador
        })
        
        # Registrar métricas individuales en el resumen (summary) del run 
        # para evitar generar gráficos vacíos de un solo punto en la sección de "Charts"
        run.summary.update({
            f"{segment}_xgb_holdout_mae": mae,
            f"{segment}_xgb_holdout_rmse": rmse,
            f"{segment}_xgb_holdout_medae": medae,
            f"{segment}_rf_holdout_mae": mae_rf,
            f"{segment}_rf_holdout_rmse": rmse_rf,
            f"{segment}_lr_holdout_mae": mae_lr,
            f"{segment}_lr_holdout_rmse": rmse_lr,
            f"{segment}_classifier_precision": precision,
            f"{segment}_classifier_recall": recall,
            f"{segment}_classifier_f1": f1,
        })
        
        # Loguear gráficos de clasificación scikit-learn provistos por W&B
        # 1. Matriz de confusión del Clasificador
        wandb.log({
            f"{segment}/matriz_confusion": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_clf,
                preds=y_preds_clf,
                class_names=["LOS < 14", "LOS >= 14"]
            )
        })
        
        # 2. Curva ROC
        # Para graficar ROC necesitamos las probabilidades en formato bidimensional [prob_neg, prob_pos]
        y_probas_2d = np.vstack([1 - y_probas, y_probas]).T
        wandb.log({
            f"{segment}/curva_roc": wandb.plot.roc_curve(
                y_true=y_true_clf,
                y_probas=y_probas_2d,
                labels=["LOS < 14", "LOS >= 14"]
            )
        })
        
        print(f"  [OK] Métricas e Histogramas de {segment} registrados en W&B.")

def entrenar_modelo_con_registro_loop(run):
    """
    Ejemplo interactivo que entrena un modelo de regresion lineal por tamanos crecientes de muestra
    y muestra cómo registrar métricas PASO A PASO en el bucle de entrenamiento,
    tal como indica el gráfico del usuario.
    """
    print("\n--- Demostración de Registro de Métricas en Bucle (Regresion Lineal) ---")
    
    # 1. Cargar datos
    train_df = pd.read_csv(DATA_SPLITS_DIR / "datos_train_urgente.csv")
    holdout_df = pd.read_csv(DATA_SPLITS_DIR / "datos_holdout_urgente.csv")
    
    # Tomar un subset de columnas básicas
    cols = ["charlson_index", "n_diag_total", "n_procedimientos"]
    X_train = train_df[cols].fillna(0)
    y_train = train_df[TARGET_COL]
    
    X_val = holdout_df[cols].fillna(0)
    y_val = holdout_df[TARGET_COL]
    
    # Registrar hiperparámetros iniciales
    wandb.config.update({
        "algorithm": "Linear Regression Demo",
        "features": cols,
        "max_iterations": 20,
        "regularization": "none"
    })
    
    # Simular iteraciones de entrenamiento usando fracciones crecientes del train.
    for epoch in range(1, 21):
        train_fraction = epoch / 20
        sample_size = max(20, int(len(X_train) * train_fraction))
        sample_size = min(sample_size, len(X_train))
        X_step = X_train.sample(n=sample_size, random_state=42 + epoch)
        y_step = y_train.loc[X_step.index]
        
        # Entrenar regresor
        model = TransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log1p,
            inverse_func=np.expm1
        )
        model.fit(X_step, y_step)
        
        # Calcular pérdidas
        pred_train = model.predict(X_step)
        pred_val = model.predict(X_val)
        
        train_mae = mean_absolute_error(y_step, pred_train)
        val_mae = mean_absolute_error(y_val, pred_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_step, pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        
        # CÓDIGO EXACTO DEL BUCLE DE ENTRENAMIENTO (Solicitado por el usuario)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_rmse,
            "val_loss": val_rmse,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_fraction": train_fraction
        })
        
        print(f"  Época {epoch:02d}/20 | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
        
    print("  [OK] Entrenamiento e historial iterativo registrado con éxito.")

def main():
    print("======================================================================")
    print("Iniciando Registro Operacional en Weights & Biases (W&B)")
    print("======================================================================")
    
    # Inicializar el experimento en WandB
    # El comando pedirá al usuario loguearse si es la primera vez que lo corre
    run = wandb.init(
        project="Stay_Intelligence_Capstone",
        name="evaluacion_modelos_entrega3",
        notes="Evaluación de modelos XGBoost en Holdout y demostración de log en bucle",
        tags=["holdout_evaluation", "xgboost", "linear_regression"]
    )
    
    try:
        # 1. Evaluar modelos reales de la entrega
        evaluar_modelos_reales(run)
        
        # 2. Correr demostración del loop de entrenamiento solicitado
        entrenar_modelo_con_registro_loop(run)
        
    except Exception as e:
        print(f"\n[ERROR] Ocurrió una excepción durante el registro: {e}")
    finally:
        # Finalizar el run de wandb para subir todos los logs
        wandb.finish()
        
    print("\n[ÉXITO] Registro de métricas y gráficos finalizado. Accede a tu dashboard en wandb.ai")

if __name__ == "__main__":
    main()
