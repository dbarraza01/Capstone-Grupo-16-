"""
tuning_random_forest.py
=======================
Fase 2.5 — Búsqueda de Hiperparámetros Óptimos para Random Forest

Utiliza RandomizedSearchCV con validación cruzada 5-Fold para encontrar
la combinación de hiperparámetros que minimiza el MAE (Mean Absolute Error).

Metodología basada en:
  - Stone K, et al. "A systematic review of the prediction of hospital length
    of stay." BMC Medical Informatics and Decision Making, 2022;22:55.
  - Oshiro TM, et al. "How Many Trees in a Random Forest?" MLDM 2012.

Este script NO entrena un modelo final. Solo busca y exporta los mejores
hiperparámetros como archivo JSON para uso posterior en la Fase 3.

Ejecuta la búsqueda sobre los 3 escenarios:
  - Escenario A: Dataset v2 original (baseline)
  - Escenario B: Dataset v2 + Charlson Index
  - Escenario C: Base + Procedimientos + Elixhauser (sin ICD dispersos)
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import uniform, randint

# ============================================================================
# Configuración
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[3]  # RF/ -> modelos/ -> ml/ -> raiz/
RANDOM_STATE = 42
N_FOLDS = 5          # Número de folds para cross-validation
N_ITER = 50          # Número de combinaciones aleatorias a probar
N_JOBS = -1          # Usar todos los cores disponibles

# Datasets a evaluar
DATASETS = {
    "escenario_A": BASE_DIR / "ml" / "feature_engineering" / "processed_v2" / "model_data_ml_v2.csv",
    "escenario_B": BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv",
    "escenario_C": BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_C_elixhauser.csv",
}

# Directorio de salida (mismo directorio que este script)
OUT_DIR = Path(__file__).resolve().parent
TARGET_COL = "los_dias"
ID_COL = "case_id"

# ============================================================================
# Espacio de búsqueda de hiperparámetros
# ============================================================================
# Rangos justificados por literatura y mejores prácticas para Random Forest
# en datos médicos tabulares.
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 1000),       # Árboles: explorar rango amplio
    "max_depth": [None, 5, 10, 15, 20, 30],  # None = sin límite (default RF)
    "min_samples_split": randint(2, 20),      # Mínimo muestras para dividir
    "min_samples_leaf": randint(1, 15),       # Mínimo muestras por hoja
    "max_features": ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],  # Features por split
    "bootstrap": [True],                       # Siempre True para RF estándar
    "max_samples": uniform(0.5, 0.5),         # Submuestreo: 0.5-1.0
}

# ============================================================================
# Funciones
# ============================================================================
def load_and_prepare(path):
    """Carga un dataset y separa features de target."""
    df = pd.read_csv(path, sep=';')
    
    # Eliminar columna de ID (no es feature)
    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]
    
    # Convertir booleanos a int si los hay
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    return X, y


def run_tuning(X, y, scenario_name):
    """Ejecuta RandomizedSearchCV para un escenario dado."""
    print(f"\n{'='*60}")
    print(f"  TUNING Random Forest — {scenario_name}")
    print(f"  Features: {X.shape[1]} | Muestras: {X.shape[0]}")
    print(f"  Combinaciones a probar: {N_ITER} | Folds: {N_FOLDS}")
    print(f"{'='*60}")
    
    # Modelo base
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    
    # Validación cruzada
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring='neg_mean_absolute_error',  # Minimizar MAE
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=True,
    )
    
    start_time = time.time()
    search.fit(X, y)
    elapsed = time.time() - start_time
    
    # Resultados
    best_params = search.best_params_
    best_mae = -search.best_score_  # Negativo porque sklearn minimiza
    
    print(f"\n  ✅ Búsqueda completada en {elapsed/60:.1f} minutos")
    print(f"  Mejor MAE (CV promedio): {best_mae:.4f} días")
    print(f"  Mejores hiperparámetros:")
    for param, value in sorted(best_params.items()):
        print(f"    {param}: {value}")
    
    # Extraer top 5 combinaciones
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    top5 = results_df[['params', 'mean_test_score', 'std_test_score',
                        'mean_train_score']].head(5)
    
    return {
        "best_params": best_params,
        "best_mae_cv": round(best_mae, 4),
        "elapsed_minutes": round(elapsed / 60, 2),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "n_iter": N_ITER,
        "n_folds": N_FOLDS,
        "top5_results": [
            {
                "rank": i + 1,
                "mae": round(-row['mean_test_score'], 4),
                "std": round(row['std_test_score'], 4),
                "train_mae": round(-row['mean_train_score'], 4),
                "params": row['params']
            }
            for i, (_, row) in enumerate(top5.iterrows())
        ]
    }


def main():
    print("="*60)
    print("  Random Forest Hyperparameter Tuning — Fase 2.5")
    print("  RandomizedSearchCV + 5-Fold Cross-Validation")
    print("="*60)
    
    all_results = {}
    
    for scenario_name, data_path in DATASETS.items():
        if not data_path.exists():
            print(f"\n⚠️  {scenario_name}: archivo no encontrado, saltando...")
            continue
            
        X, y = load_and_prepare(data_path)
        result = run_tuning(X, y, scenario_name)
        all_results[scenario_name] = result
    
    # ========================================================================
    # Exportar resultados como JSON
    # ========================================================================
    
    # Convertir tipos numpy a tipos nativos de Python para JSON
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    all_results = convert_types(all_results)
    
    out_path = OUT_DIR / "mejores_hiperparametros_random_forest.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  RESULTADOS EXPORTADOS")
    print(f"  Archivo: {out_path.name}")
    print(f"{'='*60}")
    
    # Resumen comparativo
    print("\n  RESUMEN COMPARATIVO:")
    print(f"  {'Escenario':<15} {'MAE (CV)':<12} {'Features':<10} {'Tiempo'}")
    print(f"  {'-'*50}")
    for name, res in all_results.items():
        print(f"  {name:<15} {res['best_mae_cv']:<12} {res['n_features']:<10} {res['elapsed_minutes']} min")


if __name__ == "__main__":
    main()
