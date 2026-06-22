import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv"

# Cargar dataset
df = pd.read_csv(DATA_PATH, sep=';')

# Rellenar nulos si existen
df['charlson_index'] = df['charlson_index'].fillna(0)
df['n_diag_total'] = df['n_diag_total'].fillna(0)
df['n_procedimientos'] = df['n_procedimientos'].fillna(0)
df['los_dias'] = df['los_dias'].fillna(0)

# Crear target log1p
df['log_los'] = np.log1p(df['los_dias'])

# Separar por urgencia
df_urg = df[df['es_urgencia'] == 1].copy()
df_no_urg = df[df['es_urgencia'] == 0].copy()

# Definición de fórmulas para evaluar
# Modelo 1: Aditivo (sin interacciones)
formula_add = "log_los ~ charlson_index + n_diag_total + n_procedimientos"
# Modelo 2: Con interacciones
formula_int = "log_los ~ charlson_index * n_diag_total + n_procedimientos * n_diag_total + charlson_index * n_procedimientos"

def analizar_grupo(df_sub, nombre_grupo):
    print(f"\n=======================================================")
    print(f"ANÁLISIS ESTADÍSTICO PARA GRUPO: {nombre_grupo}")
    print(f"=======================================================")
    
    # Ajustar modelos
    model_add = ols(formula_add, data=df_sub).fit()
    model_int = ols(formula_int, data=df_sub).fit()
    
    # 1. Comparar AIC y BIC
    print(f"Modelo Aditivo:  AIC = {model_add.aic:.2f}, BIC = {model_add.bic:.2f}, R2_adj = {model_add.rsquared_adj:.4f}")
    print(f"Modelo Interac:  AIC = {model_int.aic:.2f}, BIC = {model_int.bic:.2f}, R2_adj = {model_int.rsquared_adj:.4f}")
    
    # 2. ANOVA (F-Test) para comparar modelos anidados
    anova_results = sm.stats.anova_lm(model_add, model_int)
    print("\nTabla ANOVA para modelos anidados:")
    print(anova_results.to_string())
    
    # 3. Resumen del modelo con interacciones (especialmente p-values de los coeficientes)
    print("\nCoeficientes y p-values del modelo con interacciones:")
    params = model_int.params
    pvalues = model_int.pvalues
    tvalues = model_int.tvalues
    
    summary_df = pd.DataFrame({
        'Coeficiente': params,
        't-stat': tvalues,
        'p-value': pvalues
    })
    print(summary_df.to_string())

analizar_grupo(df_urg, "URGENCIAS")
analizar_grupo(df_no_urg, "NO URGENCIAS")
analizar_grupo(df, "GLOBAL (PARA COMPARAR)")
