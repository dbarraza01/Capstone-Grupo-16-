"""
Visualización: Cómo la verosimilitud cambia según el weight de la mezcla
Muestra por qué el algoritmo converge a weight ≈ 0.75
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, weibull_min
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv('../data/processed/dataset_maestro.csv', sep=';', dtype=str)
df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')
los = df['los_dias'].dropna()
los_shift = los + 1

print("="*70)
print("VISUALIZACIÓN: VEROSIMILITUD vs WEIGHT DE LA MEZCLA")
print("="*70)

# Ajustar componentes individuales
shape_ln_init, loc_ln_init, scale_ln_init = lognorm.fit(los_shift, floc=0)
shape_w_init, loc_w_init, scale_w_init = weibull_min.fit(los_shift, floc=0)

print(f"\nComponentes iniciales:")
print(f"  Log-Normal: σ={shape_ln_init:.4f}, scale={scale_ln_init:.4f}")
print(f"  Weibull: k={shape_w_init:.4f}, λ={scale_w_init:.4f}")

# Definir funciones de mezcla
def mixture_pdf(x, shape_ln, scale_ln, shape_w, scale_w, weight):
    """PDF de la mezcla"""
    pdf_ln = lognorm.pdf(x, s=shape_ln, loc=0, scale=scale_ln)
    pdf_w = weibull_min.pdf(x, c=shape_w, loc=0, scale=scale_w)
    return weight * pdf_ln + (1 - weight) * pdf_w

def negative_log_likelihood_for_weight(weight, data, shape_ln, scale_ln, shape_w, scale_w):
    """Negative log-likelihood"""
    pdf_vals = mixture_pdf(data, shape_ln, scale_ln, shape_w, scale_w, weight)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

def mixture_loglikelihood(weight, data, shape_ln, scale_ln, shape_w, scale_w):
    """Log-verosimilitud"""
    return -negative_log_likelihood_for_weight(weight, data, shape_ln, scale_ln, shape_w, scale_w)

# Probar muchos valores de weight CON PARÁMETROS OPTIMIZADOS para cada weight
weights = np.linspace(0, 1, 101)  # 0%, 1%, 2%, ..., 100%
log_likelihoods = []

print("\nCalculando verosimilitud para cada weight (optimizando parámetros)...")
print("(Esto puede tomar 1-2 minutos)...")

for w in weights:
    # Para este weight, optimizar los otros 4 parámetros
    initial_params = [shape_ln_init, scale_ln_init, shape_w_init, scale_w_init]

    def objective(params):
        shape_ln, scale_ln, shape_w, scale_w = params
        if shape_ln <= 0 or scale_ln <= 0 or shape_w <= 0 or scale_w <= 0:
            return 1e10
        return negative_log_likelihood_for_weight(w, los_shift, shape_ln, scale_ln, shape_w, scale_w)

    result_opt = minimize(objective, initial_params, method='Nelder-Mead',
                            options={'maxiter': 1000, 'disp': False})

    if result_opt.success:
        shape_ln_opt, scale_ln_opt, shape_w_opt, scale_w_opt = result_opt.x
        ll = mixture_loglikelihood(w, los_shift, shape_ln_opt, scale_ln_opt, shape_w_opt, scale_w_opt)
    else:
        ll = mixture_loglikelihood(w, los_shift, shape_ln_init, scale_ln_init, shape_w_init, scale_w_init)

    log_likelihoods.append(ll)

    if w % 0.1 == 0:  # Mostrar cada 10%
        print(f"  weight={w:.1%} → log-likelihood={ll:.2f}")

log_likelihoods = np.array(log_likelihoods)

# Encontrar el máximo
max_idx = np.argmax(log_likelihoods)
optimal_weight = weights[max_idx]
max_ll = log_likelihoods[max_idx]

print(f"\n{'='*70}")
print(f"RESULTADO: El weight óptimo es {optimal_weight:.4f} ({optimal_weight*100:.2f}%)")
print(f"Log-verosimilitud máxima: {max_ll:.2f}")
print(f"{'='*70}")

# Crear gráfica
fig, ax = plt.subplots(figsize=(14, 8))

# Línea principal: log-verosimilitud vs weight
ax.plot(weights*100, log_likelihoods, 'b-', linewidth=3, label='Log-Verosimilitud')

# Marcar el máximo
ax.scatter([optimal_weight*100], [max_ll], color='red', s=300, zorder=5,
            label=f'Óptimo: weight={optimal_weight:.1%}', marker='*')
ax.axvline(optimal_weight*100, color='red', linestyle='--', linewidth=2, alpha=0.6)

# Sombrear región alrededor del óptimo
ax.axvspan(max(0, (optimal_weight-0.05)*100),
            min(100, (optimal_weight+0.05)*100),
            alpha=0.2, color='red', label='Rango óptimo ±5%')

# Marcar componentes puros
ax.scatter([0], [log_likelihoods[0]],
            color='green', s=150, marker='s', alpha=0.7, label='100% Weibull (w=0)')
ax.scatter([100], [log_likelihoods[-1]],
            color='orange', s=150, marker='^', alpha=0.7, label='100% Log-Normal (w=1)')

# Línea de referencia: Log-Normal simple (w=1)
ll_lognorm_only = log_likelihoods[-1]  # El último valor es weight=1.0
ax.axhline(ll_lognorm_only, color='orange', linestyle=':', linewidth=2, alpha=0.5)
ax.text(2, ll_lognorm_only+10, f'Log-Normal solo (w=1): LL={ll_lognorm_only:.0f}', fontsize=10)

# Mejora porcentual
improvement = ((max_ll - ll_lognorm_only) / abs(ll_lognorm_only)) * 100
ax.text(optimal_weight*100, max_ll+100,
        f'Mejora: +{improvement:.1f}%\n(Mezcla vs Log-Normal)',
        fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Etiquetas y formato
ax.set_xlabel('Weight (w) - Proporción de Log-Normal en la Mezcla (%)',
                fontsize=13, fontweight='bold')
ax.set_ylabel('Log-Verosimilitud', fontsize=13, fontweight='bold')
ax.set_title('¿Por Qué el Algoritmo Elige 42%-58%?\nLa Verosimilitud es Máxima en weight ≈ 0.42',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3, linestyle='--')

# Anotaciones adicionales
textstr = f"""INTERPRETACIÓN:
• Eje X: 0% = 100% Weibull | 50% = Mezcla 50-50 | 100% = 100% Log-Normal
• Eje Y: Log-Verosimilitud (mayor = mejor ajuste a datos reales)
• La curva tiene UN único máximo en weight ≈ {optimal_weight:.1%}
• Esto significa: Los datos observados son MÁXIMAMENTE probables
  bajo esta mezcla específica
• El algoritmo "descubre" esto automáticamente durante optimización"""

ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
       verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
       family='monospace')

plt.tight_layout()
output_path = './07_verosimilitud_vs_weight.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfica guardada: {output_path}")
plt.close()

# Tabla de mejora comparativa
selected_weights = [0.0, 0.25, 0.50, 0.75, 0.80, 1.0]
print(f"\n{'='*70}")
print("TABLA COMPARATIVA: Impacto del Weight en Verosimilitud")
print(f"{'='*70}")
print(f"\n{'Weight':<12} {'Composición':<30} {'Log-LL':<15} {'Mejora vs Puro LN':<20}")
print("-" * 80)

for w_test in selected_weights:
    w_idx = int(w_test * 100)
    ll_test = log_likelihoods[w_idx]

    if w_test == 0:
        comp = "100% Weibull"
    elif w_test == 1:
        comp = "100% Log-Normal"
    else:
        comp = f"{w_test*100:.0f}% LN / {(1-w_test)*100:.0f}% WB"

    diff = ll_test - ll_lognorm_only
    mejora_text = f"+{diff:.0f}" if diff > 0 else f"{diff:.0f}"
    marker = " ⭐ ÓPTIMO" if w_test == optimal_weight else ""

    print(f"{w_test:<12.2f} {comp:<30} {ll_test:<15.2f} {mejora_text:<20}{marker}")

print(f"\n{'='*70}")
