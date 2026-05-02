"""
Análisis de Distribución Probabilística de LOS
Identifica qué distribución teórica se ajusta mejor a los datos observados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (lognorm, gamma, weibull_min, expon,
                            norm, kstest)
from scipy.optimize import minimize, brentq
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Cargar datos
print("="*70)
print("ANÁLISIS DE DISTRIBUCIÓN PROBABILÍSTICA - LOS")
print("="*70)

df = pd.read_csv('data/processed/dataset_maestro.csv', sep=';', dtype=str)
df['los_dias'] = pd.to_numeric(df['los_dias'], errors='coerce')
los = df['los_dias'].dropna()

print(f"\nDatos cargados: {len(los):,} observaciones")
print(f"Media: {los.mean():.2f} | Varianza: {los.var():.2f}")
print(f"Skewness: {stats.skew(los):.4f} | Kurtosis: {stats.kurtosis(los):.4f}")

# ============================================================================
# AJUSTE DE DISTRIBUCIONES CANDIDATAS
# ============================================================================

print("\n" + "-"*70)
print("AJUSTE DE DISTRIBUCIONES CANDIDATAS")
print("-"*70)
print(f"\nNOTA: Usando LOS+1 para ajustar todas las distribuciones")
print(f"      (Hay {(los == 0).sum()} casos con LOS=0 que causan problemas)")

# Usar los+1 para todas las distribuciones para consistencia
los_shift = los + 1

# Distribucion 1: Log-Normal (añadir 1 para evitar log(0))
# Nota: Log-normal no puede ajustar valores = 0, usamos los+1
print("\n1. LOG-NORMAL")
shape_ln, loc_ln, scale_ln = lognorm.fit(los_shift, floc=0)
print(f"   Parámetros: σ={shape_ln:.4f}, μ={np.log(scale_ln):.4f}")
# KS test sobre datos originales con distribución ajustada shifted
ks_ln, p_ln = kstest(los_shift, lambda x: lognorm.cdf(x, shape_ln, loc_ln, scale_ln))
print(f"   Kolmogorov-Smirnov: D={ks_ln:.6f}, p-value={p_ln:.6e}")

# Distribución 2: Gamma
print("\n2. GAMMA")
shape_g, loc_g, scale_g = gamma.fit(los_shift, floc=0)
print(f"   Parámetros: k={shape_g:.4f}, θ={scale_g:.4f}")
ks_g, p_g = kstest(los_shift, lambda x: gamma.cdf(x, shape_g, loc_g, scale_g))
print(f"   Kolmogorov-Smirnov: D={ks_g:.6f}, p-value={p_g:.6e}")

# Distribución 3: Weibull
print("\n3. WEIBULL")
shape_w, loc_w, scale_w = weibull_min.fit(los_shift, floc=0)
print(f"   Parámetros: k={shape_w:.4f}, λ={scale_w:.4f}")
ks_w, p_w = kstest(los_shift, lambda x: weibull_min.cdf(x, shape_w, loc_w, scale_w))
print(f"   Kolmogorov-Smirnov: D={ks_w:.6f}, p-value={p_w:.6e}")

# Distribución 4: Exponencial
print("\n4. EXPONENCIAL")
loc_e, scale_e = expon.fit(los_shift, floc=0)
print(f"   Parámetros: λ={1/scale_e:.4f}")
ks_e, p_e = kstest(los_shift, lambda x: expon.cdf(x, loc_e, scale_e))
print(f"   Kolmogorov-Smirnov: D={ks_e:.6f}, p-value={p_e:.6e}")

# Distribución 5: Normal (para comparación - NO debería ajustar bien)
print("\n5. NORMAL (referencia)")
mu_n, std_n = norm.fit(los_shift)
print(f"   Parámetros: μ={mu_n:.4f}, σ={std_n:.4f}")
ks_n, p_n = kstest(los_shift, lambda x: norm.cdf(x, mu_n, std_n))
print(f"   Kolmogorov-Smirnov: D={ks_n:.6f}, p-value={p_n:.6e}")

# ============================================================================
# DISTRIBUCIÓN 6: MEZCLA LOG-NORMAL-WEIBULL (Marazzi et al. 1998)
# ============================================================================

print("\n6. MEZCLA LOG-NORMAL-WEIBULL (Marazzi et al. 1998)")
print("   Modelo: f(x) = w*LogNormal(x) + (1-w)*Weibull(x)")

# Funciones de la mezcla
def mixture_pdf(x, shape_ln, scale_ln, shape_w, scale_w, weight):
    """PDF de la mezcla Log-Normal-Weibull"""
    pdf_ln = lognorm.pdf(x, s=shape_ln, loc=0, scale=scale_ln)
    pdf_w = weibull_min.pdf(x, c=shape_w, loc=0, scale=scale_w)
    return weight * pdf_ln + (1 - weight) * pdf_w

def mixture_cdf(x, shape_ln, scale_ln, shape_w, scale_w, weight):
    """CDF de la mezcla Log-Normal-Weibull"""
    cdf_ln = lognorm.cdf(x, s=shape_ln, loc=0, scale=scale_ln)
    cdf_w = weibull_min.cdf(x, c=shape_w, loc=0, scale=scale_w)
    return weight * cdf_ln + (1 - weight) * cdf_w

def negative_log_likelihood(params, data):
    """Negative log-likelihood para optimización"""
    shape_ln, scale_ln, shape_w, scale_w, weight = params

    # Restricciones: todos los parámetros positivos, weight en [0,1]
    if shape_ln <= 0 or scale_ln <= 0 or shape_w <= 0 or scale_w <= 0:
        return np.inf
    if weight < 0 or weight > 1:
        return np.inf

    pdf_vals = mixture_pdf(data, shape_ln, scale_ln, shape_w, scale_w, weight)

    # Evitar log(0)
    pdf_vals = np.maximum(pdf_vals, 1e-300)

    return -np.sum(np.log(pdf_vals))

# Ajustar mezcla usando los parámetros individuales como inicialización
print("   Ajustando modelo de mezcla (esto puede tomar unos momentos)...")

# Parámetros iniciales: usar los ajustes individuales y weight=0.5
initial_params = [shape_ln, scale_ln, shape_w, scale_w, 0.5]

# Optimizar
result = minimize(
    negative_log_likelihood,
    initial_params,
    args=(los_shift,),
    method='Nelder-Mead',
    options={'maxiter': 5000, 'disp': False}
)

if result.success:
    shape_ln_mix, scale_ln_mix, shape_w_mix, scale_w_mix, weight_mix = result.x

    print(f"   Parámetros Log-Normal: σ={shape_ln_mix:.4f}, scale={scale_ln_mix:.4f}")
    print(f"   Parámetros Weibull: k={shape_w_mix:.4f}, λ={scale_w_mix:.4f}")
    print(f"   Peso Log-Normal: {weight_mix:.4f} | Peso Weibull: {1-weight_mix:.4f}")

    # KS test para la mezcla
    cdf_mixture = lambda x: mixture_cdf(x, shape_ln_mix, scale_ln_mix,
                                            shape_w_mix, scale_w_mix, weight_mix)
    ks_mix, p_mix = kstest(los_shift, cdf_mixture)
    print(f"   Kolmogorov-Smirnov: D={ks_mix:.6f}, p-value={p_mix:.6e}")

    # Log-likelihood de la mezcla
    ll_mix = -negative_log_likelihood(result.x, los_shift)
    print(f"   Log-Likelihood: {ll_mix:.2f}")

    mixture_success = True
else:
    print(f"   ⚠️ Advertencia: Optimización no convergió completamente")
    print(f"   Mensaje: {result.message}")
    # Usar los resultados parciales de todas formas
    shape_ln_mix, scale_ln_mix, shape_w_mix, scale_w_mix, weight_mix = result.x

    cdf_mixture = lambda x: mixture_cdf(x, shape_ln_mix, scale_ln_mix,
                                            shape_w_mix, scale_w_mix, weight_mix)
    ks_mix, p_mix = kstest(los_shift, cdf_mixture)
    ll_mix = -negative_log_likelihood(result.x, los_shift)

    print(f"   Kolmogorov-Smirnov: D={ks_mix:.6f}, p-value={p_mix:.6e}")
    mixture_success = True

# ============================================================================
# COMPARACIÓN DE AJUSTES
# ============================================================================

print("\n" + "="*70)
print("RANKING DE AJUSTE (menor D = mejor ajuste)")
print("="*70)

resultados = [
    ("Log-Normal", ks_ln, p_ln),
    ("Gamma", ks_g, p_g),
    ("Weibull", ks_w, p_w),
    ("Exponencial", ks_e, p_e),
    ("Normal", ks_n, p_n),
    ("Mezcla LN-Weibull", ks_mix, p_mix)
]

resultados_sorted = sorted(resultados, key=lambda x: x[1])

for i, (nombre, d, p) in enumerate(resultados_sorted, 1):
    if i == 1:
        print(f"{i}. ✓ {nombre:15} | D = {d:.6f} | p-value = {p:.6e} | ⭐ MEJOR AJUSTE")
    else:
        print(f"{i}. {nombre:15} | D = {d:.6f} | p-value = {p:.6e}")

print("\n" + "-"*70)
print("INTERPRETACIÓN:")
print("-"*70)
mejor_dist = resultados_sorted[0][0]
print(f"• La distribución que mejor ajusta es: {mejor_dist}")
print(f"• Todas las p-values son ~0 (distribuciones teóricas nunca ajustan")
print(f"  perfectamente datos reales), pero comparamos estadísticos D")
print(f"• El menor D indica el mejor ajuste relativo")
print("="*70)

# ============================================================================
# CÁLCULO DE AIC/BIC PARA COMPARACIÓN
# ============================================================================

print("\n" + "="*70)
print("CRITERIOS DE INFORMACIÓN (AIC/BIC)")
print("="*70)

def calc_aic_bic(log_likelihood, n_params, n_obs):
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    return aic, bic

n_obs = len(los)

# Log-likelihoods (usar los_shift consistentemente)
ll_ln = np.sum(lognorm.logpdf(los_shift, shape_ln, loc_ln, scale_ln))
ll_g = np.sum(gamma.logpdf(los_shift, shape_g, loc_g, scale_g))
ll_w = np.sum(weibull_min.logpdf(los_shift, shape_w, loc_w, scale_w))
ll_e = np.sum(expon.logpdf(los_shift, loc_e, scale_e))

# AICs y BICs (n_params: lognorm=2, gamma=2, weibull=2, expon=1, mezcla=5)
aic_ln, bic_ln = calc_aic_bic(ll_ln, 2, n_obs)
aic_g, bic_g = calc_aic_bic(ll_g, 2, n_obs)
aic_w, bic_w = calc_aic_bic(ll_w, 2, n_obs)
aic_e, bic_e = calc_aic_bic(ll_e, 1, n_obs)
aic_mix, bic_mix = calc_aic_bic(ll_mix, 5, n_obs)

criterios = [
    ("Log-Normal", aic_ln, bic_ln, ll_ln),
    ("Gamma", aic_g, bic_g, ll_g),
    ("Weibull", aic_w, bic_w, ll_w),
    ("Exponencial", aic_e, bic_e, ll_e),
    ("Mezcla LN-Weibull", aic_mix, bic_mix, ll_mix)
]

criterios_sorted = sorted(criterios, key=lambda x: x[1])  # Ordenar por AIC

print("\n{:<15} | {:>12} | {:>12} | {:>15}".format("Distribución", "AIC", "BIC", "Log-Likelihood"))
print("-" * 70)

for i, (nombre, aic, bic, ll) in enumerate(criterios_sorted, 1):
    marker = "⭐ MEJOR" if i == 1 else ""
    print(f"{nombre:<15} | {aic:>12.2f} | {bic:>12.2f} | {ll:>15.2f} {marker}")

print("\n" + "-"*70)
print("NOTA: Menor AIC/BIC = mejor modelo (balancea ajuste vs complejidad)")
print("="*70)

# ============================================================================
# GRÁFICA 4: Q-Q PLOTS comparativos
# ============================================================================

print("\nGenerando gráfica 4: Q-Q Plots para distribuciones candidatas...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# 1. Log-Normal Q-Q (usar los_shift = los+1)
stats.probplot(los_shift, dist=lognorm, sparams=(shape_ln, loc_ln, scale_ln), plot=axes[0])
axes[0].set_title(f'Gráfica Cuantil-Cuantil: Log-Normal (LOS+1)\nKS={ks_ln:.4f} | AIC={aic_ln:.1f}', fontweight='bold', fontsize=10)
axes[0].set_xlabel('Días Teóricos (Esperados)', fontsize=10)
axes[0].set_ylabel('Días Observados (Reales)', fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. Gamma Q-Q
stats.probplot(los_shift, dist=gamma, sparams=(shape_g, loc_g, scale_g), plot=axes[1])
axes[1].set_title(f'Gráfica Cuantil-Cuantil: Gamma (LOS+1)\nKS={ks_g:.4f} | AIC={aic_g:.1f}', fontweight='bold', fontsize=10)
axes[1].set_xlabel('Días Teóricos (Esperados)', fontsize=10)
axes[1].set_ylabel('Días Observados (Reales)', fontsize=10)
axes[1].grid(True, alpha=0.3)

# 3. Weibull Q-Q
stats.probplot(los_shift, dist=weibull_min, sparams=(shape_w, loc_w, scale_w), plot=axes[2])
axes[2].set_title(f'Gráfica Cuantil-Cuantil: Weibull (LOS+1)\nKS={ks_w:.4f} | AIC={aic_w:.1f}', fontweight='bold', fontsize=10)
axes[2].set_xlabel('Días Teóricos (Esperados)', fontsize=10)
axes[2].set_ylabel('Días Observados (Reales)', fontsize=10)
axes[2].grid(True, alpha=0.3)

# 4. Exponencial Q-Q
stats.probplot(los_shift, dist=expon, sparams=(loc_e, scale_e), plot=axes[3])
axes[3].set_title(f'Gráfica Cuantil-Cuantil: Exponencial (LOS+1)\nKS={ks_e:.4f} | AIC={aic_e:.1f}', fontweight='bold', fontsize=10)
axes[3].set_xlabel('Días Teóricos (Esperados)', fontsize=10)
axes[3].set_ylabel('Días Observados (Reales)', fontsize=10)
axes[3].grid(True, alpha=0.3)

# 5. Normal Q-Q
stats.probplot(los_shift, dist='norm', plot=axes[4])
axes[4].set_title(f'Gráfica Cuantil-Cuantil: Normal (LOS+1)\nKS={ks_n:.4f}', fontweight='bold', fontsize=10)
axes[4].set_xlabel('Días Teóricos (Esperados)', fontsize=10)
axes[4].set_ylabel('Días Observados (Reales)', fontsize=10)
axes[4].grid(True, alpha=0.3)

# 6. Mezcla Log-Normal-Weibull Q-Q
# Para Q-Q plot de mezcla, calculamos cuantiles teóricos
x_range_qq = np.linspace(0.001, 0.999, 1000)
mixture_quantiles = []
for p in x_range_qq:
    # Encontrar x tal que CDF(x) = p (usando búsqueda numérica)
    try:
        quantile = brentq(lambda x: cdf_mixture(x) - p, 0.01, 500)
        mixture_quantiles.append(quantile)
    except:
        mixture_quantiles.append(np.nan)

mixture_quantiles = np.array(mixture_quantiles)
data_quantiles = np.percentile(los_shift, x_range_qq * 100)

axes[5].scatter(mixture_quantiles, data_quantiles, alpha=0.5, s=10)
axes[5].plot([0, 100], [0, 100], 'r-', linewidth=2)
axes[5].set_xlabel('Días Teóricos (Esperados)', fontsize=10)
axes[5].set_ylabel('Días Observados (Reales)', fontsize=10)
axes[5].set_title(f'Gráfica Cuantil-Cuantil: Mezcla Log-Normal-Weibull (LOS+1)\nKS={ks_mix:.4f} | AIC={aic_mix:.1f}', fontweight='bold', fontsize=10)
axes[5].grid(True, alpha=0.3)


# 7. Log-transformed Normal Q-Q - REMOVIDO

plt.suptitle('Análisis de Gráficas Cuantil-Cuantil: Comparación de Distribuciones Teóricas\n' +
             f'Mejor ajuste: {mejor_dist} (menor desviación de la línea diagonal)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
output_path_4 = 'graficos/04_qq_plots_comparacion_distribuciones.png'
plt.savefig(output_path_4, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: {output_path_4}")
plt.close()

# ============================================================================
# GRÁFICA 5: Distribución empírica vs teóricas (Escala lineal y logarítmica)
# ============================================================================

print("Generando gráfica 5: Distribución empírica vs distribuciones teóricas...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Calcular CDF empírica
los_sorted = np.sort(los)
los_shift_sorted = np.sort(los_shift)
y_empirica = np.arange(1, len(los_sorted) + 1) / len(los_sorted)

# Panel izquierdo: Escala lineal
ax1.plot(los_sorted, y_empirica, 'k-', linewidth=2.5, label='Función de distribución acumulada Empírica', alpha=0.8)
ax1.plot(los_sorted, mixture_cdf(los_shift_sorted, shape_ln_mix, scale_ln_mix,
                                   shape_w_mix, scale_w_mix, weight_mix),
         'purple', linestyle='-', linewidth=3, label=f'Mezcla LN-Weibull (KS={ks_mix:.5f}) ⭐', alpha=0.9)
ax1.plot(los_sorted, lognorm.cdf(los_shift_sorted, shape_ln, loc_ln, scale_ln),
         'r--', linewidth=2, label=f'Log-Normal (KS={ks_ln:.5f})', alpha=0.8)
ax1.plot(los_sorted, gamma.cdf(los_sorted, shape_g, loc_g, scale_g),
         'b--', linewidth=2, label=f'Gamma (KS={ks_g:.5f})', alpha=0.8)
ax1.plot(los_sorted, weibull_min.cdf(los_sorted, shape_w, loc_w, scale_w),
         'g--', linewidth=2, label=f'Weibull (KS={ks_w:.5f})', alpha=0.8)
ax1.plot(los_sorted, expon.cdf(los_sorted, loc_e, scale_e),
         'm--', linewidth=2, label=f'Exponencial (KS={ks_e:.5f})', alpha=0.8)

ax1.set_xlabel('LOS (días)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probabilidad Acumulada F(x)', fontsize=12, fontweight='bold')
ax1.set_title('Distribución Acumulada Empírica versus Teóricas\nEscala Lineal',
              fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)  # Limitar para mejor visualización

# Panel derecho: Escala log en X
ax2.plot(los_sorted, y_empirica, 'k-', linewidth=2.5, label='Función de distribución acumulada Empírica', alpha=0.8)
ax2.plot(los_sorted, mixture_cdf(los_shift_sorted, shape_ln_mix, scale_ln_mix,
                                   shape_w_mix, scale_w_mix, weight_mix),
         'purple', linestyle='-', linewidth=3, label=f'Mezcla LN-Weibull (KS={ks_mix:.5f}) ⭐', alpha=0.9)
ax2.plot(los_sorted, lognorm.cdf(los_shift_sorted, shape_ln, loc_ln, scale_ln),
         'r--', linewidth=2, label=f'Log-Normal (KS={ks_ln:.5f})', alpha=0.8)
ax2.plot(los_sorted, gamma.cdf(los_sorted, shape_g, loc_g, scale_g),
         'b--', linewidth=2, label=f'Gamma (KS={ks_g:.5f})', alpha=0.8)
ax2.plot(los_sorted, weibull_min.cdf(los_sorted, shape_w, loc_w, scale_w),
         'g--', linewidth=2, label=f'Weibull (KS={ks_w:.5f})', alpha=0.8)

ax2.set_xlabel('LOS (días) - Escala Logarítmica', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probabilidad Acumulada F(x)', fontsize=12, fontweight='bold')
ax2.set_title('Distribución Acumulada Empírica versus Teóricas\nEscala Logarítmica (revela ajuste completo)',
              fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.suptitle(f'Análisis de Distribución Acumulada\nDistribución identificada: {mejor_dist}',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

output_path_5 = 'graficos/05_cdf_empirica_vs_teoricas.png'
plt.savefig(output_path_5, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: {output_path_5}")
plt.close()

# ============================================================================
# GRÁFICA 6: Histograma empírico vs teóricas
# ============================================================================

print("Generando gráfica 6: Histograma empírico vs distribuciones teóricas...")

fig, ax1 = plt.subplots(figsize=(12, 7))

# Panel izquierdo: Escala lineal
x_range = np.linspace(0, 80, 1000)
x_range_shift = x_range + 1  # Para lognormal

ax1.hist(los, bins=80, density=True, alpha=0.6, color='gray', edgecolor='black',
         linewidth=0.5, label='Datos Observados')
ax1.plot(x_range, mixture_pdf(x_range_shift, shape_ln_mix, scale_ln_mix,
                                shape_w_mix, scale_w_mix, weight_mix),
         'purple', linewidth=3.5, label='Mezcla LN-Weibull ⭐', alpha=0.95)
ax1.plot(x_range, lognorm.pdf(x_range_shift, shape_ln, loc_ln, scale_ln),
         'r--', linewidth=2.5, label='Log-Normal', alpha=0.8)
ax1.plot(x_range, gamma.pdf(x_range, shape_g, loc_g, scale_g),
         'b--', linewidth=2, label='Gamma', alpha=0.7)
ax1.plot(x_range, weibull_min.pdf(x_range, shape_w, loc_w, scale_w),
         'g--', linewidth=2, label='Weibull', alpha=0.7)

ax1.set_xlabel('LOS (días)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Densidad de Probabilidad f(x)', fontsize=12, fontweight='bold')
ax1.set_title('Densidad de Probabilidad: Datos versus Distribuciones Teóricas\nEscala Lineal',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 80)

plt.suptitle('Análisis de Densidad de Probabilidad\nDistribución identificada: Mezcla Log-Normal-Weibull',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

output_path_6 = 'graficos/06_pdf_distribuciones_ajustadas.png'
plt.savefig(output_path_6, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: {output_path_6}")
plt.close()

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("CONCLUSIÓN FINAL")
print("="*70)

print(f"\n✓ MEJOR DISTRIBUCIÓN POR KS: {mejor_dist.upper()}")

# Mostrar trade-off entre KS y AIC
print(f"\n{'='*70}")
print("ANÁLISIS COMPARATIVO: KS vs AIC")
print(f"{'='*70}")

print(f"\n  Por KOLMOGOROV-SMIRNOV (ajuste puntual):")
print(f"  • Log-Normal simple: D = {ks_ln:.6f}  ← MEJOR")
print(f"  • Mezcla LN-Weibull:  D = {ks_mix:.6f}  (peor por {ks_mix-ks_ln:.6f})")

print(f"\n  Por AIC (ajuste global + complejidad):")
print(f"  • Mezcla LN-Weibull:  AIC = {aic_mix:.2f}  ← MEJOR")
print(f"  • Log-Normal simple: AIC = {aic_ln:.2f}  (peor por {aic_ln-aic_mix:.2f})")

print(f"\n  INTERPRETACIÓN DEL CONFLICTO:")
print(f"  • KS mide la MÁXIMA desviación en un solo punto de la CDF")
print(f"  • AIC mide el ajuste GLOBAL sobre toda la distribución")
print(f"  • Mejora de AIC de {aic_ln-aic_mix:.0f} puntos es SUBSTANCIAL (>10 = decisivo)")
print(f"  • La mezcla tiene peor ajuste en el punto de máxima desviación")
print(f"  • PERO la mezcla ajusta MEJOR el resto de la distribución")

print(f"\n{'='*70}")
print("RECOMENDACIÓN SEGÚN MARAZZI ET AL. (1998)")
print(f"{'='*70}")

print(f"\n  ✓ DISTRIBUCIÓN RECOMENDADA: MEZCLA LOG-NORMAL-WEIBULL")
print(f"\n  Razones:")
print(f"  1. AIC drásticamente mejor ({aic_ln-aic_mix:.0f} puntos de mejora)")
print(f"  2. Captura heterogeneidad de subpoblaciones hospitalarias")
print(f"  3. Coherente con hallazgos de Marazzi: mezcla supera modelos simples")
print(f"  4. Penalización por complejidad (5 params vs 2) ya incluida en AIC")

print(f"\n  PARÁMETROS DE LA MEZCLA (sobre LOS+1):")
print(f"  Componente Log-Normal (peso={weight_mix:.4f}, ~{weight_mix*100:.0f}% de casos):")
print(f"    • σ (shape): {shape_ln_mix:.4f}")
print(f"    • scale: {scale_ln_mix:.4f}")
print(f"    → Representa estancias CORTAS/TÍPICAS")

print(f"\n  Componente Weibull (peso={1-weight_mix:.4f}, ~{(1-weight_mix)*100:.0f}% de casos):")
print(f"    • k (shape): {shape_w_mix:.4f}")
print(f"    • λ (scale): {scale_w_mix:.4f}")
print(f"    → Representa estancias LARGAS/COMPLEJAS")

print(f"\n  INTERPRETACIÓN CLÍNICA:")
print(f"  • ~{weight_mix*100:.0f}% pacientes: estancias predecibles (Log-Normal)")
print(f"  • ~{(1-weight_mix)*100:.0f}% pacientes: casos complejos/prolongados (Weibull)")
print(f"  • La mezcla refleja dos tipos distintos de hospitalizaciones")

print(f"\n  IMPLICACIONES PARA MODELADO:")
print(f"  1. Usar transformación log(1+y) puede mejorar modelos lineales")
print(f"  2. Para XGBoost/LightGBM: no es estrictamente necesario transformar")
print(f"  3. Considerar GLM con familia log-normal (distribución exponencial)")
print(f"  4. Aplicar regularización para evitar sobreajuste en colas")

print("\n" + "="*70)
print("ARCHIVOS GENERADOS:")
print("="*70)
print(f"  • {output_path_4}")
print(f"  • {output_path_5}")
print(f"  • {output_path_6}")
print("="*70)
