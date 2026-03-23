import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# =====================================================
# Parámetros generales
# =====================================================
rng = np.random.default_rng(200)

lam = 0.5          # tasa lambda != 1
n = 100
alpha = 0.05

media_real = 1 / lam
var_real = 1 / lam**2

# Cuantiles de la Gamma(n, 1)
# scipy usa shape=a y scale=1 por defecto
g_inf = gamma.ppf(alpha / 2, a=n, scale=1)
g_sup = gamma.ppf(1 - alpha / 2, a=n, scale=1)

# =====================================================
# PARTE 1: Primera muestra y gráfico original
# =====================================================

# -----------------------------
# Generación de la muestra
# -----------------------------
muestra = rng.exponential(scale=1/lam, size=n)

# -----------------------------
# Estadísticos muestrales
# -----------------------------
media_muestral = np.mean(muestra)
S2_n = np.var(muestra, ddof=1)

# Estimador de lambda
lam_hat = 1 / media_muestral

# -----------------------------
# IC exacto 95% para lambda
# usando: lambda * sum(Xi) ~ Gamma(n,1)
# -----------------------------
suma_muestral = np.sum(muestra)

LI_lam = g_inf / suma_muestral
LS_lam = g_sup / suma_muestral

# -----------------------------
# Transformación a IC para la varianza:
# Var(X) = 1 / lambda^2
# Ojo con el orden, porque la función 1/lambda^2 es decreciente
# -----------------------------
LI_var = 1 / (LS_lam**2)
LS_var = 1 / (LI_lam**2)

# -----------------------------
# Mostrar resultados
# -----------------------------
print("=== Primera muestra (Exponencial) ===")
print(f"Tasa real lambda = {lam:.4f}")
print(f"Media muestral = {media_muestral:.4f}")
print(f"Varianza muestral S_n^2 = {S2_n:.4f}")
print(f"IC 95% para lambda = [{LI_lam:.4f}, {LS_lam:.4f}]")
print(f"IC 95% para la varianza = [{LI_var:.4f}, {LS_var:.4f}]")
print(f"Varianza real = {var_real:.4f}")

if LI_var <= var_real <= LS_var:
    print("La varianza real está DENTRO del intervalo.")
else:
    print("La varianza real está FUERA del intervalo.")

# -----------------------------
# Figura 1: gráfico original con dos paneles
# -----------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [3, 1]}
)

# ===== Panel 1: distribución de los datos =====
ax1.hist(
    muestra, bins=15, density=True, alpha=0.6,
    edgecolor='black', label='Histograma de la muestra'
)

x = np.linspace(0, max(muestra) + 2, 400)

# Curva exponencial teórica
ax1.plot(
    x, lam * np.exp(-lam * x),
    linewidth=2, label='Exponencial teórica'
)

# Curva exponencial ajustada a la muestra
ax1.plot(
    x, lam_hat * np.exp(-lam_hat * x),
    linestyle='--', linewidth=2,
    label='Exponencial ajustada a la muestra'
)

ax1.axvline(
    media_muestral, linestyle=':', linewidth=2,
    label=f'Media muestral = {media_muestral:.2f}'
)

ax1.set_title('Distribución de la muestra')
ax1.set_xlabel('Valores de X')
ax1.set_ylabel('Densidad')
ax1.legend()
ax1.grid(alpha=0.3)

# ===== Panel 2: intervalo para la varianza =====
ax2.hlines(y=1, xmin=LI_var, xmax=LS_var, linewidth=4,
           label='IC 95% para la varianza')
ax2.plot([LI_var, LS_var], [1, 1], 'o')
ax2.axvline(var_real, linestyle='--', linewidth=2,
            label='Varianza real')

ax2.set_yticks([])
ax2.set_xlabel('Valor de la varianza')
ax2.set_title('Intervalo de confianza al 95% para la varianza poblacional')
ax2.legend()
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# PARTE 2: Repetir 1000 veces
# =====================================================

def intervalo_varianza_exp(rng, lam, n, g_inf, g_sup):
    muestra = rng.exponential(scale=1/lam, size=n)

    suma_x = np.sum(muestra)
    S2 = np.var(muestra, ddof=1)

    # IC para lambda
    LI_lam = g_inf / suma_x
    LS_lam = g_sup / suma_x

    # IC para varianza = 1/lambda^2
    LI_var = 1 / (LS_lam**2)
    LS_var = 1 / (LI_lam**2)

    mediana = (LI_var + LS_var) / 2
    contiene = (LI_var <= 1/lam**2 <= LS_var)

    return {
        "S2": S2,
        "LI": LI_var,
        "LS": LS_var,
        "mediana": mediana,
        "contiene": contiene
    }

resultados = []
for _ in range(1000):
    resultados.append(intervalo_varianza_exp(rng, lam, n, g_inf, g_sup))

# =====================================================
# PARTE 3: Graficar los primeros 100 intervalos
# =====================================================
primeros_100 = resultados[:100]
primeros_100_ordenados = sorted(primeros_100, key=lambda r: r["mediana"])

plt.figure(figsize=(10, 12))

for i, r in enumerate(primeros_100_ordenados, start=1):
    color = 'tab:blue' if r["contiene"] else 'tab:red'
    plt.hlines(y=i, xmin=r["LI"], xmax=r["LS"], linewidth=2, color=color)
    plt.plot([r["LI"], r["LS"]], [i, i], 'o', color=color, markersize=4)

plt.axvline(var_real, color='black', linestyle='--', linewidth=2,
            label=fr'Varianza real = {var_real:.2f}')

plt.xlabel('Valor del intervalo para la varianza')
plt.ylabel('Intervalos ordenados')
plt.title('Primeros 100 intervalos de confianza al 95% para la varianza')
plt.grid(axis='x', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# PARTE 4: Conteo total en 1000 repeticiones
# =====================================================
dentro = sum(r["contiene"] for r in resultados)
fuera = len(resultados) - dentro

print("\n=== Cobertura en 1000 intervalos ===")
print(f"Cantidad de intervalos que contienen la varianza real: {dentro}")
print(f"Cantidad de intervalos que NO contienen la varianza real: {fuera}")
print(f"Proporción de cobertura observada: {dentro / len(resultados):.4f}")