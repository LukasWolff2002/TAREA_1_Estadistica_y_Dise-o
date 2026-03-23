import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

# =====================================================
# Parámetros generales
# =====================================================
mu = 5
sigma = 2
sigma2_real = sigma**2
n = 100
alpha = 0.05
gl = n - 1

# Semilla pedida
rng = np.random.default_rng(200)

# Cuantiles chi-cuadrado
chi2_inf = chi2.ppf(alpha / 2, df=gl)
chi2_sup = chi2.ppf(1 - alpha / 2, df=gl)

# =====================================================
# PARTE 1: Primera muestra y gráfico original
# =====================================================

# -----------------------------
# Generación de la muestra
# -----------------------------
muestra = rng.normal(loc=mu, scale=sigma, size=n)

# -----------------------------
# Estadísticos muestrales
# -----------------------------
media_muestral = np.mean(muestra)
S2_n = np.var(muestra, ddof=1)
s_muestral = np.sqrt(S2_n)

# -----------------------------
# Intervalo de confianza 95% para sigma^2
# -----------------------------
LI = (gl * S2_n) / chi2_sup
LS = (gl * S2_n) / chi2_inf

# -----------------------------
# Mostrar resultados de la primera muestra
# -----------------------------
print("=== Primera muestra ===")
print(f"Media muestral = {media_muestral:.4f}")
print(f"Varianza muestral S_n^2 = {S2_n:.4f}")
print(f"IC 95% para sigma^2 = [{LI:.4f}, {LS:.4f}]")
print(f"Varianza real = {sigma2_real:.4f}")

if LI <= sigma2_real <= LS:
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
ax1.hist(muestra, bins=15, density=True, alpha=0.6, edgecolor='black',
         label='Histograma de la muestra')

x = np.linspace(min(muestra) - 1, max(muestra) + 1, 400)

# Curva normal teórica
ax1.plot(x, norm.pdf(x, loc=mu, scale=sigma), linewidth=2,
         label='Normal teórica')

# Curva normal ajustada a la muestra
ax1.plot(x, norm.pdf(x, loc=media_muestral, scale=s_muestral),
         linestyle='--', linewidth=2,
         label='Normal ajustada a la muestra')

ax1.axvline(media_muestral, linestyle=':', linewidth=2,
            label=f'Media muestral = {media_muestral:.2f}')

ax1.set_title('Distribución de la muestra')
ax1.set_xlabel('Valores de X')
ax1.set_ylabel('Densidad')
ax1.legend()
ax1.grid(alpha=0.3)

# ===== Panel 2: intervalo para la varianza =====
ax2.hlines(y=1, xmin=LI, xmax=LS, linewidth=4,
           label='IC 95% para $\sigma^2$')
ax2.plot([LI, LS], [1, 1], 'o')
ax2.axvline(sigma2_real, linestyle='--', linewidth=2,
            label='Varianza real $\sigma^2$')

ax2.set_yticks([])
ax2.set_xlabel('Valor de la varianza')
ax2.set_title('Intervalo de confianza al 95% para la varianza poblacional')
ax2.legend()
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# PARTE 2: Repetir 100 veces y graficar los intervalos
# =====================================================

def intervalo_varianza(rng, mu, sigma, n, gl, chi2_inf, chi2_sup):
    muestra = rng.normal(loc=mu, scale=sigma, size=n)
    S2 = np.var(muestra, ddof=1)

    LI = (gl * S2) / chi2_sup
    LS = (gl * S2) / chi2_inf
    mediana = (LI + LS) / 2
    contiene = (LI <= sigma**2 <= LS)

    return {
        "S2": S2,
        "LI": LI,
        "LS": LS,
        "mediana": mediana,
        "contiene": contiene
    }

# Generar 1000 intervalos adicionales
resultados = []
for i in range(1000):
    resultados.append(intervalo_varianza(rng, mu, sigma, n, gl, chi2_inf, chi2_sup))

# Tomar los primeros 100 para graficar
primeros_100 = resultados[:100]

# Ordenar por mediana del intervalo
primeros_100_ordenados = sorted(primeros_100, key=lambda r: r["mediana"])

# -----------------------------
# Figura 2: 100 intervalos contiguos
# -----------------------------
plt.figure(figsize=(10, 12))

for i, r in enumerate(primeros_100_ordenados, start=1):
    color = 'tab:blue' if r["contiene"] else 'tab:red'
    plt.hlines(y=i, xmin=r["LI"], xmax=r["LS"], linewidth=2, color=color)
    plt.plot([r["LI"], r["LS"]], [i, i], 'o', color=color, markersize=4)

plt.axvline(sigma2_real, color='black', linestyle='--', linewidth=2,
            label=fr'Varianza real $\sigma^2 = {sigma2_real:.2f}$')

plt.xlabel(r'Valor del intervalo para $\sigma^2$')
plt.ylabel('Intervalos ordenados')
plt.title('Primeros 100 intervalos de confianza al 95% para la varianza')
plt.grid(axis='x', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# PARTE 3: Conteo total en 1000 repeticiones
# =====================================================
dentro = sum(r["contiene"] for r in resultados)
fuera = len(resultados) - dentro

print("\n=== Cobertura en 1000 intervalos ===")
print(f"Cantidad de intervalos que contienen sigma^2: {dentro}")
print(f"Cantidad de intervalos que NO contienen sigma^2: {fuera}")
print(f"Proporción de cobertura observada: {dentro / len(resultados):.4f}")