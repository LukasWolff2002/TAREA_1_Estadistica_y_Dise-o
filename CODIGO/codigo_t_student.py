import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# =====================================================
# Parámetros generales
# =====================================================
rng = np.random.default_rng(200)

nu = 5              # grados de libertad, nu > 2
n = 100
alpha = 0.05

var_real = nu / (nu - 2)

# Número de remuestras bootstrap
B = 2000

# =====================================================
# Función para IC bootstrap percentil de la varianza
# =====================================================
def ic_bootstrap_varianza(muestra, B, alpha, rng):
    n = len(muestra)
    vars_boot = np.empty(B)

    for b in range(B):
        remuestra = rng.choice(muestra, size=n, replace=True)
        vars_boot[b] = np.var(remuestra, ddof=1)

    LI = np.quantile(vars_boot, alpha / 2)
    LS = np.quantile(vars_boot, 1 - alpha / 2)

    return LI, LS, vars_boot

# =====================================================
# PARTE 1: Primera muestra y gráfico original
# =====================================================

# -----------------------------
# Generación de la muestra
# -----------------------------
muestra = rng.standard_t(df=nu, size=n)

# -----------------------------
# Estadísticos muestrales
# -----------------------------
media_muestral = np.mean(muestra)
S2_n = np.var(muestra, ddof=1)
s_muestral = np.sqrt(S2_n)

# -----------------------------
# IC bootstrap 95% para la varianza
# -----------------------------
LI, LS, _ = ic_bootstrap_varianza(muestra, B, alpha, rng)

# -----------------------------
# Mostrar resultados de la primera muestra
# -----------------------------
print("=== Primera muestra (t de Student) ===")
print(f"Grados de libertad nu = {nu}")
print(f"Media muestral = {media_muestral:.4f}")
print(f"Varianza muestral S_n^2 = {S2_n:.4f}")
print(f"IC bootstrap 95% para la varianza = [{LI:.4f}, {LS:.4f}]")
print(f"Varianza real = {var_real:.4f}")

if LI <= var_real <= LS:
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

x = np.linspace(min(muestra) - 1, max(muestra) + 1, 400)

# Curva t teórica
ax1.plot(
    x, t.pdf(x, df=nu),
    linewidth=2, label='t de Student teórica'
)

# Curva normal ajustada a la muestra
# (igual que en tus códigos anteriores, para comparación visual)
from scipy.stats import norm
ax1.plot(
    x, norm.pdf(x, loc=media_muestral, scale=s_muestral),
    linestyle='--', linewidth=2,
    label='Normal ajustada a la muestra'
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
ax2.hlines(y=1, xmin=LI, xmax=LS, linewidth=4,
           label='IC bootstrap 95% para la varianza')
ax2.plot([LI, LS], [1, 1], 'o')
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

def intervalo_varianza_t(rng, nu, n, B, alpha):
    muestra = rng.standard_t(df=nu, size=n)

    S2 = np.var(muestra, ddof=1)
    LI, LS, _ = ic_bootstrap_varianza(muestra, B, alpha, rng)

    mediana = (LI + LS) / 2
    contiene = (LI <= nu / (nu - 2) <= LS)

    return {
        "S2": S2,
        "LI": LI,
        "LS": LS,
        "mediana": mediana,
        "contiene": contiene
    }

resultados = []
for _ in range(1000):
    resultados.append(intervalo_varianza_t(rng, nu, n, B, alpha))

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