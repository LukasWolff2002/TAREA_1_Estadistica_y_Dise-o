import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, gamma, norm, t

# =====================================================
# Configuración general
# =====================================================
SEED = 200
N = 100
ALPHA = 0.05
N_REP = 1000
N_PLOT = 100
B_BOOT = 2000   # bootstrap para t de Student

rng = np.random.default_rng(SEED)

# =====================================================
# Funciones auxiliares
# =====================================================

def ic_var_normal(muestra, alpha):
    """
    IC exacto para la varianza en el caso Normal
    usando chi-cuadrado.
    """
    n = len(muestra)
    gl = n - 1
    S2 = np.var(muestra, ddof=1)

    chi2_inf = chi2.ppf(alpha / 2, df=gl)
    chi2_sup = chi2.ppf(1 - alpha / 2, df=gl)

    LI = (gl * S2) / chi2_sup
    LS = (gl * S2) / chi2_inf
    return S2, LI, LS


def ic_var_exponencial(muestra, lam, alpha):
    """
    IC exacto para la varianza en el caso Exponencial,
    construyendo primero un IC para lambda y luego
    transformando Var(X)=1/lambda^2.
    """
    n = len(muestra)
    S2 = np.var(muestra, ddof=1)
    suma_x = np.sum(muestra)

    g_inf = gamma.ppf(alpha / 2, a=n, scale=1)
    g_sup = gamma.ppf(1 - alpha / 2, a=n, scale=1)

    LI_lam = g_inf / suma_x
    LS_lam = g_sup / suma_x

    LI_var = 1 / (LS_lam**2)
    LS_var = 1 / (LI_lam**2)
    return S2, LI_var, LS_var


def ic_bootstrap_varianza(muestra, alpha, B, rng):
    """
    IC bootstrap percentil para la varianza muestral.
    """
    n = len(muestra)
    vars_boot = np.empty(B)

    for b in range(B):
        remuestra = rng.choice(muestra, size=n, replace=True)
        vars_boot[b] = np.var(remuestra, ddof=1)

    LI = np.quantile(vars_boot, alpha / 2)
    LS = np.quantile(vars_boot, 1 - alpha / 2)
    return LI, LS


def ic_var_student(muestra, nu, alpha, B, rng):
    """
    IC bootstrap percentil para la varianza en el caso t de Student.
    """
    S2 = np.var(muestra, ddof=1)
    LI, LS = ic_bootstrap_varianza(muestra, alpha, B, rng)
    return S2, LI, LS


def simular_normal(rng, mu, sigma, n, alpha, n_rep):
    sigma2_real = sigma**2
    resultados = []

    # Primera muestra
    muestra0 = rng.normal(loc=mu, scale=sigma, size=n)
    media0 = np.mean(muestra0)
    S20, LI0, LS0 = ic_var_normal(muestra0, alpha)
    s0 = np.sqrt(S20)

    for _ in range(n_rep):
        muestra = rng.normal(loc=mu, scale=sigma, size=n)
        S2, LI, LS = ic_var_normal(muestra, alpha)
        resultados.append({
            "S2": S2,
            "LI": LI,
            "LS": LS,
            "mid": (LI + LS) / 2,
            "contiene": LI <= sigma2_real <= LS
        })

    return {
        "nombre": "Normal",
        "muestra0": muestra0,
        "media0": media0,
        "S20": S20,
        "LI0": LI0,
        "LS0": LS0,
        "real_var": sigma2_real,
        "x_pdf": np.linspace(min(muestra0) - 1, max(muestra0) + 1, 400),
        "pdf_teorica": lambda x: norm.pdf(x, loc=mu, scale=sigma),
        "pdf_ajustada": lambda x: norm.pdf(x, loc=media0, scale=s0),
        "label_teorica": "Normal teórica",
        "label_ajustada": "Normal ajustada",
        "resultados": resultados
    }


def simular_exponencial(rng, lam, n, alpha, n_rep):
    var_real = 1 / lam**2
    resultados = []

    # Primera muestra
    muestra0 = rng.exponential(scale=1/lam, size=n)
    media0 = np.mean(muestra0)
    S20, LI0, LS0 = ic_var_exponencial(muestra0, lam, alpha)
    lam_hat = 1 / media0

    for _ in range(n_rep):
        muestra = rng.exponential(scale=1/lam, size=n)
        S2, LI, LS = ic_var_exponencial(muestra, lam, alpha)
        resultados.append({
            "S2": S2,
            "LI": LI,
            "LS": LS,
            "mid": (LI + LS) / 2,
            "contiene": LI <= var_real <= LS
        })

    return {
        "nombre": "Exponencial",
        "muestra0": muestra0,
        "media0": media0,
        "S20": S20,
        "LI0": LI0,
        "LS0": LS0,
        "real_var": var_real,
        "x_pdf": np.linspace(0, max(muestra0) + 2, 400),
        "pdf_teorica": lambda x: lam * np.exp(-lam * x),
        "pdf_ajustada": lambda x: lam_hat * np.exp(-lam_hat * x),
        "label_teorica": "Exponencial teórica",
        "label_ajustada": "Exponencial ajustada",
        "resultados": resultados
    }


def simular_student(rng, nu, n, alpha, n_rep, B):
    var_real = nu / (nu - 2)
    resultados = []

    # Primera muestra
    muestra0 = rng.standard_t(df=nu, size=n)
    media0 = np.mean(muestra0)
    S20, LI0, LS0 = ic_var_student(muestra0, nu, alpha, B, rng)
    s0 = np.sqrt(S20)

    for _ in range(n_rep):
        muestra = rng.standard_t(df=nu, size=n)
        S2, LI, LS = ic_var_student(muestra, nu, alpha, B, rng)
        resultados.append({
            "S2": S2,
            "LI": LI,
            "LS": LS,
            "mid": (LI + LS) / 2,
            "contiene": LI <= var_real <= LS
        })

    return {
        "nombre": "t de Student",
        "muestra0": muestra0,
        "media0": media0,
        "S20": S20,
        "LI0": LI0,
        "LS0": LS0,
        "real_var": var_real,
        "x_pdf": np.linspace(min(muestra0) - 1, max(muestra0) + 1, 400),
        "pdf_teorica": lambda x: t.pdf(x, df=nu),
        "pdf_ajustada": lambda x: norm.pdf(x, loc=media0, scale=s0),
        "label_teorica": "t teórica",
        "label_ajustada": "Normal ajustada",
        "resultados": resultados
    }


def imprimir_resumen(res):
    dentro = sum(r["contiene"] for r in res["resultados"])
    fuera = len(res["resultados"]) - dentro

    print(f"\n=== {res['nombre']} ===")
    print(f"Media muestral (primera muestra) = {res['media0']:.4f}")
    print(f"Varianza muestral S_n^2 (primera muestra) = {res['S20']:.4f}")
    print(f"IC 95% (primera muestra) = [{res['LI0']:.4f}, {res['LS0']:.4f}]")
    print(f"Varianza real = {res['real_var']:.4f}")
    print(f"Cobertura en {len(res['resultados'])} repeticiones: {dentro}")
    print(f"No cobertura en {len(res['resultados'])} repeticiones: {fuera}")
    print(f"Proporción de cobertura: {dentro / len(res['resultados']):.4f}")


# =====================================================
# MAIN
# =====================================================

def main():
    # Parámetros específicos
    mu = 5
    sigma = 2

    lam = 0.5

    nu = 5

    # ---------------------------------------------
    # Simulación de las 3 distribuciones
    # ---------------------------------------------
    res_normal = simular_normal(rng, mu, sigma, N, ALPHA, N_REP)
    res_exp = simular_exponencial(rng, lam, N, ALPHA, N_REP)
    res_t = simular_student(rng, nu, N, ALPHA, N_REP, B_BOOT)

    resultados_todos = [res_normal, res_exp, res_t]

    # ---------------------------------------------
    # Resumen numérico
    # ---------------------------------------------
    for res in resultados_todos:
        imprimir_resumen(res)

    # ---------------------------------------------
    # Figura 1: primeras muestras (3 filas x 2 columnas)
    # ---------------------------------------------
    fig1, axes = plt.subplots(3, 2, figsize=(14, 15),
                              gridspec_kw={'width_ratios': [3, 2]})

    for i, res in enumerate(resultados_todos):
        ax_hist = axes[i, 0]
        ax_ic = axes[i, 1]

        # Panel distribución
        ax_hist.hist(
            res["muestra0"], bins=15, density=True,
            alpha=0.6, edgecolor='black', label='Histograma'
        )

        x = res["x_pdf"]
        ax_hist.plot(x, res["pdf_teorica"](x), linewidth=2, label=res["label_teorica"])
        ax_hist.plot(x, res["pdf_ajustada"](x), linestyle='--', linewidth=2,
                     label=res["label_ajustada"])
        ax_hist.axvline(res["media0"], linestyle=':', linewidth=2,
                        label=f"Media = {res['media0']:.2f}")

        ax_hist.set_title(f"{res['nombre']}: distribución de la primera muestra")
        ax_hist.set_xlabel("Valores de X")
        ax_hist.set_ylabel("Densidad")
        ax_hist.grid(alpha=0.3)
        ax_hist.legend()

        # Panel IC de la primera muestra
        ax_ic.hlines(y=1, xmin=res["LI0"], xmax=res["LS0"], linewidth=4,
                     label='IC 95% para la varianza')
        ax_ic.plot([res["LI0"], res["LS0"]], [1, 1], 'o')
        ax_ic.axvline(res["real_var"], linestyle='--', linewidth=2,
                      label='Varianza real')

        ax_ic.set_yticks([])
        ax_ic.set_xlabel("Valor de la varianza")
        ax_ic.set_title(f"{res['nombre']}: IC de la primera muestra")
        ax_ic.grid(True, axis='x', alpha=0.3)
        ax_ic.legend()

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------
    # Figura 2: 100 primeros intervalos en 3 subgráficos
    # ---------------------------------------------
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 14), sharex=False)

    for ax, res in zip(axes2, resultados_todos):
        primeros_100 = res["resultados"][:N_PLOT]
        primeros_100_ordenados = sorted(primeros_100, key=lambda r: r["mid"])

        for j, r in enumerate(primeros_100_ordenados, start=1):
            color = 'tab:blue' if r["contiene"] else 'tab:red'
            ax.hlines(y=j, xmin=r["LI"], xmax=r["LS"], linewidth=2, color=color)
            ax.plot([r["LI"], r["LS"]], [j, j], 'o', color=color, markersize=4)

        ax.axvline(res["real_var"], color='black', linestyle='--', linewidth=2,
                   label=f"Varianza real = {res['real_var']:.3f}")

        ax.set_title(f"{res['nombre']}: primeros 100 intervalos de confianza")
        ax.set_ylabel("Intervalos ordenados")
        ax.grid(axis='x', alpha=0.3)
        ax.legend()

    axes2[-1].set_xlabel("Valor del intervalo para la varianza")
    plt.tight_layout()
    plt.show()


# =====================================================
# Ejecutar
# =====================================================
if __name__ == "__main__":
    main()