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
    # Configuración de estilo para paper académico
    # ---------------------------------------------
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })

    # ---------------------------------------------
    # Figura 1: Distribuciones y primeros IC (layout compacto)
    # ---------------------------------------------
    fig1 = plt.figure(figsize=(7.5, 4.2))
    gs = fig1.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                           height_ratios=[2, 0.9])

    # Fila superior: histogramas con distribuciones
    for i, res in enumerate(resultados_todos):
        ax = fig1.add_subplot(gs[0, i])
        
        # Histograma
        ax.hist(res["muestra0"], bins=15, density=True,
                alpha=0.5, edgecolor='black', linewidth=0.5, 
                color='lightgray')

        x = res["x_pdf"]
        ax.plot(x, res["pdf_teorica"](x), 'k-', linewidth=1.5)
        ax.plot(x, res["pdf_ajustada"](x), 'k--', linewidth=1.5)

        ax.set_title(res['nombre'], fontweight='bold', pad=6)
        ax.set_xlabel('$x$')
        if i == 0:
            ax.set_ylabel('Densidad')
        ax.grid(alpha=0.25, linewidth=0.5)
        
    # Fila inferior: intervalos de confianza iniciales
    for i, res in enumerate(resultados_todos):
        ax = fig1.add_subplot(gs[1, i])
        
        # IC con error bars más elegante
        mid_point = (res["LI0"] + res["LS0"]) / 2
        error = res["LS0"] - mid_point
        
        ax.errorbar([mid_point], [0.5], xerr=[[error]], 
                   fmt='o', color='black', elinewidth=2, 
                   capsize=5, capthick=2, markersize=4)
        
        ax.axvline(res["real_var"], color='red', linestyle='--', 
                  linewidth=1.5)
        
        # Añadir etiqueta de varianza real solo en el centro
       
        
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Varianza')
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        

    # Leyenda común para toda la figura 1
    from matplotlib.lines import Line2D
    legend_elements_fig1 = [
        Line2D([0], [0], color='lightgray', linewidth=6, label='Muestra'),
        Line2D([0], [0], color='k', linestyle='-', linewidth=1.5, label='Teórica'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label='Ajustada'),
    ]
    fig1.legend(handles=legend_elements_fig1, loc='upper center', 
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8)

    plt.savefig('INFORME/Imagenes/figura1_distribuciones.pdf', 
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig('INFORME/Imagenes/figura1_distribuciones.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    plt.close(fig1)  # Cerrar figura 1 para liberar memoria

    # ---------------------------------------------
    # Figura 2: Intervalos de confianza (layout horizontal compacto)
    # ---------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(7.5, 2.3), sharey=True)

    for ax, res in zip(axes2, resultados_todos):
        primeros = res["resultados"][:N_PLOT]
        primeros_ordenados = sorted(primeros, key=lambda r: r["mid"])

        # Contar cobertura
        n_cubren = sum(1 for r in primeros if r["contiene"])
        prop_cobertura = n_cubren / len(primeros)

        for j, r in enumerate(primeros_ordenados, start=1):
            color = '#2E86AB' if r["contiene"] else '#A23B72'
            alpha = 0.6 if r["contiene"] else 0.8
            ax.hlines(y=j, xmin=r["LI"], xmax=r["LS"], 
                     linewidth=0.8, color=color, alpha=alpha)

        ax.axvline(res["real_var"], color='black', linestyle='--', 
                  linewidth=1.5, zorder=10)

        ax.set_title(res['nombre'], fontweight='bold', pad=6)
        ax.set_xlabel('Varianza')
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        ax.set_ylim(0, N_PLOT + 1)
        
        # Texto con estadísticas - más compacto
        ax.text(0.98, 0.98, 
               f'{prop_cobertura:.1%}',
               transform=ax.transAxes, fontsize=8, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', edgecolor='gray', 
                        linewidth=0.5, alpha=0.85))

    axes2[0].set_ylabel('Índice de intervalo')
    
    # Leyenda común más compacta
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2E86AB', linewidth=2, label='Cubre $\\sigma^2$'),
        Line2D([0], [0], color='#A23B72', linewidth=2, label='No cubre'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, 
               label='$\\sigma^2$ real')
    ]
    fig2.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig('INFORME/Imagenes/figura2_intervalos.pdf', 
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig('INFORME/Imagenes/figura2_intervalos.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2)  # Cerrar figura 2 para liberar memoria


# =====================================================
# Ejecutar
# =====================================================
if __name__ == "__main__":
    main()