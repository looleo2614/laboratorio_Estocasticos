import numpy as np
import matplotlib.pyplot as plt
from html.parser import HTMLParser
from urllib.parse import urljoin, urldefrag

# ============================================================
# MÓDULO 1: CONSTRUCCIÓN Y SOLUCIÓN DEL MODELO MARKOVIANO
# ============================================================

def construir_matriz_google(diccionario_red, d=0.85):
    """
    Construye la matriz de transición estocástica P y la Matriz de Google M.
    """
    nodos = sorted(list(diccionario_red.keys()))
    m = len(nodos)
    idx = {nodo: i for i, nodo in enumerate(nodos)}

    P = np.zeros((m, m))

    for nodo, enlaces in diccionario_red.items():
        i = idx[nodo]
        if len(enlaces) == 0:
            # Reparación de Dead End: Distribución equiprobable (1/m)
            P[i, :] = 1.0 / m
        else:
            total_enlaces = len(enlaces)
            for destino in enlaces:
                j = idx[destino]
                P[i, j] += 1.0 / total_enlaces

    E = np.ones((m, m))
    M = d * P + (1.0 - d) * (1.0 / m) * E

    return nodos, P, M


def metodo_potencia(M, tol=1e-8, max_iter=500):
    """
    Calcula el vector estacionario pi mediante iteración matricial determinista.
    """
    m = M.shape[0]
    pi = np.ones(m) / m
    historial = [pi.copy()]

    for _ in range(max_iter):
        pi_sig = pi @ M
        historial.append(pi_sig.copy())

        if np.linalg.norm(pi_sig - pi, 1) < tol:
            break
        pi = pi_sig

    return pi, np.array(historial)


def monte_carlo_surfer(M, pasos=150000):
    """
    Simula estocásticamente el recorrido de un navegante aleatorio.
    """
    m = M.shape[0]
    estado = np.random.randint(0, m)
    visitas = np.zeros(m)

    for _ in range(pasos):
        visitas[estado] += 1
        estado = np.random.choice(m, p=M[estado, :])

    return visitas / pasos


# ============================================================
# MÓDULO 2: EXTRACCIÓN DINÁMICA DE HIPERVÍNCULOS HTML
# ============================================================

class ExploradorHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.enlaces = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for atributo, valor in attrs:
                if atributo.lower() == "href" and valor:
                    self.enlaces.append(valor)


def extraer_enlaces_html(html, url_base=None):
    parser = ExploradorHTML()
    parser.feed(html)
    parser.close()

    enlaces = []
    for enlace in parser.enlaces:
        if url_base is not None:
            enlace = urljoin(url_base, enlace)
        enlace, _ = urldefrag(enlace)
        enlaces.append(enlace)
    return enlaces


def construir_red_desde_html(paginas_html):
    red = {}
    for nombre_pagina, html in paginas_html.items():
        enlaces = extraer_enlaces_html(html)
        enlaces_validos = [e for e in enlaces if e in paginas_html]
        red[nombre_pagina] = enlaces_validos
    return red


def demostrar_extraccion_html():
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN: EXTRAER RED DESDE CÓDIGO FUENTE HTML")
    print("=" * 70)
    
    html_simulado = {
        "P1": '<html><body><a href="P2">Link1</a><a href="P3">Link2</a></body></html>',
        "P2": '<html><body><a href="P3">Link a P3</a></body></html>',
        "P3": '<html><body><a href="P1">Regresar a P1</a></body></html>'
    }
    
    red_extraida = construir_red_desde_html(html_simulado)
    print("Red construida dinámicamente desde HTML:")
    for nodo, enlaces in red_extraida.items():
        print(f"  Página {nodo} -> Enlaces detectados: {enlaces}")


# ============================================================
# MÓDULO 3: ANÁLISIS DE ANOMALÍAS Y AUTOVALORES
# ============================================================

def detectar_dead_ends(red):
    return [nodo for nodo, enlaces in red.items() if len(enlaces) == 0]


def detectar_enlaces_duplicados(red):
    duplicados = {}
    for nodo, enlaces in red.items():
        conteo = {}
        for destino in enlaces:
            conteo[destino] = conteo.get(destino, 0) + 1
        repetidos = {dest: cant for dest, cant in conteo.items() if cant > 1}
        if repetidos:
            duplicados[nodo] = repetidos
    return duplicados


def detectar_spider_traps(red):
    """Detecta Spider Traps usando Tarjan SCC (evita falsos positivos)."""
    nodos = list(red.keys())
    m = len(nodos)
    
    index = 0
    stack = []
    indices = {}
    lowlink = {}
    on_stack = set()
    sccs = []

    def strongconnect(nodo):
        nonlocal index
        indices[nodo] = index
        lowlink[nodo] = index
        index += 1
        stack.append(nodo)
        on_stack.add(nodo)

        for vecino in red.get(nodo, []):
            if vecino in red:
                if vecino not in indices:
                    strongconnect(vecino)
                    lowlink[nodo] = min(lowlink[nodo], lowlink[vecino])
                elif vecino in on_stack:
                    lowlink[nodo] = min(lowlink[nodo], indices[vecino])

        if lowlink[nodo] == indices[nodo]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == nodo:
                    break
            sccs.append(scc)

    for nodo in nodos:
        if nodo not in indices:
            strongconnect(nodo)

    traps = []
    for scc in sccs:
        set_scc = set(scc)
        if len(set_scc) >= m:
            continue
            
        es_trap = True
        for u in set_scc:
            enlaces = red.get(u, [])
            if len(enlaces) == 0 or any(v not in set_scc for v in enlaces):
                es_trap = False
                break
                
        if es_trap:
            traps.append(sorted(scc))
            
    return traps


def calcular_segundo_autovalor(M):
    """Calcula |lambda_2| para la tasa de convergencia."""
    autovalores = np.linalg.eigvals(M)
    modulos = np.sort(np.abs(autovalores))[::-1]
    return modulos[0], modulos[1] if len(modulos) > 1 else 0.0


def resumen_anomalias(red):
    print("\n" + "=" * 60)
    print("ANÁLISIS DE ANOMALÍAS")
    print("=" * 60)
    dead_ends = detectar_dead_ends(red)
    duplicados = detectar_enlaces_duplicados(red)
    traps = detectar_spider_traps(red)

    print(f"Dead Ends: {dead_ends if dead_ends else 'Ninguno'}")
    print(f"Enlaces Duplicados: {duplicados if duplicados else 'Ninguno'}")
    print(f"Spider Traps: {traps if traps else 'Ninguna'}")


# ============================================================
# MÓDULO 4: FUNCIONES DE DIBUJO DE SUBPLOTS (DASHBOARD)
# ============================================================

def dibujar_convergencia(ax, nodos, historial, max_iter=30):
    """Dibuja en el eje asignado la curva pi(n) vs Iteraciones."""
    cantidad = min(max_iter, len(historial))
    for i, nodo in enumerate(nodos):
        ax.plot(range(cantidad), historial[:cantidad, i], marker="o", markersize=3, label=nodo)
    ax.set_xlabel("Iteraciones (n)")
    ax.set_ylabel("Probabilidad pi(n)")
    ax.set_title("1. Convergencia del Vector (Topología 3)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small', ncol=2)


def dibujar_comparacion(ax, nodos, pi_potencia, pi_monte_carlo):
    """Dibuja en el eje asignado la comparativa Potencia vs Monte Carlo."""
    x = np.arange(len(nodos))
    ancho = 0.35
    ax.bar(x - ancho / 2, pi_potencia, ancho, label="Potencia", color="navy")
    ax.bar(x + ancho / 2, pi_monte_carlo, ancho, label="Monte Carlo", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(nodos, rotation=45 if len(nodos) > 5 else 0)
    ax.set_xlabel("Páginas Web")
    ax.set_ylabel("PageRank")
    ax.set_title("2. Monte Carlo vs. Método de la Potencia")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def dibujar_sensibilidad(ax, valores_d, iteraciones):
    """Dibuja en el eje asignado la sensibilidad de convergencia vs d."""
    ax.plot(valores_d, iteraciones, marker="o", color="crimson")
    ax.set_xlabel("Factor de Amortiguación d")
    ax.set_ylabel("Iteraciones")
    ax.set_title("3. Sensibilidad de Convergencia vs. Factor d")
    ax.grid(True, alpha=0.3)


def dibujar_ranking(ax, nodos, pi, titulo="4. Ranking Final PageRank"):
    """Dibuja en el eje asignado la distribución del PageRank final."""
    ranking = sorted(zip(nodos, pi), key=lambda x: x[1], reverse=True)
    nombres = [item[0] for item in ranking]
    valores = [item[1] for item in ranking]

    ax.bar(nombres, valores, color="teal", edgecolor="black")
    ax.set_xlabel("Páginas Web")
    ax.set_ylabel("PageRank Final")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)


def mostrar_dashboard_integrado(res_t3, val_d, iters_d):
    """
    Crea una única ventana con una cuadrícula de 2x2 para desplegar
    las 4 gráficas obligatorias simultáneamente.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("LABORATORIO PAGERANK - DASHBOARD INTEGRADO DE RESULTADOS", fontsize=15, fontweight="bold")

    # Subplot 1: Convergencia (Topología 3)
    dibujar_convergencia(axes[0, 0], res_t3["nodos"], res_t3["historial"])

    # Subplot 2: Comparación Potencia vs Monte Carlo (Topología 3)
    dibujar_comparacion(axes[0, 1], res_t3["nodos"], res_t3["pi"], res_t3["monte_carlo"])

    # Subplot 3: Sensibilidad a d
    dibujar_sensibilidad(axes[1, 0], val_d, iters_d)

    # Subplot 4: Ranking del Ataque SEO
    dibujar_ranking(axes[1, 1], res_t3["nodos"], res_t3["pi"], "4. Ranking PageRank (Ataque SEO - Topología 3)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()  # Se llama una sola vez para abrir una única ventana con todo


# ============================================================
# MÓDULO 5: TOPOLOGÍAS Y EJECUCIÓN
# ============================================================

def crear_topologia_1():
    return {"P1": ["P2", "P3"], "P2": ["P3"], "P3": ["P1"]}

def crear_topologia_2():
    return {"A": ["B", "C"], "B": ["C"], "C": [], "D": ["E"], "E": ["D"]}

def crear_topologia_3():
    return {
        "T": ["M"],
        "M": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
        "F1": ["T"], "F2": ["T"], "F3": ["T"], "F4": ["T"],
        "F5": ["T"], "F6": ["T"], "F7": ["T"],
        "F8": ["T", "T", "T"]
    }


def analizar_sensibilidad_d(red, valores_d=None, tol=1e-8, max_iter=5000):
    if valores_d is None:
        valores_d = np.linspace(0.10, 0.99, 19)

    iteraciones = []
    for d in valores_d:
        _, _, M = construir_matriz_google(red, d=d)
        _, historial = metodo_potencia(M, tol=tol, max_iter=max_iter)
        iteraciones.append(len(historial) - 1)

    return np.array(valores_d), np.array(iteraciones)


def ejecutar_topologia(red, nombre_topologia, pasos_mc=150000, d=0.85):
    print("\n" + "=" * 70)
    print(f"EJECUTANDO: {nombre_topologia}")
    print("=" * 70)

    resumen_anomalias(red)
    nodos, P, M = construir_matriz_google(red, d=d)
    
    l1, l2 = calcular_segundo_autovalor(M)
    print(f"\nAutovalores: |lambda_1| = {l1:.4f} | |lambda_2| = {l2:.4f} (Coincide con d={d:.2f})")

    pi_potencia, historial = metodo_potencia(M)
    print(f"Método de la Potencia convergido en {len(historial) - 1} iteraciones.")

    pi_mc = monte_carlo_surfer(M, pasos=pasos_mc)
    error_mc = np.linalg.norm(pi_potencia - pi_mc, 1)
    print(f"Monte Carlo ({pasos_mc} pasos) completado. Error L1: {error_mc:.6f}")

    return {
        "nodos": nodos, "M": M, "pi": pi_potencia,
        "historial": historial, "monte_carlo": pi_mc
    }


def ejecutar_laboratorio():
    np.random.seed(42)

    # 1. Extracción HTML
    demostrar_extraccion_html()

    # 2. Análisis numérico de Topologías 1, 2 y 3
    _ = ejecutar_topologia(crear_topologia_1(), "TOPOLOGÍA 1: VALIDACIÓN BÁSICA (3 NODOS)")
    _ = ejecutar_topologia(crear_topologia_2(), "TOPOLOGÍA 2: DEAD ENDS Y SPIDER TRAPS (5 NODOS)")
    res3 = ejecutar_topologia(crear_topologia_3(), "TOPOLOGÍA 3: ATAQUE SEO Y PÁGINAS FALSAS (10 NODOS)")

    # 3. Análisis de sensibilidad al factor d
    val_d, iters_d = analizar_sensibilidad_d(crear_topologia_1())

    # 4. Mostrar TODAS las gráficas simultáneamente en una sola ventana
    print("\n" + "=" * 70)
    print("DESPLEGANDO DASHBOARD DE GRÁFICAS OBLIGATORIAS (VENTANA ÚNICA)")
    print("=" * 70)
    mostrar_dashboard_integrado(res3, val_d, iters_d)

    print("\n" + "=" * 70)
    print("LABORATORIO FINALIZADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_laboratorio()