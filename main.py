import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CÓDIGO BASE DEL LABORATORIO
# ============================================================

def construir_matriz_google(diccionario_red, d=0.85):
    nodos = sorted(list(diccionario_red.keys()))
    m = len(nodos)
    idx = {nodo: i for i, nodo in enumerate(nodos)}

    P = np.zeros((m, m))

    for nodo, enlaces in diccionario_red.items():
        i = idx[nodo]

        if len(enlaces) == 0:
            P[i, :] = 1.0 / m  # Reparacion de Dead End
        else:
            total_enlaces = len(enlaces)  # Incluye duplicados

            for destino in enlaces:
                j = idx[destino]
                P[i, j] += 1.0 / total_enlaces

    E = np.ones((m, m))

    M = d * P + (1.0 - d) * (1.0 / m) * E

    return nodos, P, M


def metodo_potencia(M, tol=1e-8, max_iter=500):
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
    m = M.shape[0]

    estado = np.random.randint(0, m)
    visitas = np.zeros(m)

    for _ in range(pasos):

        visitas[estado] += 1

        estado = np.random.choice(
            m,
            p=M[estado, :]
        )

    return visitas / pasos


# ============================================================
# EXTENSIÓN 1
# EXPLORACIÓN Y EXTRACCIÓN DE ENLACES HTML
# ============================================================

from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.parse import urljoin, urldefrag


class ExploradorHTML(HTMLParser):

    def __init__(self):
        super().__init__()
        self.enlaces = []

    def handle_starttag(self, tag, attrs):

        if tag.lower() != "a":
            return

        for atributo, valor in attrs:

            if atributo.lower() == "href" and valor:
                self.enlaces.append(valor)


def extraer_enlaces_html(html, url_base=None):
    """
    Extrae todos los enlaces <a href="..."> encontrados
    en una estructura HTML.

    Si se proporciona url_base, los enlaces relativos se
    convierten en URLs absolutas.
    """

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


def construir_red_desde_html(paginas):
    """
    Construye dinámicamente el diccionario de red a partir
    de páginas HTML.

    pagínas:
        {
            "A": "<html>...</html>",
            "B": "<html>...</html>"
        }

    Los enlaces duplicados NO se eliminan.
    """

    red = {}

    for nombre_pagina, html in paginas.items():

        enlaces = extraer_enlaces_html(html)

        # Solo conservamos enlaces que pertenecen
        # a las páginas que forman parte de la red.
        enlaces_validos = [
            enlace for enlace in enlaces
            if enlace in paginas
        ]

        red[nombre_pagina] = enlaces_validos

    return red


def obtener_html_url(url, timeout=10):
    """
    Descarga el HTML de una URL.
    """

    request = Request(
        url,
        headers={
            "User-Agent": "PageRank-Laboratorio/1.0"
        }
    )

    with urlopen(request, timeout=timeout) as respuesta:

        contenido = respuesta.read()

        charset = respuesta.headers.get_content_charset() or "utf-8"

        return contenido.decode(charset, errors="replace")


def explorar_urls(urls):
    """
    Descarga varias URLs y construye la red a partir
    de los hipervínculos encontrados.

    urls:
        {
            "A": "https://ejemplo.com/a",
            "B": "https://ejemplo.com/b"
        }
    """

    html_paginas = {}

    for nombre, url in urls.items():

        try:

            html_paginas[nombre] = obtener_html_url(url)

            print(f"[OK] Página descargada: {nombre}")

        except Exception as error:

            print(f"[ERROR] No se pudo descargar {nombre}: {error}")

            html_paginas[nombre] = ""

    red = {}

    for nombre, html in html_paginas.items():

        enlaces = extraer_enlaces_html(
            html,
            urls[nombre]
        )

        red[nombre] = enlaces

    return red


# ============================================================
# EXTENSIÓN 2
# ANÁLISIS DE ENLACES Y ANOMALÍAS
# ============================================================

def contar_enlaces(red):
    """
    Cuenta los enlaces salientes de cada página.
    Los duplicados cuentan individualmente.
    """

    conteo = {}

    for nodo, enlaces in red.items():
        conteo[nodo] = len(enlaces)

    return conteo


def detectar_dead_ends(red):
    """
    Identifica páginas sin enlaces salientes.
    """

    return [
        nodo
        for nodo, enlaces in red.items()
        if len(enlaces) == 0
    ]


def detectar_enlaces_duplicados(red):
    """
    Detecta enlaces duplicados dentro de cada página.
    """

    duplicados = {}

    for nodo, enlaces in red.items():

        encontrados = {}

        for destino in enlaces:
            encontrados[destino] = encontrados.get(destino, 0) + 1

        repetidos = {
            destino: cantidad
            for destino, cantidad in encontrados.items()
            if cantidad > 1
        }

        if repetidos:
            duplicados[nodo] = repetidos

    return duplicados


def detectar_spider_traps(red):
    """
    Detecta componentes cerrados.

    Un conjunto de páginas es considerado una posible
    Spider Trap si ninguna de ellas apunta fuera del conjunto.

    Esta implementación permite detectar componentes cerrados
    de cualquier tamaño.
    """

    nodos = set(red.keys())
    visitados_global = set()
    trampas = []

    def dfs(inicio):

        pendientes = [inicio]
        componente = set()

        while pendientes:

            actual = pendientes.pop()

            if actual in componente:
                continue

            componente.add(actual)

            for destino in red.get(actual, []):

                if destino in nodos:
                    pendientes.append(destino)

        return componente

    for nodo in nodos:

        if nodo in visitados_global:
            continue

        componente = dfs(nodo)

        visitados_global.update(componente)

        cerrado = True

        for pagina in componente:

            for destino in red.get(pagina, []):

                if destino not in componente:
                    cerrado = False
                    break

            if not cerrado:
                break

        if cerrado and len(componente) < len(nodos):

            trampas.append(sorted(componente))

    return trampas


def resumen_anomalias(red):

    print("\n" + "=" * 60)
    print("ANÁLISIS DE ANOMALÍAS")
    print("=" * 60)

    dead_ends = detectar_dead_ends(red)

    duplicados = detectar_enlaces_duplicados(red)

    traps = detectar_spider_traps(red)

    print("\nDead Ends:")

    if dead_ends:
        print(dead_ends)
    else:
        print("Ninguno")

    print("\nEnlaces duplicados:")

    if duplicados:
        for pagina, enlaces in duplicados.items():
            print(f"{pagina}: {enlaces}")
    else:
        print("Ninguno")

    print("\nSpider Traps:")

    if traps:
        for trap in traps:
            print(trap)
    else:
        print("Ninguna")

    return dead_ends, duplicados, traps


# ============================================================
# EXTENSIÓN 3
# VALIDACIONES MATEMÁTICAS
# ============================================================

def validar_matriz_estocastica(P, nombre="P"):

    suma_filas = np.sum(P, axis=1)

    es_no_negativa = np.all(P >= -1e-12)

    es_estocastica = np.allclose(
        suma_filas,
        1.0,
        atol=1e-10
    )

    print(f"\nValidación de matriz {nombre}")

    print(f"Valores no negativos: {es_no_negativa}")

    print(f"Filas suman 1: {es_estocastica}")

    print("Suma de filas:")

    print(suma_filas)

    return es_no_negativa and es_estocastica


def validar_distribucion(pi):

    suma = np.sum(pi)

    no_negativa = np.all(pi >= -1e-12)

    print("\nValidación del vector estacionario")

    print(f"Suma de probabilidades = {suma:.12f}")

    print(f"Probabilidades no negativas = {no_negativa}")

    return np.isclose(suma, 1.0) and no_negativa


def error_estacionario(pi, M):

    return np.linalg.norm(
        pi @ M - pi,
        1
    )


def mostrar_ranking(nodos, pi):

    ranking = sorted(
        zip(nodos, pi),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRanking PageRank:")

    print("-" * 35)

    for posicion, (nodo, valor) in enumerate(ranking, 1):

        print(
            f"{posicion:2d}. {nodo:5s} "
            f"{valor:.8f}"
        )

    return ranking


# ============================================================
# EXTENSIÓN 4
# GRÁFICA 1:
# CONVERGENCIA DEL MÉTODO DE LA POTENCIA
# ============================================================

def graficar_convergencia(nodos, historial, max_iteraciones=30):

    cantidad = min(
        max_iteraciones,
        len(historial)
    )

    plt.figure(figsize=(10, 6))

    for i, nodo in enumerate(nodos):

        plt.plot(
            range(cantidad),
            historial[:cantidad, i],
            marker="o",
            markersize=3,
            label=nodo
        )

    plt.xlabel("Iteraciones")

    plt.ylabel("Probabilidad")

    plt.title(
        "Convergencia del vector PageRank"
    )

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# EXTENSIÓN 5
# GRÁFICA 2:
# MONTE CARLO VS MÉTODO DE LA POTENCIA
# ============================================================

def graficar_comparacion(
    nodos,
    pi_potencia,
    pi_monte_carlo
):

    x = np.arange(len(nodos))

    ancho = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - ancho / 2,
        pi_potencia,
        ancho,
        label="Método de la Potencia"
    )

    plt.bar(
        x + ancho / 2,
        pi_monte_carlo,
        ancho,
        label="Monte Carlo"
    )

    plt.xticks(
        x,
        nodos
    )

    plt.xlabel("Página")

    plt.ylabel("PageRank")

    plt.title(
        "Comparación: Método de la Potencia vs Monte Carlo"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# EXTENSIÓN 6
# GRÁFICA 3:
# SENSIBILIDAD AL FACTOR DE AMORTIGUACIÓN
# ============================================================

def analizar_sensibilidad_d(
    red,
    valores_d=None,
    tol=1e-8,
    max_iter=5000
):

    if valores_d is None:

        valores_d = np.linspace(
            0.10,
            0.99,
            19
        )

    iteraciones = []

    for d in valores_d:

        nodos, P, M = construir_matriz_google(
            red,
            d=d
        )

        pi, historial = metodo_potencia(
            M,
            tol=tol,
            max_iter=max_iter
        )

        if (
            len(historial) >= max_iter + 1
            and np.linalg.norm(
                historial[-1] - historial[-2],
                1
            ) >= tol
        ):

            numero_iteraciones = max_iter

        else:

            numero_iteraciones = len(historial) - 1

        iteraciones.append(
            numero_iteraciones
        )

    return np.array(valores_d), np.array(iteraciones)


def graficar_sensibilidad(
    valores_d,
    iteraciones
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        valores_d,
        iteraciones,
        marker="o"
    )

    plt.xlabel(
        "Factor de amortiguación d"
    )

    plt.ylabel(
        "Iteraciones para converger"
    )

    plt.title(
        "Sensibilidad de la convergencia al factor d"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# EXTENSIÓN 7
# GRÁFICA 4:
# RANKING DE LA TOPOLOGÍA 3
# ============================================================

def graficar_ranking(
    nodos,
    pi,
    titulo="Distribución del PageRank"
):

    ranking = sorted(
        zip(nodos, pi),
        key=lambda x: x[1],
        reverse=True
    )

    nombres = [
        item[0]
        for item in ranking
    ]

    valores = [
        item[1]
        for item in ranking
    ]

    plt.figure(figsize=(11, 6))

    plt.bar(
        nombres,
        valores
    )

    plt.xlabel("Página")

    plt.ylabel("PageRank")

    plt.title(titulo)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# EXTENSIÓN 8
# TOPOLOGÍAS OBLIGATORIAS
# ============================================================

def crear_topologia_1():

    return {

        "P1": ["P2", "P3"],

        "P2": ["P3"],

        "P3": ["P1"]
    }


def crear_topologia_2():

    return {

        "A": ["B", "C"],

        "B": ["C"],

        "C": [],

        "D": ["E"],

        "E": ["D"]
    }


def crear_topologia_3():

    return {

        "T": ["M"],

        "M": [
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
            "F6",
            "F7",
            "F8"
        ],

        "F1": ["T"],

        "F2": ["T"],

        "F3": ["T"],

        "F4": ["T"],

        "F5": ["T"],

        "F6": ["T"],

        "F7": ["T"],

        # Tres enlaces duplicados hacia T
        "F8": ["T", "T", "T"]
    }


# ============================================================
# EXTENSIÓN 9
# EJECUCIÓN COMPLETA DE UNA TOPOLOGÍA
# ============================================================

def ejecutar_topologia(
    red,
    nombre_topologia,
    pasos_monte_carlo=150000,
    d=0.85
):

    print("\n")
    print("=" * 70)
    print(nombre_topologia)
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Mostrar red
    # --------------------------------------------------------

    print("\nRed:")

    for nodo, enlaces in red.items():

        print(
            f"{nodo} -> {enlaces}"
        )

    # --------------------------------------------------------
    # 2. Analizar anomalías
    # --------------------------------------------------------

    resumen_anomalias(red)

    # --------------------------------------------------------
    # 3. Construir matrices
    # --------------------------------------------------------

    nodos, P, M = construir_matriz_google(
        red,
        d=d
    )

    print("\nNodos:")

    print(nodos)

    print("\nMatriz P:")

    print(
        np.round(P, 4)
    )

    print("\nMatriz M:")

    print(
        np.round(M, 4)
    )

    # --------------------------------------------------------
    # 4. Validar P
    # --------------------------------------------------------

    validar_matriz_estocastica(
        P,
        "P"
    )

    # --------------------------------------------------------
    # 5. Validar M
    # --------------------------------------------------------

    validar_matriz_estocastica(
        M,
        "M"
    )

    # --------------------------------------------------------
    # 6. Método de la Potencia
    # --------------------------------------------------------

    pi_potencia, historial = metodo_potencia(
        M
    )

    print("\nMétodo de la Potencia:")

    print(
        np.round(
            pi_potencia,
            8
        )
    )

    print(
        f"Iteraciones: {len(historial) - 1}"
    )

    # --------------------------------------------------------
    # 7. Validación del vector
    # --------------------------------------------------------

    validar_distribucion(
        pi_potencia
    )

    print(
        "Error estacionario:",
        f"{error_estacionario(pi_potencia, M):.12e}"
    )

    # --------------------------------------------------------
    # 8. Ranking
    # --------------------------------------------------------

    ranking = mostrar_ranking(
        nodos,
        pi_potencia
    )

    # --------------------------------------------------------
    # 9. Monte Carlo
    # --------------------------------------------------------

    pi_monte_carlo = monte_carlo_surfer(
        M,
        pasos=pasos_monte_carlo
    )

    print("\nMonte Carlo:")

    print(
        np.round(
            pi_monte_carlo,
            8
        )
    )

    # --------------------------------------------------------
    # 10. Error entre ambos métodos
    # --------------------------------------------------------

    error_mc = np.linalg.norm(
        pi_potencia - pi_monte_carlo,
        1
    )

    print(
        "\nError L1 Monte Carlo vs Potencia:",
        f"{error_mc:.8f}"
    )

    # --------------------------------------------------------
    # 11. Gráficas
    # --------------------------------------------------------

    graficar_convergencia(
        nodos,
        historial
    )

    graficar_comparacion(
        nodos,
        pi_potencia,
        pi_monte_carlo
    )

    return {
        "nodos": nodos,
        "P": P,
        "M": M,
        "pi": pi_potencia,
        "historial": historial,
        "monte_carlo": pi_monte_carlo,
        "ranking": ranking
    }


# ============================================================
# EXTENSIÓN 10
# ANÁLISIS ESPECÍFICO DE TOPOLOGÍA 3
# ============================================================

def analizar_ataque_seo():

    red = crear_topologia_3()

    resultado = ejecutar_topologia(
        red,
        "TOPOLOGÍA 3 - ATAQUE SEO",
        pasos_monte_carlo=150000
    )

    graficar_ranking(
        resultado["nodos"],
        resultado["pi"],
        "Distribución del PageRank - Topología 3"
    )

    return resultado


# ============================================================
# EXTENSIÓN 11
# PROGRAMA PRINCIPAL DEL LABORATORIO
# ============================================================

def ejecutar_laboratorio():

    np.random.seed(42)

    # --------------------------------------------------------
    # TOPOLOGÍA 1
    # --------------------------------------------------------

    red1 = crear_topologia_1()

    resultado1 = ejecutar_topologia(
        red1,
        "TOPOLOGÍA 1 - VALIDACIÓN BÁSICA"
    )

    # --------------------------------------------------------
    # TOPOLOGÍA 2
    # --------------------------------------------------------

    red2 = crear_topologia_2()

    resultado2 = ejecutar_topologia(
        red2,
        "TOPOLOGÍA 2 - DEAD END Y SPIDER TRAP"
    )

    # --------------------------------------------------------
    # TOPOLOGÍA 3
    # --------------------------------------------------------

    red3 = crear_topologia_3()

    resultado3 = ejecutar_topologia(
        red3,
        "TOPOLOGÍA 3 - ATAQUE SEO"
    )

    # --------------------------------------------------------
    # SENSIBILIDAD DE d
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ANÁLISIS DE SENSIBILIDAD AL FACTOR d")
    print("=" * 70)

    valores_d, iteraciones = analizar_sensibilidad_d(
        red1
    )

    for d, it in zip(
        valores_d,
        iteraciones
    ):

        print(
            f"d = {d:.2f} -> "
            f"{it} iteraciones"
        )

    graficar_sensibilidad(
        valores_d,
        iteraciones
    )

    # --------------------------------------------------------
    # RANKING FINAL TOPOLOGÍA 3
    # --------------------------------------------------------

    graficar_ranking(
        resultado3["nodos"],
        resultado3["pi"],
        "Ranking PageRank - Ataque SEO"
    )

    print("\n")
    print("=" * 70)
    print("LABORATORIO FINALIZADO")
    print("=" * 70)


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    ejecutar_laboratorio()