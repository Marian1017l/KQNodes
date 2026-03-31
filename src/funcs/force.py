from itertools import product, chain, combinations, islice
from typing import Generator, Tuple, Union
import numpy as np


def generar_candidatos(n_vars: int):
    """
    Genera todas las combinaciones posibles para condicionamiento.
    Empieza desde conjuntos pequeños hasta el sistema completo.

    Args:
        n_vars: Número total de variables en el sistema

    Returns:
        Generador de conjuntos de variables a condicionar
    """
    return (combo for r in range(n_vars) for combo in combinations(range(n_vars), r))


def generar_subsistemas(vars: tuple[int]):
    """
    Genera las combinaciones posibles para un sistema candidato de N variables.
    Son dos conjuntos de combinaciones que hacen un producto cartesiano.
    Las combinaciones van desde vacío hasta la N-1 combinación puesto generar la n-ésima combinación complicaría la posterior marginalización, esta recibe las dimensiones a descartar, pero, si no tiene dimensiones para marginalizar (conjunto vacío) no se realiza nada, por lo que retornar el n-ésimo elemento con la diferencia de dimensiones activas del sistema candidato haría se envíe una tupla vacua, es por ello iteramos cuantas variables tengamos y no N+1.

    Args:
        n_vars (int): Tamaño del sistema candidato, sus variables.

    Returns:
        Generador con combinaciones de subsistemas.
    """
    tiempos = [combo for r in range(len(vars) + 1) for combo in combinations(vars, r)]
    return product(tiempos, tiempos)


def generar_particiones(
    m: int,
    n: int,
    *,
    as_matrix: bool = False,
    as_generator: bool = True,
) -> Union[Generator[Tuple[np.ndarray, np.ndarray], None, None], np.ndarray]:
    """
    Versión para generar particiones binarias.
    Eficiente para valores grandes de M y N.

    Args:
        m: Tamaño de la primera parte
        n: Tamaño de la segunda parte
        square: Si True, retorna matriz 2D. Si False, retorna tuplas
        as_generator: Si True, usa generador para memoria eficiente
    """
    # Usar desplazamiento bit a bit para potencias de 2
    if m < 1:
        # print("Error: m=0")
        # return
        raise ValueError(f"Alcance trivial: Future no debe tener {m} elementos")

    m_combinations = 1 << (m - 1)  # 2^(M-1)
    n_combinations = 1 << n  # 2^N

    # Usar empty es más rápido que zeros porque no inicializa memoria
    m_bits = np.empty((m_combinations, m), dtype=np.uint8)
    n_bits = np.empty((n_combinations, n), dtype=np.uint8)

    # Vectorizar la generación de bits usando operaciones de NumPy
    m_indices = np.arange(m_combinations, dtype=np.uint32)[:, np.newaxis]
    n_indices = np.arange(n_combinations, dtype=np.uint32)[:, np.newaxis]

    # Crear máscaras de bits vectorizadas
    m_shifts = np.arange(m - 1, -1, -1, dtype=np.uint8)
    n_shifts = np.arange(n - 1, -1, -1, dtype=np.uint8)

    # Generar todos los bits de una vez usando broadcasting
    m_bits = (m_indices >> m_shifts) & 1
    n_bits = (n_indices >> n_shifts) & 1

    if as_generator:

        def partition_generator():
            # Para i = 0, empezamos desde j = 1 para evitar (0,0)
            m_row = m_bits[0]
            for j in range(1, n_combinations):
                yield m_row, n_bits[j]

            # Para el resto de i, usamos todo j
            for i in range(1, m_combinations):
                m_row = m_bits[i]
                for j in range(n_combinations):
                    yield m_row, n_bits[j]

        return partition_generator()

    if as_matrix:
        # Optimizar creación de matriz resultado
        total_rows = m_combinations * n_combinations
        result = np.empty((total_rows, m + n), dtype=np.uint8)

        # Usar vectorización para llenar la matriz
        result_view_m = result[:, :m].reshape(m_combinations, n_combinations, m)
        result_view_n = result[:, m:].reshape(m_combinations, n_combinations, n)

        # Asignar valores usando broadcasting
        result_view_m[:] = m_bits[:, np.newaxis, :]
        result_view_n[:] = n_bits

        return result if not as_generator else (row for row in result)

    # Si no es generador, crear lista de tuplas
    # Usar views en lugar de copias cuando sea posible
    return [
        (m_bits[i], n_bits[j])
        for i in range(m_combinations)
        for j in range(n_combinations)
    ]


def biparticiones(
    alcances: np.ndarray,
    mecanismos: np.ndarray,
    total=None,
):
    if total is None:
        total = (1 << alcances.size) * (1 << mecanismos.size)
    return islice(
        product(subconjuntos(alcances), subconjuntos(mecanismos)), 1, total - 1
    )


def subconjuntos(arr: np.ndarray):
    return chain.from_iterable(combinations(arr, r) for r in range(len(arr) + 1))


def generar_k_particiones(num_elementos: int, k: int):
    """
    Genera todos los pares (asig_futuro, asig_presente) para una k-partición.

    Cada asignación es una tupla de longitud `num_elementos` con valores en
    {0, ..., k-1}. El futuro (t+1) y el presente (t) se asignan de forma
    independiente, igual que en bipartir donde alcance y mecanismo son
    parámetros separados.

    Se excluyen los pares triviales: aquellos en los que tanto asig_futuro
    como asig_presente asignan todos los elementos al mismo grupo (no hay corte).

    Args:
        num_elementos: número de nodos del sistema
        k: número de grupos

    Yields:
        (asig_futuro, asig_presente): par de tuplas de longitud num_elementos
    """
    if k < 2:
        raise ValueError(f"k debe ser al menos 2, se recibió k={k}")
    if num_elementos < 1:
        raise ValueError(f"num_elementos debe ser al menos 1, se recibió {num_elementos}")

    uniformes = {tuple([g] * num_elementos) for g in range(k)}

    for asig_futuro in product(range(k), repeat=num_elementos):
        for asig_presente in product(range(k), repeat=num_elementos):
            if asig_futuro in uniformes and asig_presente in uniformes:
                continue
            yield asig_futuro, asig_presente
