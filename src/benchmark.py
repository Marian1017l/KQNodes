import traceback
import time
from src.controllers.manager import Manager
from src.strategies.q_nodes import QNodes

# ---------------------------------------------------------------------------
# Casos de prueba
# Cada entrada: (id, estado_inicial, condiciones, alcance, mecanismo)
# ---------------------------------------------------------------------------
PRUEBAS = [
# Test 1: Tu prueba original (Mecanismo casi completo)
    (1, "1000000000000000000000", "1111111111111111111111", "1111111111111111110011", "1101101101101101101100"),


    # Test 2: Foco en los primeros 10 nodos (Subconjunto localizado)
    # Analiza si los primeros 10 nodos están integrados entre sí.
    (2, "0000000000000000000000", "1111111111000000000000", "1111111111000000000000", "0000000000000000000000"),

    # Test 3: Mecanismo disperso (Interacciones de larga distancia)
    # Analiza nodos alternos para ver si hay dependencias cruzadas.
    (3, "1010101010101010101010", "1111111111111111111111", "1010101010101010101010", "0101010101010101010101"),
]

# ---------------------------------------------------------------------------
# Estrategias a ejecutar
# Cada entrada: (nombre_columna, clase, habilitada)
# ---------------------------------------------------------------------------
ESTRATEGIAS = [
    ("QNodes", QNodes, True),
]


def asegurar_red(estado_inicial: str) -> None:
    gestor = Manager(estado_inicial)
    if not gestor.tpm_filename.exists():
        print(f"Generando red N{len(estado_inicial)}... (esto puede tardar)")
        gestor.generar_red(len(estado_inicial))


def correr_estrategia(clase, tpm, estado_inicial, condiciones, alcance, mecanismo):
    t0 = time.time()
    try:
        instancia = clase(tpm)
        resultado = instancia.aplicar_estrategia(
            estado_inicial, condiciones, alcance, mecanismo
        )
        print(resultado)
    except Exception:
        traceback.print_exc()
    finally:
        print(f"Tiempo total (incluyendo setup): {time.time() - t0:.2f}s")


def ejecutar():
    estrategias_activas = [(n, c) for n, c, hab in ESTRATEGIAS if hab]

    for prueba in PRUEBAS:
        _, estado_inicial, condiciones, alcance, mecanismo = prueba

        asegurar_red(estado_inicial)
        gestor = Manager(estado_inicial)
        tpm    = gestor.cargar_red()

        for nombre, clase in estrategias_activas:
            correr_estrategia(clase, tpm, estado_inicial, condiciones, alcance, mecanismo)

        del tpm


if __name__ == "__main__":
    ejecutar()
