import traceback

from src.controllers.manager import Manager
from src.strategies.force import BruteForce
from src.strategies.q_nodes import QNodes
from src.strategies.geometric import GeometricSIA
from src.strategies.phi import Phi

# ---------------------------------------------------------------------------
# Casos de prueba
# Cada entrada: (id, estado_inicial, condiciones, alcance, mecanismo)
# ---------------------------------------------------------------------------
PRUEBAS = [
    (1, "100000000000000", "111111111101010", "111111111111100", "110110110110101"),
]

# ---------------------------------------------------------------------------
# Estrategias a ejecutar
# Cada entrada: (nombre_columna, clase, habilitada)
# ---------------------------------------------------------------------------
ESTRATEGIAS = [
    ("QNodes",     QNodes,       True),
    ("Geometrica", GeometricSIA, True),
    ("Pyphi",      Phi,          True),
]


def correr_estrategia(clase, tpm, estado_inicial, condiciones, alcance, mecanismo):
    try:
        instancia = clase(tpm)
        resultado = instancia.aplicar_estrategia(
            estado_inicial, condiciones, alcance, mecanismo
        )
        print(resultado)
    except Exception:
        traceback.print_exc()


def ejecutar():
    estrategias_activas = [(n, c) for n, c, hab in ESTRATEGIAS if hab]

    for prueba in PRUEBAS:
        _, estado_inicial, condiciones, alcance, mecanismo = prueba

        gestor = Manager(estado_inicial)
        tpm    = gestor.cargar_red()

        for nombre, clase in estrategias_activas:
            correr_estrategia(clase, tpm, estado_inicial, condiciones, alcance, mecanismo)


if __name__ == "__main__":
    ejecutar()
