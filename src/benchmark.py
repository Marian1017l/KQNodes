import traceback

from src.controllers.manager import Manager
from src.strategies.q_nodes import QNodes

# ---------------------------------------------------------------------------
# Casos de prueba
# Cada entrada: (id, estado_inicial, condiciones, alcance, mecanismo)
# ---------------------------------------------------------------------------
PRUEBAS = [
    (1, "10000000000000000000", "11111111111111111111", "11111111111111111100", "11011011011011011011"),
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

        asegurar_red(estado_inicial)
        gestor = Manager(estado_inicial)
        tpm    = gestor.cargar_red()

        for nombre, clase in estrategias_activas:
            correr_estrategia(clase, tpm, estado_inicial, condiciones, alcance, mecanismo)


if __name__ == "__main__":
    ejecutar()
