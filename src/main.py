from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.force import BruteForce
from src.strategies.geometric import GeometricSIA


def iniciar():
    """Punto de entrada"""

    # ABCD #
    estado_inicial = "1000"
    condiciones =    "1110"
    alcance =        "1110"
    mecanismo =      "1110"

    gestor_redes = Manager(estado_inicial)
    mpt = gestor_redes.cargar_red()

    analizador_bf = BruteForce(mpt)
    resultado_bf = analizador_bf.aplicar_estrategia(
        estado_inicial, condiciones, alcance, mecanismo,
    )
    print(resultado_bf)


    # Prueba GeometricSIA
    analizador_geo = GeometricSIA(mpt)
    resultado = analizador_geo.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(resultado)

if __name__ == "__main__":
    iniciar()