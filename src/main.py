from src.controllers.manager import Manager
from src.strategies.force import BruteForce
from src.strategies.geometric import GeometricSIA


def iniciar():
    """Punto de entrada"""

    # ABCD #
    estado_inicial = "100000"
    condiciones    = "111110"
    alcance        = "111110"
    mecanismo      = "111110"

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