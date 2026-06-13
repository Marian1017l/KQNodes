from src.controllers.manager import Manager
from src.strategies.force import BruteForce
from src.strategies.geometric import GeometricSIA
from src.strategies.q_nodes import QNodes
from src.strategies.kq_nodes import KQNodes


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

    analizador_geo = GeometricSIA(mpt)
    resultado_geo = analizador_geo.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(resultado_geo)

    analizador_q = QNodes(mpt)
    resultado_q = analizador_q.aplicar_estrategia(
        estado_inicial, condiciones, alcance, mecanismo,
    )
    print(resultado_q)

    analizador_kq = KQNodes(mpt)
    resultado_kq = analizador_kq.aplicar_estrategia(
        estado_inicial, condiciones, alcance, mecanismo, k=3,
    )
    print(resultado_kq)

if __name__ == "__main__":
    iniciar()