from typing import List

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import GEOMETRIC_STRAREGY_TAG


class GeometricSIA(SIA):
    """
    Estrategia geométrica para encontrar la bipartición de mínima información (MIP).
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.n: int = 0
        self.m: int = 0
        self.tensors: List[np.ndarray] = []
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        
        raise NotImplementedError
    
    def _representacion_inicial(self) -> None:
        """
        Construye la representación tensorial del subsistema.

        Para cada n-cubo del subsistema aplana sus datos en un vector 1-D.
        Estos vectores representan geométricamente cada nodo en el espacio
        de probabilidad y sirven de base para calcular distancias entre nodos.
        """
        self.n = self.sia_subsistema.indices_ncubos.size
        self.m = self.sia_subsistema.dims_ncubos.size
        self.tensors = [
            self.sia_subsistema.ncubos[i].data.flatten()
            for i in range(self.n)
    ]
    
    def _hallar_candidatos(self):
        indices = self.sia_subsistema.indices_ncubos
        n = self.n
        for mask in range(1, (1 << n) // 2 + 1):
            subalcance = tuple(
                int(indices[i]) for i in range(n) if (mask >> i) & 1
            )
            submecanismo = tuple(
                int(indices[j]) for j in range(n) if not ((mask >> j) & 1)
            )
            yield subalcance, submecanismo
    
    def _calcular_costo_transicion(self, i: int, j: int, tensor: np.ndarray) -> float:
        """
        Calcula t_x(i,j) según la fórmula recursiva de GeoMIP (Algorithm 1).

        t_x(i,j) = y · ( |X[i] - X[j]| + Σ_{k ∈ N(i,j)} t_x(k, j) )
        y = 2^(-d_H(i,j))
        N(i,j) = vecinos v con d_H(i,v)=1 y d_H(v,j) < d_H(i,j)
        """
        if i == j:
            return 0.0

        d = bin(i ^ j).count('1')        # distancia Hamming
        gamma = 2.0 ** (-d)

        costo_directo = abs(float(tensor[i]) - float(tensor[j]))

        costo_vecinos = 0.0
        for bit in range(self.m):
            v = i ^ (1 << bit)           # flip de un bit
            if bin(v ^ j).count('1') < d:  # v está más cerca de j → es vecino válido
                costo_vecinos += self._calcular_costo_transicion(v, j, tensor)

        return gamma * (costo_directo + costo_vecinos)


    

