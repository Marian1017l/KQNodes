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
            self._tensor_de(i) for i in range(self.n)
        ]
    
    def _tensor_de(self, var_index:int) -> np.ndarray:
        ncube = self.sia_subsistema.ncubos[var_index]
        return ncube.data.flatten()

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
    
    def _construir_tabla_costos(self) -> List[np.ndarray]:
        """
        Construye la tabla T de costos usando DP bottom-up.
        Procesa pares (i,j) en orden creciente de distancia Hamming,
        garantizando que cada subproblema se calcula exactamente una vez.
        """
        num_estados = 1 << self.m
        tabla: List[np.ndarray] = []

        for x in range(self.n):
            T_x = np.zeros((num_estados, num_estados), dtype=np.float64)
            tensor = self.tensors[x]

            for j in range(num_estados):
                # Procesa distancias de menor a mayor → d=1 primero, d=m al final
                for d in range(1, self.m + 1):
                    gamma = 2.0 ** (-d)
                    for i in range(num_estados):
                        if bin(i ^ j).count('1') != d:
                            continue
                        costo_directo = abs(float(tensor[i]) - float(tensor[j]))
                        # Los vecinos (d-1) ya están calculados en T_x
                        costo_vecinos = sum(
                            T_x[i ^ (1 << bit), j]
                            for bit in range(self.m)
                            if bin((i ^ (1 << bit)) ^ j).count('1') < d
                        )
                        T_x[i, j] = gamma * (costo_directo + costo_vecinos)

            tabla.append(T_x)
        return tabla


    def _identificar_candidatos(self, tabla: List[np.ndarray]) -> list:
        """
        Identifica biparticiones candidatas usando patrones de costo máximo en T.

        Para cada variable x y cada estado j, el estado i con mayor costo
        t_x(i,j) indica qué bits del mecanismo deben separarse.
        La máscara (i XOR j) define el candidato a bipartición.
        """
        indices = self.sia_subsistema.indices_ncubos  # variables de alcance (futuro)
        dims    = self.sia_subsistema.dims_ncubos     # variables de mecanismo (presente)
        candidatos = set()

        for x in range(self.n):
            T_x = tabla[x]
            for j in range(1 << self.m):
                costos = T_x[:, j].copy()
                costos[j] = -1.0                      # excluir i == j
                i_max = int(np.argmax(costos))

                mascara = i_max ^ j                   # bits que más "cuestan" separar
                if mascara == 0:
                    continue

                # sub_alcance: la variable x actual
                # sub_mecanismo: las variables de presente marcadas por la máscara
                sub_alcance   = (int(indices[x]),)
                sub_mecanismo = tuple(
                    int(dims[b]) for b in range(self.m) if (mascara >> b) & 1
                )

                if sub_alcance and sub_mecanismo:
                    candidatos.add((sub_alcance, sub_mecanismo))

        return list(candidatos)



        

